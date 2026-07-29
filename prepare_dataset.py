"""Download and prepare the official PlantVillage colour dataset for Vita AI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import httpx
from huggingface_hub import hf_hub_download


REPOSITORY = "mohanty/PlantVillage"
REVISION = "main"
EXPECTED_CLASSES = 38
ARCHIVE_SIZE = 2_184_723_441
ARCHIVE_SHA256 = "fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84"
ARCHIVE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install PlantVillage using the official leaf-grouped train/test split."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "plantvillage",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "_source",
    )
    return parser.parse_args()


def model_class_name(source_name: str) -> str:
    """Convert upstream PlantVillage labels to Vita AI's model label spelling."""
    name = source_name
    name = name.replace("Cherry_(including_sour)___", "Cherry___")
    name = name.replace("Corn_(maize)___", "Corn___")
    name = name.replace("Cercospora_leaf_spot Gray_leaf_spot", "Cercospora_leaf_spot_Gray_leaf_spot")
    name = name.replace("Common_rust_", "Common_rust")
    name = name.replace(
        "Spider_mites Two-spotted_spider_mite",
        "Spider_mites_Two-spotted_spider_mite",
    )
    if name.endswith("___healthy"):
        name = f"{name[:-len('healthy')]}Healthy"
    return name


def download(filename: str, cache: Path) -> Path:
    print(f"Downloading {filename} ...", flush=True)
    for attempt in range(1, 11):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    filename=filename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=cache,
                )
            )
        except Exception as error:
            if attempt == 10:
                raise
            print(
                f"Download interrupted ({type(error).__name__}); "
                f"resuming, attempt {attempt + 1}/10 ...",
                flush=True,
            )
            time.sleep(5)


def read_split(path: Path) -> list[str]:
    paths = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [path for path in paths if path]


def download_archive(cache: Path) -> Path:
    """Download the large archive with durable byte-range resume support."""
    target = cache / "data.zip"
    partial = cache / "data.zip.part"
    if target.is_file() and target.stat().st_size == ARCHIVE_SIZE:
        return target
    if partial.is_file() and partial.stat().st_size > ARCHIVE_SIZE:
        partial.unlink()

    cache.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(connect=30, read=60, write=60, pool=30)
    last_reported = -1
    for attempt in range(1, 31):
        downloaded = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        mode = "ab" if downloaded else "wb"
        try:
            with httpx.stream(
                "GET",
                ARCHIVE_URL,
                headers=headers,
                follow_redirects=True,
                timeout=timeout,
            ) as response:
                if downloaded and response.status_code != 206:
                    raise RuntimeError(
                        f"Server did not honor resume request (HTTP {response.status_code})."
                    )
                response.raise_for_status()
                with partial.open(mode) as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        progress = int(downloaded * 100 / ARCHIVE_SIZE)
                        if progress // 5 != last_reported // 5:
                            print(
                                f"Archive download: {downloaded / 1_000_000_000:.2f}/"
                                f"{ARCHIVE_SIZE / 1_000_000_000:.2f} GB ({progress}%)",
                                flush=True,
                            )
                            last_reported = progress
            if partial.stat().st_size == ARCHIVE_SIZE:
                break
            raise RuntimeError(
                f"Incomplete response: {partial.stat().st_size}/{ARCHIVE_SIZE} bytes"
            )
        except Exception as error:
            if attempt == 30:
                raise
            print(
                f"Archive connection interrupted ({type(error).__name__}); "
                f"resuming, attempt {attempt + 1}/30 ...",
                flush=True,
            )
            time.sleep(3)
    else:
        raise RuntimeError("Archive download did not complete.")

    print("Verifying archive checksum ...", flush=True)
    digest = hashlib.sha256()
    with partial.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("The downloaded archive failed its SHA-256 integrity check.")
    partial.replace(target)
    return target


def resolve_members(archive: zipfile.ZipFile, wanted: set[str]) -> dict[str, str]:
    """Map requested upstream paths to archive member names, tolerating a root prefix."""
    members: dict[str, str] = {}
    for member in archive.namelist():
        normalized = member.replace("\\", "/")
        for marker in ("raw/color/",):
            index = normalized.find(marker)
            if index >= 0:
                relative = normalized[index:]
                if relative in wanted:
                    members[relative] = member
                break
    return members


def extract_split(
    archive: zipfile.ZipFile,
    members: dict[str, str],
    entries: list[str],
    split_name: str,
    destination: Path,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    total = len(entries)
    for index, source_path in enumerate(entries, start=1):
        parts = PurePosixPath(source_path).parts
        source_class = parts[2]
        target_class = model_class_name(source_class)
        target = destination / split_name / target_class / parts[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        member = members[source_path]
        expected_size = archive.getinfo(member).file_size
        if not target.is_file() or target.stat().st_size != expected_size:
            temporary = target.with_suffix(f"{target.suffix}.part")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)
        counts[target_class] += 1
        if index % 1000 == 0 or index == total:
            print(f"{split_name}: {index:,}/{total:,} images prepared", flush=True)
    return counts


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.source_cache.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": download("splits/color_train.txt", args.source_cache),
        "test": download("splits/color_test.txt", args.source_cache),
    }
    entries = {name: read_split(path) for name, path in split_paths.items()}

    project_root = Path(__file__).resolve().parent
    labels_path = project_root / "models" / "class_names.json"
    model_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    upstream_classes = {
        model_class_name(PurePosixPath(item).parts[2])
        for split_entries in entries.values()
        for item in split_entries
    }
    if len(upstream_classes) != EXPECTED_CLASSES or upstream_classes != set(model_labels):
        missing = sorted(set(model_labels) - upstream_classes)
        extra = sorted(upstream_classes - set(model_labels))
        raise SystemExit(f"Class mismatch. Missing={missing}; extra={extra}")

    print("Downloading data.zip with resume support ...", flush=True)
    archive_path = download_archive(args.source_cache)
    wanted = set(entries["train"]) | set(entries["test"])
    print("Indexing colour images in the archive ...", flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = resolve_members(archive, wanted)
        missing_members = wanted - set(members)
        if missing_members:
            sample = sorted(missing_members)[:5]
            raise SystemExit(f"Archive is missing {len(missing_members)} images; examples: {sample}")
        counts = {
            split: extract_split(archive, members, split_entries, split, args.destination)
            for split, split_entries in entries.items()
        }

    manifest = {
        "dataset": "PlantVillage",
        "source": f"https://huggingface.co/datasets/{REPOSITORY}",
        "revision": REVISION,
        "configuration": "color",
        "split_method": "official leaf-grouped train/test split",
        "classes": model_labels,
        "counts": {
            split: {"total": sum(class_counts.values()), "by_class": dict(class_counts)}
            for split, class_counts in counts.items()
        },
    }
    manifest_path = args.destination / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset ready at {args.destination}", flush=True)
    print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
