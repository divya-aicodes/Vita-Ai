"""Prepare leakage-aware train/validation/test folders from PlantVillage.

The official authors' train/test manifests are preserved. Validation groups are selected only
from the official training side and grouped by the supplied leaf map whenever an identifier exists.
Files are hard-linked, so the prepared split does not duplicate image bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

SOURCE_URL = "https://huggingface.co/datasets/mohanty/PlantVillage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/") for line in path.read_text().splitlines() if line.strip()
    ]


def extract_color_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = [item for item in handle.infolist() if item.filename.startswith("raw/color/")]
        for index, member in enumerate(members, start=1):
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if index % 5000 == 0:
                print(f"Extracted {index:,}/{len(members):,} color entries")
    marker.write_text("complete\n", encoding="utf-8")


def leaf_group(relative_path: str, leaf_map: dict[str, list[str]]) -> str:
    class_name = relative_path.split("/")[2]
    filename = relative_path.split("/")[-1]
    identifier = filename.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.split("copy")[0]
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        identifier = identifier.replace(suffix, "")
    identifier = identifier.strip()
    suggestions = leaf_map.get(identifier.casefold().strip(), [])
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback::{relative_path}"


def validation_paths(
    training_paths: list[str], leaf_map: dict[str, list[str]], fraction: float, seed: int
) -> set[str]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for path in training_paths:
        by_class[path.split("/")[2]].append(path)

    selected: set[str] = set()
    for class_index, (class_name, paths) in enumerate(sorted(by_class.items())):
        groups: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            groups[leaf_group(path, leaf_map)].append(path)
        group_names = sorted(groups)
        random.Random(seed + class_index).shuffle(group_names)
        target = max(1, round(len(paths) * fraction))
        class_validation: list[str] = []
        for group_name in group_names[:-1]:
            class_validation.extend(groups[group_name])
            if len(class_validation) >= target:
                break
        if not class_validation:
            class_validation.extend(groups[group_names[0]])
        selected.update(class_validation)
        print(
            f"{class_name}: {len(paths) - len(class_validation):,} train, "
            f"{len(class_validation):,} validation"
        )
    return selected


def materialize(paths: list[str], source_root: Path, split_root: Path) -> None:
    for relative_path in paths:
        source = source_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {source}")
        class_name = relative_path.split("/")[2]
        target = split_root / class_name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4:
        raise SystemExit("validation-fraction must be between 0.05 and 0.4")
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    archive = source_dir / "data.zip"
    train_manifest = source_dir / "splits" / "color_train.txt"
    test_manifest = source_dir / "splits" / "color_test.txt"
    leaf_map_path = source_dir / "leaf_grouping" / "leaf-map.json"
    for required in (archive, train_manifest, test_manifest, leaf_map_path):
        if not required.is_file():
            raise SystemExit(f"Missing PlantVillage source file: {required}")

    extracted = source_dir / "extracted"
    extract_color_archive(archive, extracted)
    official_train = read_manifest(train_manifest)
    official_test = read_manifest(test_manifest)
    leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    validation = validation_paths(official_train, leaf_map, args.validation_fraction, args.seed)
    training = [path for path in official_train if path not in validation]
    validation_list = [path for path in official_train if path in validation]

    materialize(training, extracted, output_dir / "train")
    materialize(validation_list, extracted, output_dir / "val")
    materialize(official_test, extracted, output_dir / "test")

    classes = sorted({path.split("/")[2] for path in official_train + official_test})
    metadata = {
        "dataset": "PlantVillage color",
        "source": SOURCE_URL,
        "classes": classes,
        "class_count": len(classes),
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "counts": {
            "train": len(training),
            "val": len(validation_list),
            "test": len(official_test),
        },
        "split_policy": (
            "Authors' official test split is locked. Validation is selected from the official "
            "training split by leaf group where identifiers are available."
        ),
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
