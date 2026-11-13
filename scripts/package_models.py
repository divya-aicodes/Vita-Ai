"""Package trained ONNX models for a verified GitHub Release deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vita_ai.ml.inference import REQUIRED_ARTIFACTS  # noqa: E402
from vita_ai.ml.registry import MODEL_SPECS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_models(artifacts_dir: Path, output_dir: Path, release_tag: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for spec in MODEL_SPECS:
        model_dir = artifacts_dir / spec.key
        missing = [name for name in REQUIRED_ARTIFACTS if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{spec.key} is missing: {', '.join(missing)}")
        archive_path = output_dir / f"{spec.key}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in REQUIRED_ARTIFACTS:
                archive.write(model_dir / name, arcname=name)
        models[spec.key] = {
            "display_name": spec.display_name,
            "url": (
                "https://github.com/divya-aicodes/Vita-Ai/releases/download/"
                f"{release_tag}/{archive_path.name}"
            ),
            "sha256": sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        }
    return {"release_tag": release_tag, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "ml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "releases")
    parser.add_argument("--release-tag", default="plantvillage-v1")
    args = parser.parse_args()

    manifest = package_models(args.artifacts_dir, args.output_dir, args.release_tag)
    data_dir = SRC_DIR / "vita_ai" / "data"
    manifest_path = data_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    leaderboard_source = args.artifacts_dir / "leaderboard.json"
    if not leaderboard_source.is_file():
        raise FileNotFoundError("leaderboard.json is missing")
    shutil.copyfile(leaderboard_source, data_dir / "model_leaderboard.json")
    print(f"Packaged {len(manifest['models'])} models in {args.output_dir}")
    print(f"Wrote deployment manifest to {manifest_path}")


if __name__ == "__main__":
    main()
