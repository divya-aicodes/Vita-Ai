"""Download the optional documented third-party model from Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import CLASS_NAMES, LABELS_PATH, METADATA_PATH, MODEL_PATH, ensure_runtime_directories

REPO_ID = "rarfileexe/Plant-Disease-Detector"
FILENAME = "model_4_mobilenet_finetuned.keras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install a compatible 38-class MobileNetV2 model.")
    parser.add_argument("--force", action="store_true", help="Replace an existing local model.")
    return parser.parse_args()


def _metadata() -> dict[str, object]:
    return {
        "model_version": "rarfileexe-plantvillage-mobilenetv2",
        "model_name": "MobileNetV2 fine-tuned on PlantVillage",
        "input_width": 224,
        "input_height": 224,
        "preprocessing": "embedded",
        "last_conv_layer": "out_relu",
        "source": f"https://huggingface.co/{REPO_ID}",
    }


def install_model(force: bool = False) -> Path:
    """Install the public model and its matching metadata, returning its local path."""
    ensure_runtime_directories()
    if MODEL_PATH.exists() and not force:
        return MODEL_PATH

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is missing. Install project requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    temporary_path = MODEL_PATH.with_suffix(f"{MODEL_PATH.suffix}.part")
    shutil.copy2(downloaded, temporary_path)
    temporary_path.replace(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(_metadata(), indent=2), encoding="utf-8")
    return MODEL_PATH


def main() -> None:
    args = parse_args()
    already_installed = MODEL_PATH.exists() and not args.force
    path = install_model(force=args.force)
    if already_installed:
        print(f"A model is already installed at {path}")
        return

    print(f"Installed model at {path}")
    print("Source: https://huggingface.co/rarfileexe/Plant-Disease-Detector")
    print("License: MIT. Accuracy claims are third-party/self-reported; evaluate locally before relying on them.")


if __name__ == "__main__":
    main()
