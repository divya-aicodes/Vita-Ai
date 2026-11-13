"""Tests for dependency-light ONNX preprocessing and artifact discovery."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vita_ai.ml.inference import (  # noqa: E402
    ModelUnavailableError,
    _safe_extract,
    available_models,
    preprocess_image,
)


class ModelInferenceTests(unittest.TestCase):
    def test_preprocessing_matches_expected_shape_and_type(self) -> None:
        image = Image.new("RGB", (320, 240), color=(128, 64, 32))
        result = preprocess_image(
            image,
            {
                "input_size": [3, 224, 224],
                "crop_pct": 0.875,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_only_complete_known_artifacts_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "efficientnet_v2_b0"
            complete.mkdir()
            for name in ("model.onnx", "labels.json", "metrics.json"):
                (complete / name).touch()
            incomplete = root / "convnext_tiny"
            incomplete.mkdir()
            (incomplete / "model.onnx").touch()
            unknown = root / "unknown"
            unknown.mkdir()
            for name in ("model.onnx", "labels.json", "metrics.json"):
                (unknown / name).touch()
            self.assertEqual(available_models(root), ("efficientnet_v2_b0",))

    def test_manifest_models_are_available_in_registry_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": {
                            "mobilenet_v3_large": {},
                            "efficientnet_v2_b0": {},
                            "unknown": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                available_models(root / "cache", manifest),
                ("efficientnet_v2_b0", "mobilenet_v3_large"),
            )

    def test_archive_extraction_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "model.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in ("model.onnx", "labels.json", "metrics.json"):
                    archive.writestr(name, "test")
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ModelUnavailableError):
                _safe_extract(archive_path, root / "extracted")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
