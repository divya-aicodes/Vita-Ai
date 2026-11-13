"""Tests for the dependency-light ML model registry and leaderboard reader."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vita_ai.ml.registry import MODEL_SPECS, get_model_spec  # noqa: E402
from vita_ai.ml.results import load_leaderboard  # noqa: E402


class ModelRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_the_approved_models(self) -> None:
        self.assertEqual(
            {item.key for item in MODEL_SPECS},
            {
                "efficientnet_v2_b0",
                "convnext_tiny",
                "densenet121",
                "mobilenet_v3_large",
                "resnet50_v2",
            },
        )
        self.assertEqual(len(MODEL_SPECS), 5)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            get_model_spec("vgg16")

    def test_leaderboard_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaderboard.json"
            self.assertIsNone(load_leaderboard(path))
            path.write_text(json.dumps({"best_model": "x", "models": []}), encoding="utf-8")
            self.assertEqual(load_leaderboard(path)["best_model"], "x")
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid structure"):
                load_leaderboard(path)


if __name__ == "__main__":
    unittest.main()
