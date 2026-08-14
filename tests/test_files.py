from __future__ import annotations

import json
import unittest

from PIL import Image

from src.config import ASSETS_DIR, CLASS_NAMES, LABELS_PATH, METADATA_PATH


class ProjectFileTests(unittest.TestCase):
    def test_label_file_matches_default_order(self):
        self.assertEqual(json.loads(LABELS_PATH.read_text(encoding="utf-8")), CLASS_NAMES)

    def test_metadata_has_required_fields(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        for key in (
            "model_version",
            "model_name",
            "input_width",
            "input_height",
            "preprocessing",
            "last_conv_layer",
            "source",
        ):
            self.assertIn(key, metadata)

    def test_diagnostic_hero_asset_is_web_ready(self):
        path = ASSETS_DIR / "vita-ai-diagnostic-hero.jpg"
        self.assertTrue(path.exists())
        self.assertLess(path.stat().st_size, 250_000)
        with Image.open(path) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertGreaterEqual(image.width, 1200)
            self.assertAlmostEqual(image.width / image.height, 16 / 9, delta=0.03)


if __name__ == "__main__":
    unittest.main()
