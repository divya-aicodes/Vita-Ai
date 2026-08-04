from __future__ import annotations

import json
import unittest

from src.config import CLASS_NAMES, LABELS_PATH, METADATA_PATH


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


if __name__ == "__main__":
    unittest.main()
