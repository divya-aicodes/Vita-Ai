from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita_ai.image_quality import ImageValidationError, assess_photo, decode_image


class ImageQualityTests(unittest.TestCase):
    def test_decode_rejects_non_image(self) -> None:
        with self.assertRaises(ImageValidationError):
            decode_image(b"not really an image")

    def test_assessment_detects_flat_blurry_image(self) -> None:
        image = Image.new("RGB", (640, 640), (100, 145, 100))
        result = assess_photo(image)
        self.assertEqual(result.status, "poor")
        self.assertTrue(any("blurred" in issue or "contrast" in issue for issue in result.issues))

    def test_assessment_accepts_detailed_leaf_like_photo(self) -> None:
        y, x = np.indices((720, 720))
        checker = ((x // 12 + y // 12) % 2) * 35
        array = np.zeros((720, 720, 3), dtype=np.uint8)
        array[..., 0] = 45 + checker
        array[..., 1] = 125 + checker
        array[..., 2] = 35 + checker // 2
        result = assess_photo(Image.fromarray(array, "RGB"))
        self.assertIn(result.status, {"good", "usable"})
        self.assertGreater(result.leaf_coverage, 0.5)

    def test_png_round_trip(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (320, 320), (30, 140, 45)).save(buffer, format="PNG")
        decoded = decode_image(buffer.getvalue())
        self.assertEqual(decoded.size, (320, 320))
        self.assertEqual(decoded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
