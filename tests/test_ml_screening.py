from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita_ai.domain import PhotoAssessment  # noqa: E402
from vita_ai.ml.inference import DiseasePrediction  # noqa: E402
from vita_ai.ml.screening import build_ml_screening  # noqa: E402


class MLScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.photo = PhotoAssessment(
            width=256,
            height=256,
            brightness=120,
            contrast=40,
            sharpness=80,
            leaf_coverage=0.8,
            yellow_ratio=0.1,
            brown_ratio=0.1,
            pale_ratio=0.1,
            quality_score=0.9,
            status="good",
        )

    def test_builds_result_directly_from_image_prediction(self) -> None:
        result = build_ml_screening(
            (
                DiseasePrediction("Apple___Apple_scab", "Apple", "Apple scab", 0.98),
                DiseasePrediction("Apple___healthy", "Apple", "healthy", 0.01),
            ),
            photo=self.photo,
            model_name="ResNet50V2",
        )
        self.assertEqual(result.method, "ml")
        self.assertEqual(result.status, "ml_screened")
        self.assertEqual(result.primary.name, "Apple scab")
        self.assertEqual(result.primary.category, "fungal")
        self.assertAlmostEqual(result.primary.match_score, 0.98)

    def test_low_confidence_prediction_requires_review(self) -> None:
        result = build_ml_screening(
            (DiseasePrediction("Tomato___Early_blight", "Tomato", "Early blight", 0.4),),
            photo=self.photo,
            model_name="MobileNetV3Large",
        )
        self.assertEqual(result.status, "needs_review")


if __name__ == "__main__":
    unittest.main()
