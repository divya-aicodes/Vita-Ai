from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from download_model import sha256_file
from src.config import CLASS_NAMES, ModelMetadata
from src.database import PredictionDatabase
from src.disease_info import (
    get_disease_info,
    library_conditions,
    parse_class_label,
    reference_conditions,
)
from src.prediction import confidence_message, predict_image
from src.preprocessing import ImageValidationError, load_image, prepare_image
from src.quality_checker import assess_image_quality


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities], dtype=np.float32)

    def predict(self, batch, verbose=0):
        if batch.shape != (1, 224, 224, 3):
            raise AssertionError(f"Unexpected batch shape: {batch.shape}")
        return self.probabilities


class ImageTests(unittest.TestCase):
    def test_load_image_converts_to_rgb(self):
        rgba = Image.new("RGBA", (300, 250), (70, 150, 50, 128))
        loaded = load_image(image_bytes(rgba))
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.size, (300, 250))

    def test_invalid_upload_rejected(self):
        with self.assertRaises(ImageValidationError):
            load_image(b"not an image")

    def test_preprocessing_shape_and_dtype(self):
        batch = prepare_image(Image.new("RGB", (320, 260), "green"))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)

    def test_quality_checker_detects_flat_image(self):
        result = assess_image_quality(Image.new("RGB", (400, 400), (2, 2, 2)))
        self.assertFalse(result.acceptable)
        self.assertGreater(len(result.warnings), 0)

    def test_quality_checker_accepts_detailed_image(self):
        y, x = np.indices((400, 400))
        array = np.zeros((400, 400, 3), dtype=np.uint8)
        array[..., 0] = (x * 7 + y * 3) % 255
        array[..., 1] = (x * 3 + y * 11) % 255
        array[..., 2] = (x * 13 + y * 5) % 255
        result = assess_image_quality(Image.fromarray(array))
        self.assertTrue(result.acceptable)
        self.assertGreater(result.score, 50)


class KnowledgeTests(unittest.TestCase):
    def test_readable_label(self):
        crop, condition, status = parse_class_label("Tomato___Early_blight")
        self.assertEqual((crop, condition, status), ("Tomato", "Early Blight", "Diseased"))

    def test_healthy_guidance(self):
        info = get_disease_info("Potato___Healthy")
        self.assertEqual(info.status, "Healthy")
        self.assertIn("screening", info.expert_warning)

    def test_professional_library_scope(self):
        items = library_conditions(CLASS_NAMES)
        self.assertEqual(len(items), 70)
        self.assertEqual(sum(item.model_supported for item in items), 38)
        self.assertEqual(sum(not item.model_supported for item in items), 32)
        self.assertEqual(
            len({(item.crop.casefold(), item.condition.casefold()) for item in items}),
            len(items),
        )

    def test_reference_entries_are_not_prediction_classes(self):
        items = reference_conditions()
        self.assertTrue(items)
        self.assertTrue(all(not item.model_supported for item in items))
        self.assertIn("Citrus Canker", {item.condition for item in items})


class PredictionTests(unittest.TestCase):
    def test_ranking_and_confidence(self):
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        probabilities[29] = 0.91
        probabilities[30] = 0.06
        probabilities[28] = 0.03
        result = predict_image(
            FakeModel(probabilities),
            Image.new("RGB", (300, 300), "green"),
            CLASS_NAMES,
            ModelMetadata(),
        )
        self.assertEqual(result.primary.condition, "Early Blight")
        self.assertEqual(result.confidence_level, "High")
        self.assertEqual(len(result.alternatives), 2)

    def test_crop_constraint_prevents_cross_plant_result(self):
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        probabilities[5] = 0.95  # Cherry powdery mildew
        probabilities[29] = 0.03  # Tomato early blight
        probabilities[30] = 0.02  # Tomato late blight
        result = predict_image(
            FakeModel(probabilities),
            Image.new("RGB", (300, 300), "green"),
            CLASS_NAMES,
            ModelMetadata(),
            expected_crop="Tomato",
        )
        self.assertEqual(result.selected_crop, "Tomato")
        self.assertTrue(
            all(item.crop == "Tomato" for item in (result.primary, *result.alternatives))
        )
        self.assertAlmostEqual(result.crop_support, 0.05, places=5)
        self.assertFalse(result.reliable)
        self.assertEqual(result.confidence_level, "Unreliable")

    def test_crop_constraint_accepts_clear_within_plant_match(self):
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        probabilities[0] = 0.28
        probabilities[29] = 0.70
        probabilities[30] = 0.02
        result = predict_image(
            FakeModel(probabilities),
            Image.new("RGB", (300, 300), "green"),
            CLASS_NAMES,
            ModelMetadata(),
            expected_crop="tomato",
        )
        self.assertEqual(result.primary.condition, "Early Blight")
        self.assertAlmostEqual(result.primary.confidence, 70 / 72, places=5)
        self.assertAlmostEqual(result.crop_support, 0.72, places=5)
        self.assertTrue(result.reliable)

    def test_crop_constraint_rejects_ambiguous_disease_match(self):
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        probabilities[0] = 0.25
        probabilities[29] = 0.40
        probabilities[30] = 0.35
        result = predict_image(
            FakeModel(probabilities),
            Image.new("RGB", (300, 300), "green"),
            CLASS_NAMES,
            ModelMetadata(),
            expected_crop="Tomato",
        )
        self.assertFalse(result.reliable)
        self.assertLess(result.primary.confidence, 0.55)

    def test_crop_constraint_rejects_unsupported_crop(self):
        probabilities = np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unsupported crop"):
            predict_image(
                FakeModel(probabilities),
                Image.new("RGB", (300, 300), "green"),
                CLASS_NAMES,
                ModelMetadata(),
                expected_crop="Banana",
            )

    def test_confidence_boundaries(self):
        self.assertEqual(confidence_message(0.80)[0], "High")
        self.assertEqual(confidence_message(0.55)[0], "Moderate")
        self.assertEqual(confidence_message(0.54)[0], "Low")


class DatabaseTests(unittest.TestCase):
    def test_save_feedback_and_analytics(self):
        with tempfile.TemporaryDirectory() as folder:
            database = PredictionDatabase(Path(folder) / "test.db")
            prediction_id = database.save_prediction(
                crop_name="Tomato",
                disease_name="Early Blight",
                health_status="Diseased",
                confidence=0.91,
                alternatives=["Tomato — Late Blight", "Tomato — Target Spot"],
                image_quality_score=88,
                inference_time_ms=42,
                model_version="test-model",
            )
            self.assertTrue(database.update_feedback(prediction_id, "Correct"))
            rows = database.recent_predictions()
            self.assertEqual(rows[0]["feedback"], "Correct")
            self.assertEqual(database.analytics()["summary"]["total"], 1)
            self.assertEqual(database.clear_history(), 1)

    def test_failed_transaction_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as folder:
            database = PredictionDatabase(Path(folder) / "test.db")
            with self.assertRaises(RuntimeError):
                with database.connection() as connection:
                    connection.execute(
                        """
                        INSERT INTO predictions (
                            created_at, crop_name, disease_name, health_status, confidence,
                            image_quality_score, inference_time_ms, model_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "2026-08-04T00:00:00+00:00",
                            "Tomato",
                            "Healthy",
                            "Healthy",
                            0.9,
                            90,
                            10,
                            "test",
                        ),
                    )
                    raise RuntimeError("simulate a failed request")
            self.assertEqual(database.analytics()["summary"]["total"], 0)


class ModelInstallerTests(unittest.TestCase):
    def test_streaming_sha256(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "artifact.bin"
            artifact.write_bytes(b"vita-ai")
            self.assertEqual(
                sha256_file(artifact),
                "a62fb49591cdf7a5b75ac1bc84e1119cf380b996bd1fb570f57028f0e1ba99ab",
            )


if __name__ == "__main__":
    unittest.main()
