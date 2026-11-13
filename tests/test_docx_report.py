from __future__ import annotations

import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docx import Document  # noqa: E402

from vita_ai.docx_report import build_docx_report  # noqa: E402
from vita_ai.domain import PhotoAssessment  # noqa: E402
from vita_ai.ml.inference import DiseasePrediction  # noqa: E402
from vita_ai.ml.screening import build_ml_screening  # noqa: E402


class DocxReportTests(unittest.TestCase):
    def test_generates_valid_editable_word_report(self) -> None:
        photo = PhotoAssessment(
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
        result = build_ml_screening(
            (
                DiseasePrediction("Apple___Apple_scab", "Apple", "Apple scab", 0.9847),
                DiseasePrediction("Potato___healthy", "Potato", "healthy", 0.0024),
                DiseasePrediction("Strawberry___Leaf_scorch", "Strawberry", "Leaf scorch", 0.0019),
            ),
            photo=photo,
            model_name="ResNet50V2",
        )
        payload = build_docx_report(
            case_id="sample-case-1234",
            crop="Apple",
            symptoms=[],
            environments=[],
            distribution="image_only",
            notes="ML model: ResNet50V2",
            result=result,
        )
        self.assertTrue(payload.startswith(b"PK"))
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("word/document.xml", archive.namelist())
        document = Document(BytesIO(payload))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("Vita AI Plant Health Screening Report", text)
        self.assertIn("Limitations and safety", text)
        self.assertEqual(document.tables[1].cell(1, 2).text, "Apple scab")


if __name__ == "__main__":
    unittest.main()
