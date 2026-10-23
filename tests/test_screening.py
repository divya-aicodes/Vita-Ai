from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita_ai.screening import screen_case


class ScreeningTests(unittest.TestCase):
    def test_tomato_early_blight_has_clear_match(self) -> None:
        result = screen_case(
            crop="Tomato",
            symptoms=["concentric_rings", "dark_spots", "yellow_halo"],
            environments=["warm", "humid"],
            distribution="lower_leaves",
        )
        self.assertEqual(result.status, "screened")
        self.assertEqual(result.primary.condition_id, "tomato-early-blight")
        self.assertGreaterEqual(result.primary.match_score, 0.70)

    def test_late_blight_is_urgent(self) -> None:
        result = screen_case(
            crop="Potato",
            symptoms=["water_soaked", "gray_growth", "wilting"],
            environments=["cool", "humid", "recent_rain"],
            distribution="whole_plant",
        )
        self.assertEqual(result.status, "screened")
        self.assertEqual(result.primary.condition_id, "potato-late-blight")
        self.assertEqual(result.primary.urgency, "urgent")

    def test_one_vague_symptom_abstains(self) -> None:
        result = screen_case(
            crop="Tomato",
            symptoms=["yellowing"],
            environments=[],
            distribution="whole_plant",
        )
        self.assertEqual(result.status, "needs_review")

    def test_missing_symptoms_is_insufficient(self) -> None:
        result = screen_case(
            crop="Apple",
            symptoms=[],
            environments=["humid"],
            distribution="scattered",
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.primary)

    def test_crop_scope_prevents_cross_crop_candidate(self) -> None:
        result = screen_case(
            crop="Cucumber",
            symptoms=["white_powder", "yellowing", "brown_edges"],
            environments=["humid", "warm"],
            distribution="lower_leaves",
        )
        self.assertTrue(all(candidate.crop == "Cucumber" for candidate in result.candidates))
        self.assertEqual(result.primary.condition_id, "cucumber-powdery-mildew")


if __name__ == "__main__":
    unittest.main()
