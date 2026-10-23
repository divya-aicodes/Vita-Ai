from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita_ai.screening import screen_case
from vita_ai.storage import list_screenings, save_feedback, save_screening


class StorageTests(unittest.TestCase):
    def test_screening_and_feedback_round_trip(self) -> None:
        workspace_temp = ROOT / ".test-tmp"
        workspace_temp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_temp) as directory:
            path = Path(directory) / "test.db"
            result = screen_case(
                crop="Tomato",
                symptoms=["concentric_rings", "dark_spots", "yellow_halo"],
                environments=["warm", "humid"],
                distribution="lower_leaves",
            )
            case_id = save_screening(
                crop="Tomato",
                symptoms=["concentric_rings", "dark_spots", "yellow_halo"],
                environments=["warm", "humid"],
                distribution="lower_leaves",
                notes="Test case",
                result=result,
                path=path,
            )
            save_feedback(case_id, "helpful", path=path)
            rows = list_screenings(path=path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], case_id)
            self.assertEqual(rows[0]["verdict"], "helpful")
            self.assertEqual(rows[0]["primary_condition"], "Early blight pattern")


if __name__ == "__main__":
    unittest.main()
