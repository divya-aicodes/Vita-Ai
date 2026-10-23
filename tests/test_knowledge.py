from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita_ai.knowledge import crops, load_conditions


class KnowledgeTests(unittest.TestCase):
    def test_knowledge_base_is_nonempty_and_unique(self) -> None:
        conditions = load_conditions()
        identifiers = [item["id"] for item in conditions]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(len(conditions), 20)
        self.assertGreaterEqual(len(crops()), 8)


if __name__ == "__main__":
    unittest.main()
