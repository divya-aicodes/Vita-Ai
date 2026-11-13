"""Read and validate the generated ML leaderboard without importing TensorFlow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_leaderboard(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("ML leaderboard has an invalid structure.")
    return payload
