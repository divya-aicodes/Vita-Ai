"""Load and validate the curated plant-health knowledge base."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class KnowledgeBaseError(RuntimeError):
    """Raised when the bundled knowledge base is inconsistent."""


@lru_cache(maxsize=1)
def load_conditions(path: Path | None = None) -> tuple[dict[str, Any], ...]:
    source = path or Path(__file__).with_name("data") / "conditions.json"
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"Unable to load condition data: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise KnowledgeBaseError("Condition data must be a non-empty list.")

    required = {
        "id",
        "crop",
        "name",
        "category",
        "overview",
        "symptoms",
        "environments",
        "distributions",
        "next_steps",
        "prevention",
        "urgency",
    }
    seen: set[str] = set()
    for index, item in enumerate(raw):
        missing = required - set(item) if isinstance(item, dict) else required
        if not isinstance(item, dict) or missing:
            raise KnowledgeBaseError(f"Condition entry {index} is missing: {sorted(missing)}")
        if item["id"] in seen:
            raise KnowledgeBaseError(f"Duplicate condition id: {item['id']}")
        if not item["symptoms"]:
            raise KnowledgeBaseError(f"Condition {item['id']} has no symptom evidence.")
        seen.add(item["id"])
    return tuple(raw)


def crops() -> tuple[str, ...]:
    return tuple(sorted({item["crop"] for item in load_conditions()}))
