"""Domain models used by the screening, reporting, and storage layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PhotoAssessment:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    leaf_coverage: float
    yellow_ratio: float
    brown_ratio: float
    pale_ratio: float
    quality_score: float
    status: str
    issues: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    condition_id: str
    name: str
    crop: str
    category: str
    match_score: float
    matched_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    overview: str
    next_steps: tuple[str, ...]
    prevention: tuple[str, ...]
    urgency: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScreeningResult:
    status: str
    headline: str
    explanation: str
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    photo: PhotoAssessment | None = None
    evidence_count: int = 0
    method: str = "rule"

    @property
    def primary(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "headline": self.headline,
            "explanation": self.explanation,
            "candidates": [item.as_dict() for item in self.candidates],
            "photo": self.photo.as_dict() if self.photo else None,
            "evidence_count": self.evidence_count,
            "method": self.method,
        }
