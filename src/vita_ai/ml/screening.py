"""Convert image-model predictions into a conservative screening result."""

from __future__ import annotations

import re

from vita_ai.domain import Candidate, PhotoAssessment, ScreeningResult
from vita_ai.knowledge import load_conditions

from .inference import DiseasePrediction


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _knowledge_crop(crop: str) -> str:
    normalized = _normalized(crop)
    aliases = {
        "corn maize": "Corn",
        "pepper bell": "Pepper",
    }
    return aliases.get(normalized, crop)


def _condition_entry(prediction: DiseasePrediction) -> dict | None:
    crop = _knowledge_crop(prediction.crop)
    condition = _normalized(prediction.condition)
    for entry in load_conditions():
        if entry["crop"] != crop:
            continue
        entry_name = _normalized(entry["name"].removesuffix(" pattern"))
        if condition == "healthy" and entry["category"] == "healthy":
            return entry
        if condition in entry_name or entry_name in condition:
            return entry
    return None


def build_ml_screening(
    predictions: tuple[DiseasePrediction, ...],
    *,
    photo: PhotoAssessment,
    model_name: str,
) -> ScreeningResult:
    """Build a display/storage result without inventing symptom evidence."""
    if not predictions:
        raise ValueError("At least one ML prediction is required.")

    candidates = []
    for prediction in predictions:
        entry = _condition_entry(prediction)
        is_healthy = _normalized(prediction.condition) == "healthy"
        candidates.append(
            Candidate(
                condition_id=prediction.class_name,
                name=prediction.condition,
                crop=prediction.crop,
                category=entry["category"] if entry else ("healthy" if is_healthy else "ml class"),
                match_score=prediction.confidence,
                matched_evidence=(f"{model_name} ranked this image as {prediction.class_name}.",),
                missing_evidence=(
                    "No symptom questionnaire was used; confirm the result on multiple leaves.",
                ),
                overview=(
                    entry["overview"]
                    if entry
                    else "This is an image-model class from the controlled PlantVillage dataset."
                ),
                next_steps=tuple(
                    entry["next_steps"]
                    if entry
                    else (
                        "Inspect several leaves and both leaf surfaces for the same pattern.",
                        "Seek local expert confirmation before treatment or crop removal.",
                    )
                ),
                prevention=tuple(
                    entry["prevention"]
                    if entry
                    else (
                        "Keep tools clean and avoid moving suspect plant material.",
                        "Monitor nearby plants and record whether symptoms spread.",
                    )
                ),
                urgency=entry["urgency"] if entry else "review",
            )
        )

    primary = predictions[0]
    confident = primary.confidence >= 0.65
    return ScreeningResult(
        status="ml_screened" if confident else "needs_review",
        headline=f"{primary.crop} · {primary.condition}",
        explanation=(
            f"{model_name} assigned {primary.confidence:.1%} confidence to its top PlantVillage "
            "class. This controlled-dataset result is a screening aid, not a field diagnosis."
        ),
        candidates=tuple(candidates),
        photo=photo,
        evidence_count=1,
        method="ml",
    )
