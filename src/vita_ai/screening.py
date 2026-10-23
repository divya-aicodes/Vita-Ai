"""Transparent symptom-led screening engine with conservative abstention."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .config import DISTRIBUTIONS, ENVIRONMENTS, SYMPTOMS
from .domain import Candidate, PhotoAssessment, ScreeningResult
from .knowledge import load_conditions


def _label(key: str) -> str:
    return SYMPTOMS.get(key) or ENVIRONMENTS.get(key) or DISTRIBUTIONS.get(key) or key


def _score_condition(
    condition: dict[str, Any],
    symptoms: set[str],
    environments: set[str],
    distribution: str,
    photo: PhotoAssessment | None,
) -> Candidate:
    symptom_weights = {key: float(value) for key, value in condition["symptoms"].items()}
    total_weight = sum(symptom_weights.values())
    matched = symptoms & symptom_weights.keys()
    symptom_score = sum(symptom_weights[key] for key in matched) / total_weight

    excluded = symptoms & set(condition.get("conflicts", []))
    symptom_score = max(0.0, symptom_score - 0.16 * len(excluded))

    expected_environments = set(condition["environments"])
    if environments and expected_environments:
        environment_score = len(environments & expected_environments) / len(expected_environments)
    else:
        environment_score = 0.35

    distribution_score = 1.0 if distribution in condition["distributions"] else 0.15
    expected_visual = set(condition.get("visual_observations", []))
    if photo and expected_visual:
        visual_score = len(set(photo.observations) & expected_visual) / len(expected_visual)
    else:
        visual_score = 0.0

    score = (
        symptom_score * 0.72
        + min(environment_score, 1.0) * 0.13
        + distribution_score * 0.10
        + visual_score * 0.05
    )
    matched_evidence = tuple(_label(key) for key in sorted(matched))
    missing = sorted(set(symptom_weights) - symptoms, key=lambda key: -symptom_weights[key])[:3]
    return Candidate(
        condition_id=condition["id"],
        name=condition["name"],
        crop=condition["crop"],
        category=condition["category"],
        match_score=round(min(score, 1.0), 3),
        matched_evidence=matched_evidence,
        missing_evidence=tuple(_label(key) for key in missing),
        overview=condition["overview"],
        next_steps=tuple(condition["next_steps"]),
        prevention=tuple(condition["prevention"]),
        urgency=condition["urgency"],
    )


def screen_case(
    *,
    crop: str,
    symptoms: Iterable[str],
    environments: Iterable[str],
    distribution: str,
    photo: PhotoAssessment | None = None,
    conditions: tuple[dict[str, Any], ...] | None = None,
) -> ScreeningResult:
    """Rank candidates and abstain when the supplied evidence is weak or contradictory."""

    symptom_set = set(symptoms)
    environment_set = set(environments)
    evidence_count = len(symptom_set) + len(environment_set) + bool(distribution)
    available = tuple(item for item in (conditions or load_conditions()) if item["crop"] == crop)
    if not available:
        return ScreeningResult(
            status="unsupported",
            headline="Crop not supported",
            explanation="Vita AI has no curated screening rules for this crop yet.",
            photo=photo,
            evidence_count=evidence_count,
        )
    if not symptom_set:
        return ScreeningResult(
            status="insufficient_evidence",
            headline="Add visible symptoms",
            explanation="A photo alone is not enough for this prototype. Select what you can see.",
            photo=photo,
            evidence_count=evidence_count,
        )

    ranked = sorted(
        (
            _score_condition(item, symptom_set, environment_set, distribution, photo)
            for item in available
        ),
        key=lambda item: item.match_score,
        reverse=True,
    )
    top = ranked[0]
    margin = top.match_score - ranked[1].match_score if len(ranked) > 1 else top.match_score
    strong_symptom_count = len(top.matched_evidence)

    if photo and photo.status == "poor":
        status = "needs_better_photo"
        headline = "Retake the photo before relying on the shortlist"
        explanation = (
            "The symptom answers produced possible matches, but the photo failed the quality gate. "
            "Use the candidates only to guide a better capture or an expert inspection."
        )
    elif top.match_score >= 0.58 and margin >= 0.10 and strong_symptom_count >= 2:
        status = "screened"
        headline = f"Closest screening match: {top.name}"
        explanation = (
            "This is a transparent symptom match, not a diagnosis or a calibrated probability. "
            "Confirm it against the evidence and escalation guidance below."
        )
    else:
        status = "needs_review"
        headline = "No reliable single match"
        explanation = (
            "The evidence is limited or the leading candidates are too close. Retake the photo, "
            "observe the plant over time, or ask a local plant-health professional."
        )

    return ScreeningResult(
        status=status,
        headline=headline,
        explanation=explanation,
        candidates=tuple(ranked[:3]),
        photo=photo,
        evidence_count=evidence_count,
    )
