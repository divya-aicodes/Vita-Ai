"""Human-readable screening report generation."""

from __future__ import annotations

from datetime import UTC, datetime

from .config import DISTRIBUTIONS, ENVIRONMENTS, SYMPTOMS
from .domain import ScreeningResult


def build_report(
    *,
    case_id: str,
    crop: str,
    symptoms: list[str],
    environments: list[str],
    distribution: str,
    notes: str,
    result: ScreeningResult,
) -> str:
    lines = [
        "VITA AI — PLANT HEALTH SCREENING REPORT",
        "=" * 45,
        f"Case: {case_id}",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Crop: {crop}",
        f"Outcome: {result.headline}",
        "",
        "Important: This prototype produces transparent symptom matches, not a diagnosis or a "
        "calibrated disease probability.",
        "",
        "Evidence",
        "--------",
        "Symptoms: " + ", ".join(SYMPTOMS.get(item, item) for item in symptoms),
        "Environment: "
        + (", ".join(ENVIRONMENTS.get(item, item) for item in environments) or "Not provided"),
        f"Distribution: {DISTRIBUTIONS.get(distribution, distribution)}",
        f"Photo quality: {result.photo.status if result.photo else 'Not assessed'}",
    ]
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")

    lines.extend(["", "Shortlist", "---------"])
    for index, candidate in enumerate(result.candidates, start=1):
        lines.extend(
            [
                f"{index}. {candidate.name} — {candidate.match_score:.0%} rule match",
                f"   Matched: {', '.join(candidate.matched_evidence) or 'No strong symptom match'}",
                f"   About: {candidate.overview}",
            ]
        )

    if result.primary:
        lines.extend(["", "Safe next steps", "---------------"])
        lines.extend(f"- {step}" for step in result.primary.next_steps)
        lines.extend(["", "Prevention", "----------"])
        lines.extend(f"- {step}" for step in result.primary.prevention)

    lines.extend(
        [
            "",
            "Safety note",
            "-----------",
            "Do not apply a pesticide, discard a crop, or make a high-impact decision from this "
            "report alone. Seek local expert confirmation when symptoms spread, plants wilt "
            "rapidly, "
            "or the outcome remains uncertain.",
        ]
    )
    return "\n".join(lines) + "\n"
