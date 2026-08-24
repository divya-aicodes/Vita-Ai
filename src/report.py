"""Portable text report generation with no extra dependencies."""

from __future__ import annotations

from datetime import UTC, datetime

from .prediction import PredictionResult
from .quality_checker import QualityResult


def build_text_report(result: PredictionResult, quality: QualityResult, model_version: str) -> str:
    alternatives = (
        (
            "\n".join(
                f"  - {item.crop} — {item.condition}: {item.confidence:.1%}"
                for item in result.alternatives
            )
            or "  - None"
        )
        if result.reliable
        else "  - Withheld because the reliability checks failed"
    )
    info = result.disease_info
    selected_crop = result.selected_crop or result.primary.crop
    if result.reliable:
        result_summary = f"""Selected plant: {selected_crop}
Condition: {result.primary.condition}
Status: {result.primary.status}
Within-plant match: {result.primary.confidence:.1%} ({result.confidence_level})"""
        educational_information = f"""Description: {info.description}
Common visible symptoms: {info.symptoms}
General prevention: {info.prevention}"""
    else:
        result_summary = f"""Selected plant: {selected_crop}
Condition: No reliable match
Status: Unknown
Candidate result: withheld"""
        educational_information = (
            "Disease-specific information was withheld because the screening did not pass "
            "the reliability checks."
        )
    return f"""ᕓ𐌉𐌕𐌀 𐌀𐌉 — Plant Screening Report
Generated: {datetime.now(UTC).isoformat(timespec="seconds")}
Model: {model_version}

SCREENING RESULT
{result_summary}
Selected-plant support: {result.crop_support:.1%}
Photo quality score: {quality.score:.1f}/100 (technical photo properties only)
Inference time: {result.inference_time_ms:.1f} ms

ALTERNATIVE PREDICTIONS
{alternatives}

EDUCATIONAL INFORMATION
{educational_information}

IMPORTANT
{result.guidance}
This screening is not a confirmed diagnosis. Model scores and photo quality are not diagnostic accuracy.

No uploaded image or personal information is included in this report.
"""
