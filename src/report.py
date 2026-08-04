"""Portable text report generation with no extra dependencies."""

from __future__ import annotations

from datetime import UTC, datetime

from .prediction import PredictionResult
from .quality_checker import QualityResult


def build_text_report(result: PredictionResult, quality: QualityResult, model_version: str) -> str:
    alternatives = (
        "\n".join(
            f"  - {item.crop} — {item.condition}: {item.confidence:.1%}"
            for item in result.alternatives
        )
        or "  - None"
    )
    info = result.disease_info
    return f"""ᕓ𐌉𐌕𐌀 𐌀𐌉 — Plant Screening Report
Generated: {datetime.now(UTC).isoformat(timespec="seconds")}
Model: {model_version}

SCREENING RESULT
Crop: {result.primary.crop}
Condition: {result.primary.condition}
Status: {result.primary.status}
Confidence: {result.primary.confidence:.1%} ({result.confidence_level})
Image quality score: {quality.score:.1f}/100
Inference time: {result.inference_time_ms:.1f} ms

ALTERNATIVE PREDICTIONS
{alternatives}

EDUCATIONAL INFORMATION
Description: {info.description}
Common visible symptoms: {info.symptoms}
General prevention: {info.prevention}

IMPORTANT
{result.guidance}
{info.expert_warning}

No uploaded image or personal information is included in this report.
"""
