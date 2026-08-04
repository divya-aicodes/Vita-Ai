"""Confidence-aware prediction service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import HIGH_CONFIDENCE, LOW_CONFIDENCE, ModelMetadata
from .disease_info import DiseaseInfo, get_disease_info
from .preprocessing import prepare_image


@dataclass(frozen=True)
class RankedPrediction:
    label: str
    crop: str
    condition: str
    status: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    primary: RankedPrediction
    alternatives: tuple[RankedPrediction, ...]
    confidence_level: str
    guidance: str
    inference_time_ms: float
    disease_info: DiseaseInfo

    def as_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.as_dict(),
            "alternatives": [item.as_dict() for item in self.alternatives],
            "confidence_level": self.confidence_level,
            "guidance": self.guidance,
            "inference_time_ms": self.inference_time_ms,
            "disease_info": self.disease_info.as_dict(),
        }


def confidence_message(confidence: float) -> tuple[str, str]:
    if confidence >= HIGH_CONFIDENCE:
        return "High", "The model found a strong match among its supported classes."
    if confidence >= LOW_CONFIDENCE:
        return (
            "Moderate",
            "Treat this as a tentative screening result. Compare the alternatives and consider another clear photo.",
        )
    return (
        "Low",
        "The result is uncertain. Do not rely on the disease advice yet; capture another close, well-lit leaf image.",
    )


def _rank(label: str, confidence: float) -> RankedPrediction:
    info = get_disease_info(label)
    return RankedPrediction(label, info.crop, info.condition, info.status, float(confidence))


def predict_image(
    model: Any,
    image: Image.Image,
    labels: list[str],
    metadata: ModelMetadata,
    top_k: int = 3,
) -> PredictionResult:
    batch = prepare_image(
        image,
        image_size=(metadata.input_width, metadata.input_height),
        preprocessing=metadata.preprocessing,
    )
    started = time.perf_counter()
    raw = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = np.asarray(raw, dtype=np.float64).squeeze()
    if probabilities.ndim != 1 or probabilities.size != len(labels):
        raise ValueError(
            f"Model returned {probabilities.size} classes, but {len(labels)} labels are configured."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model output contains invalid numeric values.")
    total = probabilities.sum()
    if not np.isclose(total, 1.0, atol=1e-3):
        shifted = probabilities - probabilities.max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
    order = np.argsort(probabilities)[::-1][: max(1, min(top_k, len(labels)))]
    ranked = tuple(_rank(labels[index], probabilities[index]) for index in order)
    level, guidance = confidence_message(ranked[0].confidence)
    return PredictionResult(
        primary=ranked[0],
        alternatives=ranked[1:],
        confidence_level=level,
        guidance=guidance,
        inference_time_ms=round(elapsed_ms, 1),
        disease_info=get_disease_info(ranked[0].label),
    )
