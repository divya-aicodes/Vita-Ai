"""Confidence-aware prediction service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    CROP_SUPPORT_THRESHOLD,
    DISEASE_ACCEPT_THRESHOLD,
    DISEASE_MARGIN_THRESHOLD,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    ModelMetadata,
)
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
    reliable: bool = True
    selected_crop: str | None = None
    crop_support: float = 1.0
    disease_margin: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.as_dict(),
            "alternatives": [item.as_dict() for item in self.alternatives],
            "confidence_level": self.confidence_level,
            "guidance": self.guidance,
            "inference_time_ms": self.inference_time_ms,
            "disease_info": self.disease_info.as_dict(),
            "reliable": self.reliable,
            "selected_crop": self.selected_crop,
            "crop_support": self.crop_support,
            "disease_margin": self.disease_margin,
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
    expected_crop: str | None = None,
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
    selected_crop: str | None = None
    crop_support = 1.0
    disease_margin = 1.0
    reliable = True

    candidate_indexes = np.arange(len(labels))
    candidate_probabilities = probabilities
    if expected_crop is not None:
        crop_lookup: dict[str, str] = {}
        crop_indexes: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            crop = get_disease_info(label).crop
            crop_lookup.setdefault(crop.casefold(), crop)
            crop_indexes.setdefault(crop.casefold(), []).append(index)

        crop_key = expected_crop.strip().casefold()
        if crop_key not in crop_indexes:
            supported = ", ".join(sorted(crop_lookup.values()))
            raise ValueError(f"Unsupported crop {expected_crop!r}. Choose one of: {supported}.")

        selected_crop = crop_lookup[crop_key]
        candidate_indexes = np.asarray(crop_indexes[crop_key], dtype=np.int64)
        raw_crop_probabilities = probabilities[candidate_indexes]
        crop_support = float(raw_crop_probabilities.sum())
        if crop_support > np.finfo(np.float64).eps:
            candidate_probabilities = raw_crop_probabilities / crop_support
        else:
            candidate_probabilities = np.zeros_like(raw_crop_probabilities)

    local_order = np.argsort(candidate_probabilities)[::-1]
    limit = max(1, min(top_k, len(candidate_indexes)))
    order = candidate_indexes[local_order[:limit]]
    ranked = tuple(
        _rank(labels[index], candidate_probabilities[local_index])
        for local_index, index in zip(local_order[:limit], order, strict=True)
    )

    if len(local_order) > 1:
        disease_margin = float(
            candidate_probabilities[local_order[0]] - candidate_probabilities[local_order[1]]
        )

    if selected_crop is not None:
        reliable = (
            crop_support >= CROP_SUPPORT_THRESHOLD
            and ranked[0].confidence >= DISEASE_ACCEPT_THRESHOLD
            and disease_margin >= DISEASE_MARGIN_THRESHOLD
        )

    if not reliable and crop_support < CROP_SUPPORT_THRESHOLD:
        level = "Unreliable"
        guidance = (
            f"No reliable match: the image does not sufficiently resemble the selected plant "
            f"({selected_crop}). Check the plant selection or try a closer leaf photo."
        )
    elif not reliable and ranked[0].confidence < DISEASE_ACCEPT_THRESHOLD:
        level = "Unreliable"
        guidance = (
            "No reliable match: the visible pattern is ambiguous among the supported conditions "
            "for this plant. Do not use the candidate as a diagnosis."
        )
    elif not reliable:
        level = "Unreliable"
        guidance = (
            "No reliable match: the leading disease candidates are too close to distinguish safely. "
            "Try another image or obtain expert confirmation."
        )
    else:
        level, guidance = confidence_message(ranked[0].confidence)
    return PredictionResult(
        primary=ranked[0],
        alternatives=ranked[1:],
        confidence_level=level,
        guidance=guidance,
        inference_time_ms=round(elapsed_ms, 1),
        disease_info=get_disease_info(ranked[0].label),
        reliable=reliable,
        selected_crop=selected_crop,
        crop_support=crop_support,
        disease_margin=disease_margin,
    )
