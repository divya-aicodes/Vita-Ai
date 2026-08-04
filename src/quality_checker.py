"""Image-quality checks implemented with NumPy and Pillow only."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    score: float
    acceptable: bool
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    entropy: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


def _normalized_entropy(gray: np.ndarray) -> float:
    histogram, _ = np.histogram(gray, bins=64, range=(0, 256))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return entropy / 6.0


def assess_image_quality(image: Image.Image) -> QualityResult:
    """Assess dimensions, exposure, contrast, detail, and visible content."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    sample = rgb.copy()
    sample.thumbnail((768, 768), Image.Resampling.BILINEAR)
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    entropy = _normalized_entropy(gray)

    # Laplacian-like second derivative; normalized for resolution-independent reporting.
    center = gray[1:-1, 1:-1]
    laplacian = -4.0 * center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    sharpness = float(np.var(laplacian))
    warnings: list[str] = []
    penalties = 0.0

    if min(width, height) < 224:
        warnings.append("The image is small. Move closer and use at least 224 × 224 pixels.")
        penalties += 25
    if brightness < 40:
        warnings.append("The image appears too dark. Photograph the leaf in brighter, even light.")
        penalties += min(30, (40 - brightness) * 0.75)
    elif brightness > 220:
        warnings.append("The image appears overexposed. Avoid harsh light and glare.")
        penalties += min(30, (brightness - 220) * 0.85)
    if contrast < 18:
        warnings.append(
            "The image has very low contrast. Use a plain background and even lighting."
        )
        penalties += min(25, (18 - contrast) * 1.2)
    if sharpness < 30:
        warnings.append("The image appears blurry. Hold the camera steady and focus on the leaf.")
        penalties += min(35, (30 - sharpness) * 0.9)
    if entropy < 0.35:
        warnings.append("The image contains too little visible detail for a reliable screening.")
        penalties += min(30, (0.35 - entropy) * 80)

    score = round(max(0.0, min(100.0, 100.0 - penalties)), 1)
    critical_exposure = brightness < 18 or brightness > 242
    acceptable = min(width, height) >= 128 and entropy >= 0.22 and not critical_exposure
    return QualityResult(
        score=score,
        acceptable=acceptable,
        width=width,
        height=height,
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        sharpness=round(sharpness, 1),
        entropy=round(entropy, 3),
        warnings=tuple(warnings),
    )
