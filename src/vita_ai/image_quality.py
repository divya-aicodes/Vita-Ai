"""Private, local photo validation and descriptive visual measurements."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_IMAGE_BYTES, SUPPORTED_IMAGE_FORMATS
from .domain import PhotoAssessment


class ImageValidationError(ValueError):
    """Raised when an upload is not a safe supported image."""


def _read_bytes(source: bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise ImageValidationError("Unsupported image source.")
    if not data:
        raise ImageValidationError("The selected file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError("The image is larger than the 8 MB limit.")
    return data


def decode_image(source: bytes | bytearray | BinaryIO) -> Image.Image:
    """Verify the actual file content, apply EXIF orientation, and return RGB pixels."""

    data = _read_bytes(source)
    try:
        with Image.open(BytesIO(data)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ImageValidationError("Use a genuine JPEG, PNG, or WebP image.")
            opened.verify()
        with Image.open(BytesIO(data)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("The upload could not be decoded as a valid image.") from exc


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def assess_photo(image: Image.Image) -> PhotoAssessment:
    """Measure capture quality and broad colour cues without diagnosing a disease."""

    preview = ImageOps.contain(image.convert("RGB"), (640, 640), Image.Resampling.LANCZOS)
    pixels = np.asarray(preview, dtype=np.float32)
    red, green, blue = (pixels[..., index] for index in range(3))
    gray = red * 0.299 + green * 0.587 + blue * 0.114

    brightness = float(gray.mean())
    contrast = float(gray.std())
    vertical = np.diff(gray, n=2, axis=0)
    horizontal = np.diff(gray, n=2, axis=1)
    sharpness = float((np.abs(vertical).mean() + np.abs(horizontal).mean()) / 2)

    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1)
    green_mask = (green > red * 1.08) & (green > blue * 1.08) & (green > 45)
    yellow_mask = (red > 105) & (green > 95) & (blue < green * 0.72) & (red > blue * 1.35)
    brown_mask = (red > 55) & (red < 190) & (green < red * 0.82) & (blue < green * 0.82)
    pale_mask = (gray > 175) & (saturation < 0.18)
    leaf_mask = green_mask | yellow_mask | brown_mask

    leaf_coverage = float(leaf_mask.mean())
    yellow_ratio = float(yellow_mask.mean())
    brown_ratio = float(brown_mask.mean())
    pale_ratio = float(pale_mask.mean())

    dimension_score = _clamp(min(image.size) / 640)
    exposure_score = _clamp(1 - abs(brightness - 132) / 120)
    contrast_score = _clamp((contrast - 10) / 42)
    sharpness_score = _clamp((sharpness - 1.5) / 16)
    coverage_score = _clamp(leaf_coverage / 0.32)
    quality_score = (
        dimension_score * 0.20
        + exposure_score * 0.20
        + contrast_score * 0.20
        + sharpness_score * 0.25
        + coverage_score * 0.15
    )

    issues: list[str] = []
    if min(image.size) < 256:
        issues.append("Resolution is too low; use at least 256 px on the shortest side.")
    if brightness < 42:
        issues.append("The photo is too dark.")
    elif brightness > 220:
        issues.append("The photo is overexposed.")
    if contrast < 14:
        issues.append("The leaf has very little visible contrast.")
    if sharpness < 3.2:
        issues.append("The photo appears blurred or lacks edge detail.")
    if leaf_coverage < 0.07:
        issues.append("A leaf is not clearly dominant in the frame.")

    if sharpness < 1.0 or contrast < 5.0:
        status = "poor"
    elif quality_score >= 0.68 and len(issues) <= 1:
        status = "good"
    elif quality_score >= 0.44 and len(issues) <= 3:
        status = "usable"
    else:
        status = "poor"

    observations: list[str] = []
    if green_mask.mean() >= 0.12:
        observations.append("green_leaf_present")
    if yellow_ratio >= 0.055:
        observations.append("yellowing_visible")
    if brown_ratio >= 0.035:
        observations.append("brown_tissue_visible")
    if pale_ratio >= 0.10:
        observations.append("pale_surface_visible")

    return PhotoAssessment(
        width=image.width,
        height=image.height,
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        sharpness=round(sharpness, 1),
        leaf_coverage=round(leaf_coverage, 3),
        yellow_ratio=round(yellow_ratio, 3),
        brown_ratio=round(brown_ratio, 3),
        pale_ratio=round(pale_ratio, 3),
        quality_score=round(quality_score, 3),
        status=status,
        issues=tuple(issues),
        observations=tuple(observations),
    )
