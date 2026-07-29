"""Secure image decoding and model input preparation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import IMAGE_SIZE, MAX_FILE_SIZE_MB, SUPPORTED_FORMATS

ImageSource = bytes | bytearray | str | Path | BinaryIO


class ImageValidationError(ValueError):
    """Raised when uploaded content is unsafe or unsuitable as an image."""


def _read_bytes(source: ImageSource) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise ImageValidationError("Unsupported image input.")
    if not data:
        raise ImageValidationError("The selected file is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ImageValidationError(f"The image exceeds the {MAX_FILE_SIZE_MB} MB upload limit.")
    return data


def load_image(source: ImageSource) -> Image.Image:
    """Decode a genuine JPEG/PNG, apply orientation, and convert to RGB."""

    data = _read_bytes(source)
    try:
        with Image.open(BytesIO(data)) as opened:
            detected_format = (opened.format or "").upper()
            if detected_format not in SUPPORTED_FORMATS:
                raise ImageValidationError("Only genuine JPG, JPEG, and PNG images are accepted.")
            opened.verify()
        with Image.open(BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("The file could not be decoded as a valid image.") from exc
    return image.copy()


def prepare_image(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
    preprocessing: str = "embedded",
) -> np.ndarray:
    """Return a float32 batch with shape ``(1, height, width, 3)``."""

    resized = ImageOps.fit(image.convert("RGB"), image_size, method=Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    if preprocessing == "mobilenet_v2":
        array = array / 127.5 - 1.0
    elif preprocessing == "rescale_0_1":
        array = array / 255.0
    elif preprocessing != "embedded":
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}")
    return np.expand_dims(array, axis=0)

