"""Lazy ONNX inference for trained PlantVillage classifiers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .registry import MODEL_SPECS, get_model_spec


class ModelUnavailableError(RuntimeError):
    """Raised when a requested trained model artifact is missing or invalid."""


@dataclass(frozen=True)
class DiseasePrediction:
    class_name: str
    crop: str
    condition: str
    confidence: float


REQUIRED_ARTIFACTS = ("model.onnx", "labels.json", "metrics.json")
MAX_MODEL_ARCHIVE_BYTES = 250 * 1024 * 1024
_DOWNLOAD_LOCK = threading.Lock()


def _is_complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in REQUIRED_ARTIFACTS)


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"models": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ModelUnavailableError("The model release manifest is invalid.")
    return payload


def available_models(artifacts_dir: Path, manifest_path: Path | None = None) -> tuple[str, ...]:
    available: set[str] = set()
    for directory in sorted(artifacts_dir.iterdir()) if artifacts_dir.exists() else ():
        if _is_complete(directory):
            try:
                get_model_spec(directory.name)
            except ValueError:
                continue
            available.add(directory.name)
    for key in _load_manifest(manifest_path)["models"]:
        try:
            get_model_spec(key)
        except ValueError:
            continue
        available.add(key)
    return tuple(spec.key for spec in MODEL_SPECS if spec.key in available)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if not set(REQUIRED_ARTIFACTS).issubset(names):
            raise ModelUnavailableError("The downloaded model archive is incomplete.")
        for item in archive.infolist():
            item_path = Path(item.filename)
            if item_path.is_absolute() or ".." in item_path.parts:
                raise ModelUnavailableError("The downloaded model archive contains unsafe paths.")
        archive.extractall(destination)


def _download_model(model_key: str, cache_dir: Path, manifest_path: Path) -> Path:
    with _DOWNLOAD_LOCK:
        return _download_model_locked(model_key, cache_dir, manifest_path)


def _download_model_locked(model_key: str, cache_dir: Path, manifest_path: Path) -> Path:
    entry = _load_manifest(manifest_path)["models"].get(model_key)
    if not isinstance(entry, dict):
        raise ModelUnavailableError(f"No downloadable artifact is configured for {model_key}.")
    url = entry.get("url")
    expected_hash = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if not isinstance(url, str) or not url.startswith(
        "https://github.com/divya-aicodes/Vita-Ai/releases/download/"
    ):
        raise ModelUnavailableError("The configured model URL is not trusted.")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ModelUnavailableError("The configured model checksum is invalid.")
    if not isinstance(expected_size, int) or not 0 < expected_size <= MAX_MODEL_ARCHIVE_BYTES:
        raise ModelUnavailableError("The configured model archive size is invalid.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cache_dir / model_key
    if _is_complete(model_dir):
        return model_dir
    with tempfile.TemporaryDirectory(dir=cache_dir) as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / "model.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "Vita-AI/0.1"})
        digest = hashlib.sha256()
        received = 0
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                archive_path.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > expected_size or received > MAX_MODEL_ARCHIVE_BYTES:
                        raise ModelUnavailableError(
                            "The model download exceeded its expected size."
                        )
                    digest.update(chunk)
                    output.write(chunk)
        except OSError as exc:
            raise ModelUnavailableError(f"The model download failed: {exc}") from exc
        if received != expected_size or digest.hexdigest() != expected_hash:
            raise ModelUnavailableError("The downloaded model failed integrity verification.")
        extracted = temp_dir / "extracted"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(extracted), str(model_dir))
    return model_dir


@lru_cache(maxsize=2)
def _session(model_path: str) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise ModelUnavailableError("ONNX Runtime is not installed.") from exc
    return ort.InferenceSession(model_path, providers=("CPUExecutionProvider",))


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    left = max(0, (image.width - size) // 2)
    top = max(0, (image.height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image: Image.Image, metadata: dict[str, Any]) -> np.ndarray:
    _, height, width = metadata["input_size"]
    if height != width:
        raise ModelUnavailableError("Only square model inputs are supported.")
    crop_pct = float(metadata.get("crop_pct", 1.0))
    resize_size = max(height, round(height / crop_pct))
    rgb = image.convert("RGB")
    scale = resize_size / min(rgb.size)
    resized = rgb.resize(
        (round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.BICUBIC
    )
    cropped = _center_crop(resized, height)
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.asarray(metadata["mean"], dtype=np.float32)
    std = np.asarray(metadata["std"], dtype=np.float32)
    normalized = (pixels - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)


def _friendly_label(class_name: str) -> tuple[str, str]:
    crop, _, condition = class_name.partition("___")
    return crop.replace("_", " ").replace(",", "").strip(), condition.replace("_", " ").strip()


def predict(
    image: Image.Image,
    *,
    artifacts_dir: Path,
    model_key: str,
    top_k: int = 3,
    manifest_path: Path | None = None,
) -> tuple[DiseasePrediction, ...]:
    get_model_spec(model_key)
    model_dir = artifacts_dir / model_key
    if not _is_complete(model_dir) and manifest_path is not None:
        model_dir = _download_model(model_key, artifacts_dir, manifest_path)
    model_path = model_dir / "model.onnx"
    labels_path = model_dir / "labels.json"
    metrics_path = model_dir / "metrics.json"
    if not all(path.is_file() for path in (model_path, labels_path, metrics_path)):
        raise ModelUnavailableError(f"Trained artifacts are incomplete for {model_key}.")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    inputs = preprocess_image(image, metadata)
    session = _session(str(model_path.resolve()))
    input_name = session.get_inputs()[0].name
    logits = np.asarray(session.run(None, {input_name: inputs})[0])[0]
    logits = logits - logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    indices = np.argsort(probabilities)[::-1][: max(1, min(top_k, len(labels)))]
    predictions = []
    for index in indices:
        crop, condition = _friendly_label(labels[int(index)])
        predictions.append(
            DiseasePrediction(
                class_name=labels[int(index)],
                crop=crop,
                condition=condition,
                confidence=float(probabilities[index]),
            )
        )
    return tuple(predictions)
