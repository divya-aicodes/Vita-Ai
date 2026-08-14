"""Central configuration shared by training, inference, and the UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATABASE_DIR = PROJECT_ROOT / "database"
ASSETS_DIR = PROJECT_ROOT / "assets"

APP_TITLE = "ᕓ𐌉𐌕𐌀 𐌀𐌉"
APP_SUBTITLE = "An Explainable CNN-Based Plant Disease Screening and Advisory System"
APP_VERSION = "1.3.0"
MODEL_VERSION = "mobilenetv2-plantvillage-1.0"

MODEL_PATH = MODELS_DIR / "plant_disease_mobilenetv2.keras"
LABELS_PATH = MODELS_DIR / "class_names.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
DATABASE_PATH = DATABASE_DIR / "vita_ai.db"

IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = {"JPEG", "PNG"}
HIGH_CONFIDENCE = 0.80
LOW_CONFIDENCE = 0.55

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Blueberry___Healthy",
    "Cherry___Powdery_mildew",
    "Cherry___Healthy",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___Healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Raspberry___Healthy",
    "Soybean___Healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___Healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy",
]


@dataclass(frozen=True)
class ModelMetadata:
    """Runtime facts needed to preprocess images safely."""

    model_version: str = MODEL_VERSION
    model_name: str = "MobileNetV2"
    input_width: int = IMAGE_SIZE[0]
    input_height: int = IMAGE_SIZE[1]
    preprocessing: str = "embedded"
    last_conv_layer: str = "out_relu"
    source: str = "local-training-pipeline"


def ensure_runtime_directories() -> None:
    """Create only app-owned runtime directories."""

    for directory in (MODELS_DIR, RESULTS_DIR, DATABASE_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_class_names(path: Path = LABELS_PATH) -> list[str]:
    """Load class order, falling back to the documented PlantVillage order."""

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("class_names", [])
        if isinstance(data, list) and data and all(isinstance(item, str) for item in data):
            return data
    return CLASS_NAMES.copy()


def load_model_metadata(path: Path = METADATA_PATH) -> ModelMetadata:
    if not path.exists():
        return ModelMetadata()
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    allowed = ModelMetadata.__dataclass_fields__.keys()
    return ModelMetadata(**{key: value for key, value in raw.items() if key in allowed})
