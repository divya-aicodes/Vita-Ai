"""Shared configuration and user-facing vocabulary."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DATABASE_PATH = RUNTIME_DIR / "vita_ai.db"

APP_NAME = "Vita AI"
APP_VERSION = "0.1.0"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

SYMPTOMS: dict[str, str] = {
    "no_visible_symptoms": "No visible symptoms",
    "dark_spots": "Dark brown or black spots",
    "concentric_rings": "Target-like concentric rings",
    "water_soaked": "Water-soaked lesions",
    "yellow_halo": "Yellow halos around spots",
    "white_powder": "White powdery coating",
    "yellowing": "General yellowing",
    "leaf_curl": "Curling or distorted leaves",
    "mosaic": "Mottled or mosaic colour pattern",
    "brown_edges": "Brown or scorched leaf edges",
    "orange_pustules": "Orange or rust-coloured pustules",
    "gray_growth": "Gray or fuzzy growth",
    "wilting": "Wilting despite available moisture",
    "holes": "Holes or chewed tissue",
}

ENVIRONMENTS: dict[str, str] = {
    "humid": "High humidity",
    "recent_rain": "Recent rain or prolonged leaf wetness",
    "warm": "Warm conditions",
    "cool": "Cool conditions",
    "dry": "Dry conditions",
    "crowded": "Crowded plants or poor airflow",
}

DISTRIBUTIONS: dict[str, str] = {
    "lower_leaves": "Mostly older/lower leaves",
    "upper_leaves": "Mostly young/upper leaves",
    "whole_plant": "Across the whole plant",
    "scattered": "Scattered individual leaves",
    "fruit_or_stem": "Also visible on fruit or stems",
}
