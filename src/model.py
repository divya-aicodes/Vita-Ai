"""TensorFlow model construction and loading.

TensorFlow is imported lazily so quality checks, database tools, and unit tests
remain usable in lightweight environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import IMAGE_SIZE, MODEL_PATH, ModelMetadata


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested without a usable trained model."""


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelUnavailableError(
            "TensorFlow is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc
    return tf


def build_mobilenetv2(
    num_classes: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    trainable_backbone: bool = False,
    imagenet_weights: bool = True,
):
    """Build the production MobileNetV2 transfer-learning classifier."""

    tf = _tensorflow()
    keras = tf.keras
    inputs = keras.Input((*image_size, 3), name="image", dtype="float32")
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.06),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )
    x = augmentation(inputs)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_preprocess")(x)
    backbone_core = keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
    )
    backbone = keras.Model(
        backbone_core.input,
        backbone_core.output,
        name="mobilenetv2_backbone",
    )
    backbone.trainable = trainable_backbone
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(0.30, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="vita_ai_mobilenetv2")


def build_custom_cnn(num_classes: int, image_size: tuple[int, int] = IMAGE_SIZE):
    """Small baseline CNN for academic comparison."""

    tf = _tensorflow()
    keras = tf.keras
    inputs = keras.Input((*image_size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for filters in (32, 64, 128):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(192, 3, padding="same", activation="relu", name="last_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="vita_ai_custom_cnn")


def load_trained_model(path: Path = MODEL_PATH):
    """Load a model only from the configured local path."""

    if not path.exists():
        raise ModelUnavailableError(
            f"No trained model was found at {path}. Run download_model.py or train.py first."
        )
    tf = _tensorflow()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ModelUnavailableError(f"The model file could not be loaded: {exc}") from exc


def compile_model(model, learning_rate: float = 1e-3):
    tf = _tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model


def runtime_metadata_for(model) -> ModelMetadata:
    backbone = next(
        (layer for layer in model.layers if getattr(layer, "name", "") == "mobilenetv2_backbone"),
        None,
    )
    return ModelMetadata(
        model_name=model.name,
        preprocessing="embedded",
        last_conv_layer="out_relu" if backbone is not None else "last_conv",
    )
