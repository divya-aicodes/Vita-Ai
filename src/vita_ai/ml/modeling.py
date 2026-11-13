"""Keras model construction with model-specific, reproducible preprocessing."""

from __future__ import annotations

from typing import Any

from .registry import ModelSpec


def _keras() -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - exercised only in an ML environment
        raise RuntimeError(
            "TensorFlow is required for ML training. Install requirements-ml.txt first."
        ) from exc
    return keras


def _preprocess(keras: Any, spec: ModelSpec, inputs: Any) -> Any:
    if spec.key == "densenet121":
        return keras.applications.densenet.preprocess_input(inputs)
    if spec.key == "resnet50_v2":
        return keras.applications.resnet_v2.preprocess_input(inputs)
    # EfficientNetV2, ConvNeXt, and MobileNetV3 include preprocessing by default.
    return inputs


def build_classifier(
    spec: ModelSpec,
    *,
    num_classes: int,
    dropout: float = 0.30,
    weights: str | None = "imagenet",
) -> tuple[Any, Any]:
    """Build a frozen ImageNet backbone and a small task-specific classification head."""

    if num_classes < 2:
        raise ValueError("At least two classes are required for classification.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1).")

    keras = _keras()
    shape = (spec.image_size, spec.image_size, 3)
    factory = getattr(keras.applications, spec.application_name)
    backbone = factory(
        include_top=False,
        weights=weights,
        input_shape=shape,
        pooling="avg",
    )
    backbone.trainable = False

    augmentation = keras.Sequential(
        (
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomContrast(0.12),
        ),
        name="training_augmentation",
    )
    inputs = keras.Input(shape=shape, name="image")
    x = augmentation(inputs)
    x = _preprocess(keras, spec, x)
    x = backbone(x, training=False)
    x = keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="condition")(x)
    model = keras.Model(inputs, outputs, name=f"vita_{spec.key}")
    return model, backbone


def compile_classifier(model: Any, *, learning_rate: float) -> None:
    keras = _keras()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=(
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ),
    )


def enable_fine_tuning(backbone: Any, *, trainable_fraction: float = 0.25) -> None:
    """Unfreeze only the final portion of a backbone while keeping BatchNorm stable."""

    if not 0 < trainable_fraction <= 1:
        raise ValueError("trainable_fraction must be in the range (0, 1].")
    keras = _keras()
    backbone.trainable = True
    first_trainable = int(len(backbone.layers) * (1 - trainable_fraction))
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= first_trainable and not isinstance(
            layer, keras.layers.BatchNormalization
        )
