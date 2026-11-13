"""Single source of truth for the five comparable transfer-learning models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    application_name: str
    image_size: int
    purpose: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="efficientnet_v2_b0",
        display_name="EfficientNetV2B0",
        application_name="EfficientNetV2B0",
        image_size=224,
        purpose="Best overall accuracy, size, and speed balance",
    ),
    ModelSpec(
        key="convnext_tiny",
        display_name="ConvNeXtTiny",
        application_name="ConvNeXtTiny",
        image_size=224,
        purpose="High-accuracy modern CNN candidate",
    ),
    ModelSpec(
        key="densenet121",
        display_name="DenseNet121",
        application_name="DenseNet121",
        image_size=224,
        purpose="Strong fine-texture feature reuse",
    ),
    ModelSpec(
        key="mobilenet_v3_large",
        display_name="MobileNetV3Large",
        application_name="MobileNetV3Large",
        image_size=224,
        purpose="Efficient mobile and CPU deployment",
    ),
    ModelSpec(
        key="resnet50_v2",
        display_name="ResNet50V2",
        application_name="ResNet50V2",
        image_size=224,
        purpose="Stable residual-network benchmark",
    ),
)


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    supported = ", ".join(item.key for item in MODEL_SPECS)
    raise ValueError(f"Unknown model '{key}'. Choose one of: {supported}")
