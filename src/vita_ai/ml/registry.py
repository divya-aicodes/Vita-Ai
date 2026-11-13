"""Single source of truth for the five comparable transfer-learning models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    timm_name: str
    image_size: int
    purpose: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="efficientnet_v2_b0",
        display_name="EfficientNetV2B0",
        timm_name="tf_efficientnetv2_b0.in1k",
        image_size=192,
        purpose="Balanced accuracy, size, and speed candidate",
    ),
    ModelSpec(
        key="convnext_tiny",
        display_name="ConvNeXtTiny",
        timm_name="convnext_tiny.in12k_ft_in1k",
        image_size=224,
        purpose="Strongest controlled-test accuracy candidate",
    ),
    ModelSpec(
        key="densenet121",
        display_name="DenseNet121",
        timm_name="densenet121.ra_in1k",
        image_size=224,
        purpose="Strong fine-texture feature reuse",
    ),
    ModelSpec(
        key="mobilenet_v3_large",
        display_name="MobileNetV3Large",
        timm_name="mobilenetv3_large_100.ra_in1k",
        image_size=224,
        purpose="Efficient mobile and CPU deployment",
    ),
    ModelSpec(
        key="resnet50_v2",
        display_name="ResNet50V2",
        timm_name="resnetv2_50x1_bit.goog_in21k_ft_in1k",
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
