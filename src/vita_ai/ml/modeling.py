"""PyTorch/timm model construction for GPU training and ONNX export."""

from __future__ import annotations

from typing import Any

from .registry import ModelSpec


def _libraries() -> tuple[Any, Any]:
    try:
        import timm
        import torch
    except ImportError as exc:  # pragma: no cover - requires the optional ML environment
        raise RuntimeError(
            "PyTorch and timm are required for training. Install requirements-ml.txt first."
        ) from exc
    return torch, timm


def build_classifier(spec: ModelSpec, *, num_classes: int, pretrained: bool = True) -> Any:
    if num_classes < 2:
        raise ValueError("At least two classes are required for classification.")
    _, timm = _libraries()
    return timm.create_model(
        spec.timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def freeze_backbone(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True


def enable_fine_tuning(model: Any, *, trainable_fraction: float = 0.25) -> None:
    if not 0 < trainable_fraction <= 1:
        raise ValueError("trainable_fraction must be in the range (0, 1].")
    parameters = list(model.parameters())
    first_trainable = int(len(parameters) * (1 - trainable_fraction))
    for index, parameter in enumerate(parameters):
        parameter.requires_grad = index >= first_trainable
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True


def data_config(model: Any) -> dict[str, Any]:
    _, timm = _libraries()
    return timm.data.resolve_model_data_config(model)


def trainable_parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
