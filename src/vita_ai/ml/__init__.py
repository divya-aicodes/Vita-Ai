"""Optional machine-learning training and inference support for Vita AI."""

from .registry import MODEL_SPECS, ModelSpec, get_model_spec

__all__ = ["MODEL_SPECS", "ModelSpec", "get_model_spec"]
