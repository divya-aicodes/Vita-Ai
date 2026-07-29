"""Grad-CAM generation for models created by this project."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .config import ModelMetadata
from .preprocessing import prepare_image


def _tf() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to generate Grad-CAM.") from exc
    return tf


def make_gradcam_heatmap(model, batch: np.ndarray, class_index: int, metadata: ModelMetadata) -> np.ndarray:
    """Generate a heatmap for the stable model architectures in ``src.model``."""

    tf = _tf()
    backbone = None
    for layer in model.layers:
        if not hasattr(layer, "layers"):
            continue
        try:
            layer.get_layer(metadata.last_conv_layer)
            backbone = layer
            break
        except (ValueError, AttributeError):
            continue
    if backbone is not None:
        feature_model = tf.keras.Model(
            backbone.input,
            [backbone.get_layer(metadata.last_conv_layer).output, backbone.output],
        )
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            features = tensor
            backbone_index = model.layers.index(backbone)
            for layer in model.layers[1:backbone_index]:
                features = layer(features, training=False)
            conv_output, features = feature_model(features, training=False)
            tape.watch(conv_output)
            for layer in model.layers[backbone_index + 1 :]:
                features = layer(features, training=False)
            logits = features
            score = tf.gather(logits, int(class_index), axis=-1)
    else:
        target = model.get_layer(metadata.last_conv_layer)
        grad_model = tf.keras.Model(model.inputs, [target.output, model.output])
        with tf.GradientTape() as tape:
            conv_output, logits = grad_model(batch, training=False)
            score = tf.gather(logits, int(class_index), axis=-1)
    gradients = tape.gradient(score, conv_output)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were unavailable for this model.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """Apply a green-yellow-red heatmap without requiring Matplotlib."""

    heat = np.clip(heatmap, 0, 1)
    red = np.clip(2.2 * heat, 0, 1)
    green = np.clip(2.0 * (1 - np.abs(heat - 0.5) * 2), 0, 1)
    blue = np.clip(1.3 * (1 - heat) - 0.45, 0, 1)
    color = np.stack((red, green, blue), axis=-1)
    color_img = Image.fromarray(np.uint8(color * 255)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), color_img, alpha)


def explain_prediction(
    model,
    image: Image.Image,
    class_index: int,
    metadata: ModelMetadata,
) -> Image.Image:
    batch = prepare_image(
        image,
        image_size=(metadata.input_width, metadata.input_height),
        preprocessing="embedded",
    )
    heatmap = make_gradcam_heatmap(model, batch, class_index, metadata)
    return overlay_heatmap(image, heatmap)
