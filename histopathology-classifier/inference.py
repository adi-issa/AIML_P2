"""Inference and Grad-CAM utilities for the pathology portal.

TensorFlow is imported lazily so the UI can show a useful setup error instead
of failing immediately when the environment has not been installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)
CLASS_NAMES = (
    "Breast InSitu",
    "Breast Benign",
    "Breast Invasive",
    "SCC",
    "LUGAD",
)


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    architecture: str
    filename: str
    test_accuracy: float


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    backbone: Any
    preprocess: Callable[[Any], Any]


@dataclass
class PredictionResult:
    model_name: str
    predicted_class: str
    confidence: float
    confidence_level: str
    probabilities: dict[str, float]
    heatmap: np.ndarray
    overlay: Image.Image
    roi: Image.Image
    inference_seconds: float


MODEL_SPECS = (
    ModelSpec(
        display_name="ResNet50V2",
        architecture="resnet50v2",
        filename="resnet50v2_Tuning.keras",
        test_accuracy=0.9094,
    ),
    ModelSpec(
        display_name="DenseNet169",
        architecture="densenet169",
        filename="densenet169_Tuning.keras",
        test_accuracy=0.9289,
    ),
)


def confidence_level(confidence: float) -> str:
    """Map a probability to the bands used in the original notebook."""
    if confidence >= 0.90:
        return "Very high"
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.70:
        return "Moderate"
    if confidence >= 0.50:
        return "Low"
    return "Very low"


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run setup_and_run.ps1 with Python 3.11."
        ) from exc
    return tf


def load_model(spec: ModelSpec, model_path: str | Path) -> LoadedModel:
    """Load a saved Keras model and rebuild the notebook's Grad-CAM graph."""
    tf = _tensorflow()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = tf.keras.models.load_model(path, compile=False)
    if len(model.layers) < 3:
        raise ValueError(
            f"{path.name} does not match the expected backbone + classifier layout."
        )

    # The training notebook saved the ImageNet backbone as layer 1. Replaying
    # the classifier head from that feature tensor preserves the trained
    # weights and gives GradientTape a direct path to the convolutional maps.
    backbone = model.layers[1]
    # The bundled backbones were trained in mixed float16. On CPU, half-precision
    # convolutions can fall back to slow software kernels, including in cloud hosting.
    if not tf.config.list_physical_devices("GPU") and backbone.compute_dtype == "float16":
        builder = (tf.keras.applications.ResNet50V2 if spec.architecture == "resnet50v2"
                   else tf.keras.applications.DenseNet169)
        cpu_backbone = builder(weights=None, include_top=False, input_shape=(224, 224, 3))
        cpu_backbone.set_weights(backbone.get_weights())
        backbone = cpu_backbone

    if spec.architecture == "resnet50v2":
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
    elif spec.architecture == "densenet169":
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")

    return LoadedModel(spec, model.layers[2:], backbone, preprocess)


def prepare_image(image: Image.Image, loaded_model: LoadedModel) -> Any:
    tf = _tensorflow()
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(rgb, dtype=np.float32)
    image_batch = tf.expand_dims(image_array, axis=0)
    return loaded_model.preprocess(image_batch)


def predict_with_gradcam(
    loaded_model: LoadedModel,
    image: Image.Image,
    roi_threshold: float = 0.50,
) -> PredictionResult:
    """Predict one image and compute a Grad-CAM explanation for that class."""
    tf = _tensorflow()
    started = perf_counter()
    image_batch = prepare_image(image, loaded_model)

    feature_maps = loaded_model.backbone(image_batch, training=False)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(feature_maps)
        predictions = feature_maps
        for layer in loaded_model.model:
            predictions = layer(predictions, training=False)
        class_index_tensor = tf.argmax(predictions[0])
        class_score = predictions[:, class_index_tensor]

    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated for this model.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMAGE_SIZE, method="bilinear"
    )
    heatmap_array = np.asarray(tf.squeeze(heatmap).numpy(), dtype=np.float32)
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)

    probabilities_array = np.asarray(predictions[0].numpy(), dtype=np.float32)
    if probabilities_array.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, but the model returned "
            f"{probabilities_array.shape[0]}."
        )

    class_index = int(class_index_tensor.numpy())
    confidence = float(probabilities_array[class_index])
    source = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    return PredictionResult(
        model_name=loaded_model.spec.display_name,
        predicted_class=CLASS_NAMES[class_index],
        confidence=confidence,
        confidence_level=confidence_level(confidence),
        probabilities={
            class_name: float(probabilities_array[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
        heatmap=heatmap_array,
        overlay=make_overlay(source, heatmap_array),
        roi=make_roi_image(source, heatmap_array, roi_threshold),
        inference_seconds=perf_counter() - started,
    )


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation of matplotlib's jet colormap."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def heatmap_image(heatmap: np.ndarray) -> Image.Image:
    colored = (_jet_colormap(heatmap) * 255.0).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def make_overlay(source: Image.Image, heatmap: np.ndarray) -> Image.Image:
    original = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    colored = _jet_colormap(heatmap)
    blended = np.clip(0.5 * (original + colored), 0.0, 1.0)
    return Image.fromarray((blended * 255.0).astype(np.uint8), mode="RGB")


def make_roi_image(
    source: Image.Image, heatmap: np.ndarray, threshold: float = 0.50
) -> Image.Image:
    """Tint salient regions red and outline their boundary in yellow."""
    original = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    mask = heatmap >= threshold
    tinted = original.copy()
    tint = np.zeros_like(tinted)
    tint[..., 0] = 1.0
    tinted[mask] = 0.62 * original[mask] + 0.38 * tint[mask]

    up = np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down = np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    boundary = mask & ~(up & down & left & right)
    tinted[boundary] = np.array([1.0, 0.90, 0.08], dtype=np.float32)
    return Image.fromarray((np.clip(tinted, 0, 1) * 255).astype(np.uint8), mode="RGB")
