from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import colormaps

from .data import image_transform


@dataclass
class ImagePrediction:
    score: float
    prediction: str
    anomaly_map: np.ndarray
    resized_image: Image.Image
    heatmap: Image.Image
    overlay: Image.Image


def render_anomaly_images(
    image: Image.Image,
    anomaly_map: np.ndarray,
    pixel_threshold: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Create display, heatmap, and overlay images with a stable color scale."""
    height, width = anomaly_map.shape
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    image_array = np.asarray(resized, dtype=np.float32)

    upper = max(float(anomaly_map.max()), pixel_threshold * 2.0, 1e-8)
    normalized = np.clip(anomaly_map / upper, 0.0, 1.0)
    colored = (colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    overlay = (0.58 * image_array + 0.42 * colored).clip(0, 255).astype(np.uint8)
    return resized, Image.fromarray(colored), Image.fromarray(overlay)


@torch.inference_mode()
def predict_pil_images(
    detector,
    images: list[Image.Image],
    image_size: int,
) -> list[ImagePrediction]:
    if not images:
        return []
    if detector.image_threshold is None or detector.pixel_threshold is None:
        raise RuntimeError("The detector checkpoint has no calibrated thresholds")

    transform = image_transform(image_size)
    batch = torch.stack([transform(image.convert("RGB")) for image in images])
    scores, maps = detector.predict(batch)

    predictions = []
    for image, score_tensor, map_tensor in zip(images, scores, maps):
        score = float(score_tensor)
        label = "anomaly" if score > detector.image_threshold else "normal"
        anomaly_map = map_tensor[0].numpy()
        resized, heatmap, overlay = render_anomaly_images(
            image, anomaly_map, detector.pixel_threshold
        )
        predictions.append(
            ImagePrediction(
                score=score,
                prediction=label,
                anomaly_map=anomaly_map,
                resized_image=resized,
                heatmap=heatmap,
                overlay=overlay,
            )
        )
    return predictions
