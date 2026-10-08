from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _threshold_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))

    def safe_divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "accuracy": safe_divide(true_positive + true_negative, len(labels)),
        "precision": safe_divide(true_positive, true_positive + false_positive),
        "recall": safe_divide(true_positive, true_positive + false_negative),
        "specificity": safe_divide(true_negative, true_negative + false_positive),
        "f1": safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "true_positives": true_positive,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
    }


def evaluate(detector, loader) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    if detector.image_threshold is None or detector.pixel_threshold is None:
        raise RuntimeError("Detector thresholds must be calibrated before evaluation")
    image_labels, image_scores = [], []
    pixel_labels, pixel_scores = [], []
    examples: list[dict[str, object]] = []

    for batch in loader:
        scores, maps = detector.predict(batch["image"])
        labels = torch.as_tensor(batch["label"])
        masks = batch["mask"]

        image_labels.extend(labels.numpy().tolist())
        image_scores.extend(scores.numpy().tolist())
        pixel_labels.append(masks.numpy().reshape(-1))
        pixel_scores.append(maps.numpy().reshape(-1))

        for index in range(len(scores)):
            examples.append(
                {
                    "path": batch["path"][index],
                    "defect_type": batch["defect_type"][index],
                    "label": int(labels[index]),
                    "score": float(scores[index]),
                    "prediction": int(scores[index] > detector.image_threshold),
                    "anomaly_map": maps[index, 0].numpy(),
                    "mask": masks[index, 0].numpy(),
                }
            )

    image_labels_np = np.asarray(image_labels)
    image_scores_np = np.asarray(image_scores)
    pixel_labels_np = np.concatenate(pixel_labels).astype(np.int8)
    pixel_scores_np = np.concatenate(pixel_scores)
    image_predictions = (image_scores_np > detector.image_threshold).astype(np.int8)
    pixel_predictions = (pixel_scores_np > detector.pixel_threshold).astype(np.int8)
    image_operating_point = _threshold_metrics(image_labels_np, image_predictions)
    pixel_operating_point = _threshold_metrics(pixel_labels_np, pixel_predictions)
    metrics = {
        "image_auroc": float(roc_auc_score(image_labels_np, image_scores_np)),
        "image_average_precision": float(
            average_precision_score(image_labels_np, image_scores_np)
        ),
        "pixel_auroc": float(roc_auc_score(pixel_labels_np, pixel_scores_np)),
        "pixel_average_precision": float(
            average_precision_score(pixel_labels_np, pixel_scores_np)
        ),
        "num_test_images": len(image_labels),
        "image_threshold": detector.image_threshold,
        "pixel_threshold": detector.pixel_threshold,
    }
    metrics.update({f"image_{key}": value for key, value in image_operating_point.items()})
    metrics.update({f"pixel_{key}": value for key, value in pixel_operating_point.items()})
    return metrics, examples


def save_report(
    metrics: dict[str, float | int],
    examples: list[dict[str, object]],
    output_dir: str | Path,
    max_visualizations: int,
) -> None:
    output_dir = Path(output_dir)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    ranked = sorted(examples, key=lambda item: item["score"], reverse=True)
    anomalies = [item for item in ranked if item["label"] == 1]
    normal = [item for item in ranked if item["label"] == 0]
    selected = (anomalies[:max_visualizations] + normal[:2])[: max_visualizations + 2]

    for rank, item in enumerate(selected):
        image = np.asarray(Image.open(item["path"]).convert("RGB"))
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(image)
        axes[1].imshow(item["anomaly_map"], cmap="jet", alpha=0.5)
        decision = "anomaly" if item["prediction"] else "normal"
        axes[1].set_title(f"Score: {item['score']:.4f} ({decision})")
        axes[2].imshow(item["mask"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Ground truth")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(str(item["defect_type"]))
        figure.tight_layout()
        filename = f"{rank:02d}_{item['defect_type']}_{Path(item['path']).stem}.png"
        figure.savefig(visualization_dir / filename, dpi=130, bbox_inches="tight")
        plt.close(figure)
