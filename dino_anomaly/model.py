from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DetectorState:
    memory_bank: torch.Tensor
    model_name: str
    image_size: int
    category: str
    layers: tuple[int, ...]
    sampling: str
    image_threshold: float
    pixel_threshold: float


def conformal_upper_threshold(scores: torch.Tensor, target_fpr: float) -> float:
    """Select a finite-sample-corrected upper normal-score threshold.

    This follows the split-conformal order statistic when that rank is observable.
    If the requested rank is n + 1, the maximum observed score is used instead
    of an infinite threshold; very small calibration sets therefore cannot
    guarantee the requested false-positive rate.
    """
    if scores.numel() == 0:
        raise ValueError("Cannot calibrate a threshold from zero scores")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    values = scores.detach().flatten().float().sort().values
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - target_fpr)))
    return float(values[rank - 1])


def empirical_upper_threshold(scores: torch.Tensor, target_fpr: float) -> float:
    """Select an empirical upper quantile for correlated normal pixel scores."""
    if scores.numel() == 0:
        raise ValueError("Cannot calibrate a threshold from zero scores")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    return float(torch.quantile(scores.detach().flatten().float(), 1.0 - target_fpr))


class DinoAnomalyDetector:
    """Frozen multi-layer DINOv2 encoder and a normal-patch memory bank."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        image_size: int,
        layers: Iterable[int] = (8, 11),
        query_chunk_size: int = 1024,
        encoder: nn.Module | None = None,
    ) -> None:
        if image_size % 14:
            raise ValueError("image_size must be divisible by the DINOv2 patch size (14)")
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.layers = tuple(sorted(set(layers)))
        if not self.layers or self.layers[0] < 0:
            raise ValueError("layers must contain non-negative transformer block indices")
        self.query_chunk_size = query_chunk_size
        if encoder is None:
            encoder = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                trust_repo=True,
                skip_validation=True,
            )
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        depth = len(self.encoder.blocks)
        if self.layers[-1] >= depth:
            raise ValueError(
                f"Layer {self.layers[-1]} is invalid for {model_name} with {depth} blocks"
            )
        self.memory_bank: torch.Tensor | None = None
        self.sampling = "unfitted"
        self.image_threshold: float | None = None
        self.pixel_threshold: float | None = None

    @torch.inference_mode()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder.get_intermediate_layers(
            images.to(self.device), n=self.layers, norm=True
        )
        outputs = [F.normalize(output, dim=-1) for output in outputs]
        return F.normalize(torch.cat(outputs, dim=-1), dim=-1)

    @staticmethod
    def _random_sample(features: torch.Tensor, count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(features), generator=generator)[:count]
        return features[indices]

    @staticmethod
    def _coreset_sample(
        features: torch.Tensor,
        count: int,
        seed: int,
        projection_dim: int,
        max_candidates: int,
    ) -> torch.Tensor:
        """Approximate greedy k-center selection in a random projection."""
        generator = torch.Generator().manual_seed(seed)
        candidates = features
        if len(candidates) > max_candidates:
            indices = torch.randperm(len(candidates), generator=generator)[
                :max_candidates
            ]
            candidates = candidates[indices]
        count = min(count, len(candidates))
        if count == len(candidates):
            return candidates

        projected_dim = min(projection_dim, candidates.shape[1])
        projection = torch.randn(
            candidates.shape[1], projected_dim, generator=generator
        ) / math.sqrt(projected_dim)
        projected = F.normalize(candidates.float() @ projection, dim=1)

        center = projected.mean(dim=0, keepdim=True)
        first = int((projected @ center.T).squeeze(1).argmax())
        selected = torch.empty(count, dtype=torch.long)
        selected[0] = first
        min_distances = 1.0 - projected @ projected[first]

        for position in range(1, count):
            next_index = int(min_distances.argmax())
            selected[position] = next_index
            distances = 1.0 - projected @ projected[next_index]
            min_distances = torch.minimum(min_distances, distances)

        return candidates[selected]

    @torch.inference_mode()
    def fit(
        self,
        loader,
        max_memory_patches: int,
        seed: int,
        sampling: str = "coreset",
        coreset_projection_dim: int = 32,
        coreset_max_candidates: int = 25_000,
    ) -> dict[str, int | str]:
        feature_batches = []
        for batch in loader:
            features = self.extract_patch_features(batch["image"])
            feature_batches.append(features.flatten(0, 1).cpu())

        all_features = torch.cat(feature_batches, dim=0)
        target_count = min(max_memory_patches, len(all_features))
        if sampling == "coreset":
            memory_bank = self._coreset_sample(
                all_features,
                target_count,
                seed,
                coreset_projection_dim,
                coreset_max_candidates,
            )
        elif sampling == "random":
            memory_bank = self._random_sample(all_features, target_count, seed)
        else:
            raise ValueError("sampling must be 'coreset' or 'random'")

        self.memory_bank = memory_bank.contiguous()
        self.sampling = sampling
        self.image_threshold = None
        self.pixel_threshold = None
        return {
            "available_patches": len(all_features),
            "memory_patches": len(memory_bank),
            "sampling": sampling,
        }

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("Detector has not been fitted or loaded")

        features = self.extract_patch_features(images)
        batch_size, num_patches, _ = features.shape
        flat_features = features.reshape(-1, features.shape[-1])
        memory = self.memory_bank.to(self.device)

        scores = []
        for query in flat_features.split(self.query_chunk_size):
            nearest_similarity = query @ memory.T
            scores.append(1.0 - nearest_similarity.max(dim=1).values)
        patch_scores = torch.cat(scores).reshape(batch_size, num_patches)

        grid_size = self.image_size // 14
        anomaly_maps = patch_scores.reshape(batch_size, 1, grid_size, grid_size)
        anomaly_maps = F.interpolate(
            anomaly_maps,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        k = max(1, num_patches // 100)
        image_scores = patch_scores.topk(k, dim=1).values.mean(dim=1)
        return image_scores.cpu(), anomaly_maps.cpu()

    @torch.inference_mode()
    def calibrate(
        self,
        loader,
        target_image_fpr: float,
        target_pixel_fpr: float,
    ) -> dict[str, float | int]:
        image_scores = []
        pixel_scores = []
        for batch in loader:
            scores, maps = self.predict(batch["image"])
            image_scores.append(scores)
            pixel_scores.append(maps.flatten())
        all_image_scores = torch.cat(image_scores)
        all_pixel_scores = torch.cat(pixel_scores)
        self.image_threshold = conformal_upper_threshold(
            all_image_scores, target_image_fpr
        )
        self.pixel_threshold = empirical_upper_threshold(
            all_pixel_scores, target_pixel_fpr
        )
        return {
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "num_validation_images": len(all_image_scores),
            "target_image_fpr": target_image_fpr,
            "target_pixel_fpr": target_pixel_fpr,
        }

    def save(self, path: str | Path, category: str) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save an unfitted detector")
        if self.image_threshold is None or self.pixel_threshold is None:
            raise RuntimeError("Calibrate validation thresholds before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "model_name": self.model_name,
                "image_size": self.image_size,
                "category": category,
                "layers": list(self.layers),
                "sampling": self.sampling,
                "image_threshold": self.image_threshold,
                "pixel_threshold": self.pixel_threshold,
            },
            path,
        )

    def load(self, path: str | Path) -> DetectorState:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_layers = tuple(checkpoint.get("layers", (11,)))
        if checkpoint["model_name"] != self.model_name:
            raise ValueError("Checkpoint model_name does not match the requested model")
        if checkpoint["image_size"] != self.image_size:
            raise ValueError("Checkpoint image_size does not match the requested size")
        if checkpoint_layers != self.layers:
            raise ValueError("Checkpoint feature layers do not match requested layers")
        if "image_threshold" not in checkpoint or "pixel_threshold" not in checkpoint:
            raise ValueError("Legacy checkpoint has no validation thresholds; refit it")
        self.memory_bank = checkpoint["memory_bank"].contiguous()
        self.sampling = checkpoint.get("sampling", "random")
        self.image_threshold = float(checkpoint["image_threshold"])
        self.pixel_threshold = float(checkpoint["pixel_threshold"])
        return DetectorState(
            memory_bank=self.memory_bank,
            model_name=checkpoint["model_name"],
            image_size=checkpoint["image_size"],
            category=checkpoint["category"],
            layers=checkpoint_layers,
            sampling=self.sampling,
            image_threshold=self.image_threshold,
            pixel_threshold=self.pixel_threshold,
        )
