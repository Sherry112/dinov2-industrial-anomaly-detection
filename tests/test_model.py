import pytest
import torch

from dino_anomaly.model import (
    DinoAnomalyDetector,
    conformal_upper_threshold,
    empirical_upper_threshold,
)


def test_conformal_threshold_uses_conservative_order_statistic():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    threshold = conformal_upper_threshold(scores, target_fpr=0.2)
    assert threshold == pytest.approx(0.4)


def test_empirical_pixel_threshold_uses_requested_quantile():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    threshold = empirical_upper_threshold(scores, target_fpr=0.5)
    assert threshold == pytest.approx(0.25)


def test_coreset_selection_is_seeded_and_returns_original_features():
    generator = torch.Generator().manual_seed(7)
    features = torch.nn.functional.normalize(
        torch.randn(100, 16, generator=generator), dim=1
    )
    first = DinoAnomalyDetector._coreset_sample(features, 12, 42, 8, 80)
    second = DinoAnomalyDetector._coreset_sample(features, 12, 42, 8, 80)

    assert first.shape == (12, 16)
    assert torch.equal(first, second)
    assert all(any(torch.equal(row, source) for source in features) for row in first)


def test_invalid_false_positive_rate_is_rejected():
    with pytest.raises(ValueError):
        conformal_upper_threshold(torch.tensor([0.1]), target_fpr=0.0)


def test_detectors_can_share_one_encoder():
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(12)]
            )

    encoder = FakeEncoder()
    first = DinoAnomalyDetector(
        "fake", torch.device("cpu"), 224, encoder=encoder
    )
    second = DinoAnomalyDetector(
        "fake", torch.device("cpu"), 224, encoder=encoder
    )

    assert first.encoder is encoder
    assert second.encoder is encoder
    assert first.encoder is second.encoder
