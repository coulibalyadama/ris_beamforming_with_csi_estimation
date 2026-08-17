from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AngleEstimatorMLP(nn.Module):
    """Simple fully connected regressor for the two angle parameters."""

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (8, 8, 8), output_dim: int = 2):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.Tanh())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor


def compute_normalization_stats(features: torch.Tensor) -> NormalizationStats:
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-8)
    return NormalizationStats(mean=mean, std=std)


def normalize_features(features: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (features - stats.mean) / stats.std
