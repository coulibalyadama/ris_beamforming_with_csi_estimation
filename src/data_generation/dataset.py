from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ChannelConfig, GeometryConfig
from system_model.channel import los_channel, sample_parameters
from system_model.signal import pilot_observation


@dataclass(frozen=True)
class ChannelSample:
    observed_channel: np.ndarray
    full_channel: np.ndarray
    features: np.ndarray
    target_angles: np.ndarray
    distance_m: float
    azimuth_rad: float
    elevation_rad: float
    snr_db: float


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    targets: np.ndarray
    samples: list[ChannelSample]


def feature_vector(observed_channel: np.ndarray) -> np.ndarray:
    return np.concatenate([observed_channel.real, observed_channel.imag]).astype(np.float32)


def create_sample(geometry: GeometryConfig, channel: ChannelConfig, rng: np.random.Generator) -> ChannelSample:
    distance, azimuth, elevation, u, v, gain, snr_db = sample_parameters(channel, rng)
    full_channel = los_channel(geometry, gain=gain, u=u, v=v)
    observed_channel = pilot_observation(geometry, channel, full_channel, snr_db, rng)
    features = feature_vector(observed_channel)
    return ChannelSample(
        observed_channel=observed_channel,
        full_channel=full_channel,
        features=features,
        target_angles=np.asarray([u, v], dtype=np.float32),
        distance_m=distance,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        snr_db=snr_db,
    )


def build_dataset(geometry: GeometryConfig, channel: ChannelConfig, num_samples: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    samples = [create_sample(geometry, channel, rng) for _ in range(num_samples)]
    features = np.stack([sample.features for sample in samples]).astype(np.float32)
    targets = np.stack([sample.target_angles for sample in samples]).astype(np.float32)
    return Dataset(features=features, targets=targets, samples=samples)


def train_validation_split(
    features: np.ndarray,
    targets: np.ndarray,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be in (0, 1)")
    indices = np.arange(len(features))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    validation_size = max(1, int(round(len(features) * validation_split)))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return (
        features[training_indices],
        targets[training_indices],
        features[validation_indices],
        targets[validation_indices],
    )
