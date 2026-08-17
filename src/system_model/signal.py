from __future__ import annotations

import math

import numpy as np

from config import ChannelConfig, GeometryConfig
from .channel import steering_vector


def pilot_sequence(channel: ChannelConfig) -> np.ndarray:
    """Generate the known pilot sequence used by the transmitter."""

    return np.ones(channel.pilot_length, dtype=np.complex128) * math.sqrt(channel.pilot_power)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(np.abs(signal) ** 2))
    if power == 0:
        raise ValueError("signal power cannot be zero")
    noise_power = power / (10 ** (snr_db / 10.0))
    sigma = math.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    return signal + noise


def pilot_observation(
    # geometry: GeometryConfig,
    channel: ChannelConfig,
    full_channel: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the noisy LS-like channel observation used as neural network input."""

    pilot = pilot_sequence(channel)
    received = full_channel[:, None] * pilot[None, :]
    noisy_received = add_awgn(received, snr_db, rng)
    pilot_norm = np.vdot(pilot, pilot)
    return noisy_received @ np.conjugate(pilot) / pilot_norm


def user_signal_projection(geometry: GeometryConfig, channel: ChannelConfig, u: float, v: float) -> np.ndarray:
    """Return the clean received signal induced by a user pilot over the RIS channel."""

    return steering_vector(geometry, u, v)[:, None] * pilot_sequence(channel)[None, :]

def signal_power(signal: np.ndarray) -> float:
    """Return the average power of a complex signal."""

    return float(np.sum(np.abs(signal) ** 2))