from __future__ import annotations

import math

import numpy as np

from config import ChannelConfig, GeometryConfig


# def wavelength(carrier_frequency_hz: float) -> float:
#     return 299_792_458.0 / carrier_frequency_hz


def angle_parameters(azimuth_rad: float, elevation_rad: float) -> tuple[float, float]:
    u = math.cos(elevation_rad)
    v = math.sin(azimuth_rad) * math.sin(elevation_rad)
    return u, v


def steering_vector(geometry: GeometryConfig, u: float, v: float) -> np.ndarray:
    """Flattened UPA steering vector ordered row-major."""

    geometry.validate()
    phase_step = 2.0 * math.pi * geometry.element_spacing_lambda
    row_phase = np.exp(1j * phase_step * np.arange(geometry.rows, dtype=np.float64) * u)
    col_phase = np.exp(1j * phase_step * np.arange(geometry.cols, dtype=np.float64) * v)
    return np.outer(row_phase, col_phase).reshape(-1)


def path_gain(distance_m: float, channel: ChannelConfig) -> float:
    return channel.beta0_linear * (distance_m / channel.reference_distance_m) ** (-channel.path_loss_exponent)


def los_channel(geometry: GeometryConfig, gain: complex, u: float, v: float) -> np.ndarray:
    return gain * steering_vector(geometry, u, v)


def estimate_complex_gain(observed_channel: np.ndarray, response: np.ndarray) -> complex:
    numerator = np.vdot(response, observed_channel)
    denominator = np.vdot(response, response)
    if denominator == 0:
        raise ValueError("reference response cannot be zero")
    return numerator / denominator


def reconstruct_channel(geometry: GeometryConfig, observed_channel: np.ndarray, predicted_u: float, predicted_v: float) -> np.ndarray:
    response = steering_vector(geometry, predicted_u, predicted_v)
    gain = estimate_complex_gain(observed_channel, response)
    return gain * response


def sample_parameters(channel: ChannelConfig, rng: np.random.Generator) -> tuple[float, float, float, float, float, float, float]:
    distance = float(rng.uniform(*channel.user_distance_range_m))
    azimuth = float(rng.uniform(*channel.azimuth_range_rad))
    elevation = float(rng.uniform(*channel.elevation_range_rad))
    u, v = angle_parameters(azimuth, elevation)
    gain = path_gain(distance, channel)
    snr_db = float(rng.uniform(*channel.snr_db_range))
    return distance, azimuth, elevation, u, v, gain, snr_db
