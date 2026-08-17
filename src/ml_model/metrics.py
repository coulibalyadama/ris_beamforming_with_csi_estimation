from __future__ import annotations

import numpy as np


def angle_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = predicted - target
    return float(np.mean(diff * diff))


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    numerator = np.linalg.norm(estimate - reference) ** 2
    denominator = np.linalg.norm(reference) ** 2
    if denominator == 0:
        raise ValueError("reference channel norm cannot be zero")
    return float(numerator / denominator)


def nmse_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(10 * np.log10(max(nmse(reference, estimate), 1e-12)))
