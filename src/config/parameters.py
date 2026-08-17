from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeometryConfig:
    """Rectangular RIS geometry."""

    rows: int = 10
    cols: int = 10
    element_spacing_lambda: float = 0.5

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be strictly positive")
        if self.element_spacing_lambda <= 0:
            raise ValueError("element_spacing_lambda must be positive")

    @property
    def num_elements(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class ChannelConfig:
    """Parameters used to generate synthetic channel samples."""

    carrier_frequency_hz: float = 30e9
    beta0_db: float = -20.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    user_distance_range_m: tuple[float, float] = (5.0, 25.0)
    azimuth_range_rad: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    elevation_range_rad: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    snr_db_range: tuple[float, float] = (0.0, 40.0)
    pilot_length: int = 1
    pilot_power: float = 1.0

    def validate(self) -> None:
        if self.user_distance_range_m[0] <= 0 or self.user_distance_range_m[0] >= self.user_distance_range_m[1]:
            raise ValueError("user_distance_range_m must be an increasing positive interval")
        if self.snr_db_range[0] > self.snr_db_range[1]:
            raise ValueError("snr_db_range must be increasing")
        if self.pilot_length <= 0:
            raise ValueError("pilot_length must be positive")
        if self.pilot_power <= 0:
            raise ValueError("pilot_power must be positive")

    @property
    def beta0_linear(self) -> float:
        return 10 ** (self.beta0_db / 10.0)


@dataclass(frozen=True)
class TrainingConfig:
    train_samples: int = 80_000
    test_samples: int = 20_000
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_layers: tuple[int, ...] = (8, 8, 8)
    validation_split: float = 0.2
    seed: int = 1234

    def validate(self) -> None:
        if self.train_samples <= 0 or self.test_samples <= 0:
            raise ValueError("sample counts must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be in (0, 1)")
        if not self.hidden_layers:
            raise ValueError("hidden_layers cannot be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    geometry: GeometryConfig
    channel: ChannelConfig
    training: TrainingConfig

    def validate(self) -> None:
        self.geometry.validate()
        self.channel.validate()
        self.training.validate()
