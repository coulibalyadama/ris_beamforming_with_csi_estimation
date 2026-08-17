from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ChannelConfig, ExperimentConfig, GeometryConfig, TrainingConfig
from data_generation.dataset import build_dataset
from ml_model.metrics import angle_mse, nmse_db
from ml_model.model import AngleEstimatorMLP
from ml_model.training import make_dataloaders, predict_angles, train_model
from system_model.channel import reconstruct_channel


@dataclass(frozen=True)
class ExperimentResult:
    model: AngleEstimatorMLP
    angle_mse_value: float
    nmse_db_value: float
    train_history: list[float]
    validation_history: list[float]
    best_val_loss: float


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        geometry=GeometryConfig(rows=10, cols=10, element_spacing_lambda=0.5),
        channel=ChannelConfig(),
        training=TrainingConfig(),
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()

    train_data = build_dataset(config.geometry, config.channel, config.training.train_samples, config.training.seed)
    test_data = build_dataset(config.geometry, config.channel, config.training.test_samples, config.training.seed + 1)

    train_loader, val_loader, stats = make_dataloaders(
        train_data.features,
        train_data.targets,
        validation_split=config.training.validation_split,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
    )

    model = AngleEstimatorMLP(
        input_dim=train_data.features.shape[1],
        hidden_layers=config.training.hidden_layers,
    )

    save_path = f"artefacts/model_{config.training.train_samples + config.training.test_samples}.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        save_path=save_path,
        stats=stats
    )

    predicted_angles = predict_angles(model, test_data.features, stats)
    predicted_channels = np.stack(
        [
            reconstruct_channel(config.geometry, sample.observed_channel, float(angles[0]), float(angles[1]))
            for sample, angles in zip(test_data.samples, predicted_angles, strict=True)
        ]
    )

    angle_error = angle_mse(predicted_angles, test_data.targets)
    channel_error_db = float(
        np.mean([nmse_db(sample.full_channel, predicted_channel) for sample, predicted_channel in zip(test_data.samples, predicted_channels, strict=True)])
    )

    return ExperimentResult(
        model=model,
        angle_mse_value=angle_error,
        nmse_db_value=channel_error_db,
        train_history=history.train_loss,
        validation_history=history.validation_loss,
        best_val_loss=history.best_val_loss,
    )
