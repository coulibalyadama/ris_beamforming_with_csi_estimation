from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainingConfig
from data_generation.dataset import train_validation_split
from ml_model.model import AngleEstimatorMLP, NormalizationStats, compute_normalization_stats, normalize_features


@dataclass(frozen=True)
class TrainingHistory:
    best_val_loss: float
    train_loss: list[float]
    validation_loss: list[float]


def make_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    validation_split: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, NormalizationStats]:
    train_x, train_y, val_x, val_y = train_validation_split(features, targets, validation_split, seed)
    train_tensor = torch.from_numpy(train_x)
    stats = compute_normalization_stats(train_tensor)
    train_tensor = normalize_features(train_tensor, stats)
    val_tensor = normalize_features(torch.from_numpy(val_x), stats)

    train_dataset = TensorDataset(train_tensor, torch.from_numpy(train_y))
    val_dataset = TensorDataset(val_tensor, torch.from_numpy(val_y))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, stats


def train_model(
    model: AngleEstimatorMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device | str | None = None,
    save_path: str | None = None,
    stats: NormalizationStats | None = None,
) -> TrainingHistory:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    best_val_loss = float("inf")

    for _ in range(epochs):
        model.train()
        train_total_loss = 0.0
        train_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = len(batch_features)
            train_total_loss += float(loss.item()) * batch_size
            train_count += batch_size

        val_loss = evaluate_model(model, val_loader, criterion, device)
        train_loss_history.append(train_total_loss / max(1, train_count))
        validation_loss_history.append(val_loss)

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save({"state_dict":model.state_dict(),
                        "mean": stats.mean,
                        "std": stats.std}, 
                       save_path)

        print(f"Epoch {_ + 1}/{epochs} - Train Loss: {train_total_loss / max(1, train_count):.6f} - Validation Loss: {val_loss:.6f}")

    return TrainingHistory(best_val_loss=best_val_loss, train_loss=train_loss_history, validation_loss=validation_loss_history)


def evaluate_model(
    model: AngleEstimatorMLP,
    data_loader: DataLoader,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> float:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = criterion or nn.MSELoss()
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            batch_size = len(batch_features)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
    return total_loss / max(1, total_examples)


def predict_angles(
    model: AngleEstimatorMLP,
    features: np.ndarray,
    stats: NormalizationStats,
    device: torch.device | str | None = None,
) -> np.ndarray:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        tensor_features = normalize_features(torch.from_numpy(features), stats).to(device)
        return model(tensor_features).cpu().numpy()
