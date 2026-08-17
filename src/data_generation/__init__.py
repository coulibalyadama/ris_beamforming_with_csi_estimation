"""Synthetic dataset generation for the RIS channel estimation pipeline."""

from .dataset import ChannelSample, Dataset, build_dataset, train_validation_split

__all__ = ["ChannelSample", "Dataset", "build_dataset", "train_validation_split"]
