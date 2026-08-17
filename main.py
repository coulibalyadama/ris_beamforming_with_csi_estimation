from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_model.experiment import default_experiment_config, run_experiment
from ml_model.inference import get_powers, powerPilotNumber

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive RIS channel estimation with a neural network")
    parser.add_argument("--train-samples", type=int, default=80_000)
    parser.add_argument("--test-samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[8, 8, 8])
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    base = default_experiment_config()
    geometry = replace(base.geometry, rows=args.rows, cols=args.cols)
    training = replace(
        base.training,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_layers=tuple(args.hidden_layers),
    )
    return replace(base, geometry=geometry, training=training)

def plot_training_history(train_loss: list[float], validation_loss: list[float], num_samples: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images/training_history_{num_samples}.png")
    plt.show()

def plot_powers(
    maximum_powers: list[float],
    achieved_powers: list[float],
    abscissa: list[int],
    n_samples: int,
    snr_db: float,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(abscissa, maximum_powers, marker="o", label="Maximum Power")

    plt.plot(abscissa, achieved_powers, marker="o", label="Achieved Power")

    plt.xlabel("Number of Pilots")
    plt.ylabel("Power")
    plt.title("Maximum vs Achieved Powers")

    plt.xticks(abscissa)

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"images/achieved_power_vs_maximum_power_{n_samples}_{snr_db}dB.png.png")
    plt.show()

def plot_relative_power(
    maximum_powers: list[float],
    achieved_powers: list[float],
    abscissa: list[int],
    snr_db: float,
    n_samples: int,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(abscissa, 
             np.array(achieved_powers)/np.array(maximum_powers), 
             marker="o", 
             #label="Real Power"
             )


    plt.xlabel("Number of Pilots")
    plt.ylabel("Achieved Power / Maximum Power")
    plt.title(f"Maximum Power vs Achieved Powers (SNR: {snr_db} dB)")

    plt.xticks(abscissa)

    plt.legend()
    plt.grid()
    plt.tight_layout()

    save_path = f"images/achieved_power_vs_maximum_power_ratio_{n_samples}_{snr_db}dB.png"
    plt.savefig(save_path)
    plt.show()


def main() -> None:
    # args = parse_args()
    # config = build_config(args)
    # result = run_experiment(config)
    # print("Angle MSE:", result.angle_mse_value)
    # print("Channel NMSE [dB]:", result.nmse_db_value)
    # # print(" training loss:", result.train_history[-1])
    # print("Best validation loss:", result.best_val_loss)

    # plot_training_history(result.train_history, result.validation_history, args.train_samples + args.test_samples)
    n_samples = 60_000
    snr_db = 20.0
    model_path = f"artefacts/model_{n_samples}.pt"
    n_pilots = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    maximum_powers, achieved_powers = powerPilotNumber(n_pilots=n_pilots, model_path=model_path, seed=42, snr_db=snr_db, n_trials=5000, n_samples=n_samples)
    
    plot_powers(maximum_powers, achieved_powers, abscissa=n_pilots, n_samples=n_samples, snr_db=snr_db)
    plot_relative_power(maximum_powers, achieved_powers, abscissa=n_pilots, snr_db=snr_db, n_samples=n_samples)

if __name__ == "__main__":
    main()
