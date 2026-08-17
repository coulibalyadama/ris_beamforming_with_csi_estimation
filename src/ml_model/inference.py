from dataclasses import replace
from logging import config
from typing import List

import torch
import numpy as np

from config import ChannelConfig, GeometryConfig
from ml_model.model import AngleEstimatorMLP, NormalizationStats
from ml_model.training import predict_angles
from system_model.channel import sample_parameters, los_channel, steering_vector
from system_model.signal import pilot_observation

# RIS parameters
geometry = GeometryConfig(rows=10, cols=10, element_spacing_lambda=0.5)


# Base station parameters
bs_config = ChannelConfig(
    carrier_frequency_hz=30e9,
    beta0_db=-20.0,
    reference_distance_m=1.0,
    path_loss_exponent=2.2,
    user_distance_range_m=(20.0, 20.0),
    azimuth_range_rad=(-np.pi / 2, np.pi / 2),
    elevation_range_rad=(-np.pi / 2, np.pi / 2),
    snr_db_range=(0.0, 40.0),
    pilot_length=1,
    pilot_power=1.0,
)

# User parameters
user_config = ChannelConfig(
    carrier_frequency_hz=30e9,
    beta0_db=-20.0,
    reference_distance_m=1.0,
    path_loss_exponent=2.2,
    user_distance_range_m=(10.0, 10.0),
    azimuth_range_rad=(-np.pi / 2, np.pi / 2),
    elevation_range_rad=(-np.pi / 2, np.pi / 2),
    snr_db_range=(0.0, 40.0),
    pilot_length=1,
    pilot_power=1.0,
)

# Model size
sample_sizes = [1_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]

def choose_codeword(channels, codebook):
    h_ue, h_bs = channels
    power_per_codeword = np.zeros(len(codebook))

    for idx, codeword in enumerate(codebook):
        y = np.conj(h_bs).T @ (codeword * h_ue)
        power_per_codeword[idx] = np.abs(y) ** 2

    best_idx = int(np.argmax(power_per_codeword))
    return codebook[best_idx], power_per_codeword[best_idx]

def exhaustive_search(channels, channels_pred, user_config, codebook, stats, model,n_pilot, rng):
    h_ue, _ = channels
    _, h_bs_pred = channels_pred
    power_per_codeword = np.zeros(len(codebook))

    for idx, codeword in enumerate(codebook):
        total_power = 0.0
        for _ in range(n_pilot):
            distance, azimuth, elevation, u, v, gain_ue, snr_db = sample_parameters(user_config, rng)
            y_ue = pilot_observation(user_config, h_ue, user_config.snr_db_range[0], rng)
            features_ue = np.concatenate([y_ue.real, y_ue.imag]).astype(np.float32)
            pred_u, pred_v = predict_angles(model, features_ue[None, :], stats)[0]
            h_ue_pred = los_channel(geometry, gain=gain_ue, u=pred_u, v=pred_v)
            y = np.conj(h_bs_pred).T @ (codeword @ h_ue_pred)
            total_power += np.abs(y) ** 2
        power_per_codeword[idx] = total_power

    best_idx = int(np.argmax(power_per_codeword))
    return codebook[best_idx], power_per_codeword[best_idx]

def exhaustive_search1(channels, codebook):
    """Teste TOUS les codewords du codebook et retourne celui qui maximise
    la puissance pour les canaux donnés."""
    h_ue, h_bs = channels
    powers = np.array([
        np.abs(h_bs.T @ (codeword * h_ue)) ** 2
        for codeword in codebook
    ])
    best_idx = int(np.argmax(powers))
    return codebook[best_idx], powers[best_idx]

def build_codebook(geometry: GeometryConfig, seed: int, size: int) -> np.ndarray:
    """Build a codebook for the RIS reflective elements."""
    N_RIS = geometry.rows * geometry.cols
    
    rng = np.random.default_rng(seed=seed)
    angles = rng.uniform(-np.pi/2, np.pi/2, size=(size, N_RIS))
    codebook = np.exp(1j * angles)

    return codebook.astype(np.complex128)

def get_powers():
    rng = np.random.default_rng(seed=42)
    codebook = build_codebook(geometry, seed=0, size=1000)
    real_power = []
    predicted_power = []

    # Generate a random user channel sample
    distance, azimuth, elevation, u, v, gain_ue, snr_db = sample_parameters(user_config, rng)
    h_ue = los_channel(geometry, gain=gain_ue, u=u, v=v)
    signal_ue = pilot_observation(user_config, h_ue, snr_db, rng)
    features_ue = np.concatenate([signal_ue.real, signal_ue.imag]).astype(np.float32)

    # Generate a random base station channel sample
    distance, azimuth, elevation, u, v, gain_bs, snr_db = sample_parameters(bs_config, rng)
    h_bs = los_channel(geometry, gain=gain_bs, u=u, v=v)
    signal_bs = pilot_observation(bs_config, h_bs, snr_db, rng)
    features_bs = np.concatenate([signal_bs.real, signal_bs.imag]).astype(np.float32)


    for size in sample_sizes:
        # Load the model
        model_path = f"artefacts/model_{size}.pt"
        model = AngleEstimatorMLP(input_dim=200, hidden_layers=[8, 8, 8], output_dim=2)
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        stats = NormalizationStats(mean=checkpoint["mean"], std=checkpoint["std"])

        predicted_angles = predict_angles(model, features_ue[None, :], stats)

        # Reconstruct the channel using the predicted angles
        predicted_u, predicted_v = predicted_angles[0]
        h_ue_pred = los_channel(geometry, gain=gain_ue, u=predicted_u, v=predicted_v)

        
        predicted_angles = predict_angles(model, features_bs[None, :], stats)

        _, pow_real = choose_codeword((h_ue, h_bs), codebook)

        # Reconstruct the channel using the predicted angles
        predicted_u, predicted_v = predicted_angles[0]
        h_bs_pred = los_channel(geometry, gain=gain_bs, u=predicted_u, v=predicted_v)

        best_code, _ = exhaustive_search((h_ue_pred, h_bs_pred), codebook)

        y = h_bs.T @ (np.diag(best_code) @ h_ue) 
        achieved_power = np.sum(np.abs(y)**2)
        
        predicted_power.append(achieved_power)
        real_power.append(pow_real)

    return real_power, predicted_power
    

def powerPilotNumber(n_pilots: List[int], model_path: str, n_trials: int = 50000, seed: int = 42, snr_db: float=10.0, n_samples: int=60000) -> tuple[List[float], List[float]]:
    model = AngleEstimatorMLP(input_dim=200, hidden_layers=[8, 8, 8], output_dim=2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = NormalizationStats(mean=checkpoint["mean"], std=checkpoint["std"])

    codebook = build_codebook(geometry, seed=0, size=10)

    maximum_powers, achieved_powers = [], []

    for n_pilot in n_pilots:
        # Configs avec la longueur de pilote testée, tout le reste inchangé
        user_cfg = replace(user_config, pilot_length=n_pilot, snr_db_range=(snr_db, snr_db))
        bs_cfg = replace(bs_config, pilot_length=n_pilot, snr_db_range=(snr_db, snr_db))

        rng = np.random.default_rng(seed=seed)  # même tirage pour tous les n_pilot -> comparaison équitable
        real_power, achieved_power = [], []

        for _ in range(n_trials):
            # Canal UE
            distance, azimuth, elevation, u, v, gain_ue, snr_db = sample_parameters(user_cfg, rng)
            h_ue = los_channel(geometry, gain=gain_ue, u=u, v=v)
            signal_ue = pilot_observation(user_cfg, h_ue, snr_db, rng)
            features_ue = np.concatenate([signal_ue.real, signal_ue.imag]).astype(np.float32)

            # Canal BS
            distance, azimuth, elevation, u, v, gain_bs, snr_db = sample_parameters(bs_cfg, rng)
            h_bs = los_channel(geometry, gain=gain_bs, u=u, v=v)
            signal_bs = pilot_observation(bs_cfg, h_bs, snr_db, rng)
            features_bs = np.concatenate([signal_bs.real, signal_bs.imag]).astype(np.float32)

            # Puissance optimale avec les vrais canaux
            _, pow_real = choose_codeword((h_ue, h_bs), codebook)

            # Angles prédits par le modèle (gain UE/BS correctement séparés)
            pred_u_ue, pred_v_ue = predict_angles(model, features_ue[None, :], stats)[0]
            h_ue_pred = los_channel(geometry, gain=gain_ue, u=pred_u_ue, v=pred_v_ue)

            pred_u_bs, pred_v_bs = predict_angles(model, features_bs[None, :], stats)[0]
            h_bs_pred = los_channel(geometry, gain=gain_bs, u=pred_u_bs, v=pred_v_bs)

            # Code choisi à partir des canaux prédits, mais appliqué aux VRAIS canaux
            best_codeword, _ = choose_codeword((h_ue_pred, h_bs_pred), codebook)
            y = np.conj(h_bs).T @ (np.diag(best_codeword) @ h_ue)
            pow_achieved = np.sum(np.abs(y) ** 2)

            real_power.append(pow_real)
            achieved_power.append(pow_achieved)

        maximum_powers.append(float(np.mean(real_power)))
        achieved_powers.append(float(np.mean(achieved_power)))

    save_path = f"data/power_vs_pilot_number_{n_samples}_{snr_db}dB.dat"
    ratios = np.array(achieved_powers) / np.array(maximum_powers)
    np.savetxt(save_path, list(zip(n_pilots, ratios)), delimiter=",", header="Pilot Number,Ratio")

    return maximum_powers, achieved_powers
