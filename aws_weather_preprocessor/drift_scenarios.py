"""
Synthetic Sensor Drift and Bias Simulation Scenarios.

Generates realistic time-series with ground-truth true atmospheric states
and injected instrument drift profiles:
1. Linear Ramp Drift (slow sensor aging / fouling)
2. Step Calibration Offset (sudden bias jump)
3. Brownian Walk Drift (stochastic cumulative drift)
4. Multi-instrument drift scenarios with periodic reference anchoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .synthetic_data import generate_synthetic_aws_data, SyntheticDataConfig


@dataclass
class DriftExperimentResult:
    """Container for synthetic drift benchmark data."""
    true_states_df: pd.DataFrame
    observed_drift_df: pd.DataFrame
    true_biases_df: pd.DataFrame
    reference_checks_df: Optional[pd.DataFrame] = None


def generate_drift_scenario(
    scenario_type: str = "linear_ramp",
    duration_days: float = 7.0,
    frequency_minutes: int = 5,
    random_seed: int = 42
) -> DriftExperimentResult:
    """
    Generate synthetic dataset with true weather and ground truth sensor biases.

    Args:
        scenario_type: 'linear_ramp', 'step_jump', 'brownian_drift', or 'multi_fault'
        duration_days: Duration of the time-series in days.
        frequency_minutes: Sampling interval in minutes.
        random_seed: Random seed for reproducibility.

    Returns:
        DriftExperimentResult with true states, observed readings, and true biases.
    """
    np.random.seed(random_seed)

    # 1. Generate clean physical atmospheric time-series (true weather)
    config = SyntheticDataConfig(
        duration_days=duration_days,
        frequency_minutes=frequency_minutes,
        inject_anomalies=False,
        random_seed=random_seed
    )
    true_df, _ = generate_synthetic_aws_data(config)
    num_steps = len(true_df)
    t_norm = np.linspace(0, duration_days, num_steps)

    # Initialize true bias profiles
    b_temp = np.zeros(num_steps, dtype=np.float64)
    b_pres = np.zeros(num_steps, dtype=np.float64)
    b_rh = np.zeros(num_steps, dtype=np.float64)
    b_wind = np.zeros(num_steps, dtype=np.float64)

    if scenario_type == "linear_ramp":
        # Linear drift: Temp drifts +1.5°C over 7 days, Pressure drifts -2.5 hPa, RH drifts +5%
        b_temp = 0.25 * t_norm                     # 0 -> +1.75 °C
        b_pres = -0.40 * t_norm                    # 0 -> -2.80 hPa
        b_rh = 0.80 * t_norm                       # 0 -> +5.60 %
        b_wind = 0.15 * t_norm                     # 0 -> +1.05 m/s

    elif scenario_type == "step_jump":
        # Sudden calibration jump on day 3
        jump_idx = int(num_steps * 0.42)
        b_temp[jump_idx:] = 2.2                    # +2.2 °C offset
        b_pres[jump_idx:] = -3.0                   # -3.0 hPa offset
        b_rh[jump_idx:] = 6.5                      # +6.5 % offset

    elif scenario_type == "brownian_drift":
        # Stochastic random walk drift
        b_temp = np.cumsum(np.random.normal(0, 0.015, size=num_steps))
        b_pres = np.cumsum(np.random.normal(0, 0.012, size=num_steps))
        b_rh = np.cumsum(np.random.normal(0, 0.035, size=num_steps))
        b_wind = np.cumsum(np.random.normal(0, 0.010, size=num_steps))

    elif scenario_type == "multi_fault":
        # Combined: ramp on temperature, step on pressure, random walk on RH
        b_temp = 0.3 * t_norm
        jump_idx = int(num_steps * 0.5)
        b_pres[jump_idx:] = -2.5
        b_rh = np.cumsum(np.random.normal(0, 0.03, size=num_steps))

    # Add white measurement noise
    noise_temp = np.random.normal(0, 0.2, size=num_steps)
    noise_pres = np.random.normal(0, 0.15, size=num_steps)
    noise_rh = np.random.normal(0, 0.8, size=num_steps)
    noise_wind = np.random.normal(0, 0.3, size=num_steps)

    # Construct observed corrupted readings: z = true + bias + noise
    obs_df = pd.DataFrame({
        "timestamp": true_df["timestamp"],
        "temperature": true_df["temperature"] + b_temp + noise_temp,
        "pressure": true_df["pressure"] + b_pres + noise_pres,
        "humidity": np.clip(true_df["humidity"] + b_rh + noise_rh, 0, 100),
        "wind_speed": np.clip(true_df["wind_speed"] + b_wind + noise_wind, 0, None),
    })

    # True biases DataFrame
    biases_df = pd.DataFrame({
        "timestamp": true_df["timestamp"],
        "temperature_true_bias": b_temp,
        "pressure_true_bias": b_pres,
        "humidity_true_bias": b_rh,
        "wind_speed_true_bias": b_wind,
    })

    # Optional reference checkpoints (e.g. daily ground calibration check at 12:00)
    ref_df = pd.DataFrame(index=true_df.index)
    ref_df["timestamp"] = true_df["timestamp"]
    hours = pd.to_datetime(true_df["timestamp"]).dt.hour
    minutes = pd.to_datetime(true_df["timestamp"]).dt.minute
    # Reference inspection once per day at 12:00
    is_ref_check = (hours == 12) & (minutes == 0)
    for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
        ref_series = pd.Series(np.nan, index=true_df.index)
        ref_series[is_ref_check] = true_df.loc[is_ref_check, ch]
        ref_df[ch] = ref_series

    return DriftExperimentResult(
        true_states_df=true_df,
        observed_drift_df=obs_df,
        true_biases_df=biases_df,
        reference_checks_df=ref_df,
    )
