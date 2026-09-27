"""
Synthetic AWS Sensor Data Generator.

Generates realistic, physically-grounded meteorological time-series for:
- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)
- Wind Speed (m/s)

Supports injecting real-world anomalies:
- Missing readings and sentinel values (-999, -9999)
- Physics violations (RH > 100%, Dew Point > Temp, negative wind speeds)
- Statistical anomalies (instantaneous spikes, sensor flatlines/stuck states)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .physics_checks import calculate_saturation_vapor_pressure


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic AWS data generation."""
    start_time: str = "2026-09-01 00:00:00"
    duration_days: float = 7.0
    frequency_minutes: int = 5
    
    # Baseline meteorological conditions
    mean_temp_c: float = 22.0
    temp_diurnal_amplitude_c: float = 8.0
    mean_pressure_hpa: float = 1013.25
    pressure_synoptic_amplitude_hpa: float = 12.0
    mean_dew_point_c: float = 14.0
    wind_weibull_k: float = 2.0
    wind_weibull_c: float = 5.5
    
    # Anomaly injection settings
    inject_anomalies: bool = True
    random_seed: int = 42


def generate_synthetic_aws_data(
    config: Optional[SyntheticDataConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic AWS time-series data with realistic diurnal cycles and injected anomalies.

    Returns:
        Tuple of (sensor_data_df, ground_truth_anomalies_df)
    """
    if config is None:
        config = SyntheticDataConfig()

    np.random.seed(config.random_seed)

    # Generate time range
    start_dt = pd.to_datetime(config.start_time)
    num_steps = int((config.duration_days * 24 * 60) / config.frequency_minutes)
    timestamps = pd.date_range(start=start_dt, periods=num_steps, freq=f"{config.frequency_minutes}min")

    # Time factors
    hours = timestamps.hour + timestamps.minute / 60.0
    days = (timestamps - start_dt).total_seconds() / 86400.0

    # 1. Base Temperature (°C) with diurnal cycle (minimum at 06:00, maximum at 14:00)
    diurnal_phase = (hours - 14.0) * (2 * np.pi / 24.0)
    synoptic_temp_wave = 3.0 * np.sin(2 * np.pi * days / 3.5)
    temp_noise = np.random.normal(0, 0.4, size=num_steps)
    temperature = (
        config.mean_temp_c
        + config.temp_diurnal_amplitude_c * np.cos(diurnal_phase)
        + synoptic_temp_wave
        + temp_noise
    )

    # 2. Base Atmospheric Pressure (hPa) with semi-diurnal atmospheric solar tide + synoptic front
    semi_diurnal_tide = 1.2 * np.sin(4 * np.pi * (hours - 9.0) / 24.0)
    synoptic_pressure_wave = config.pressure_synoptic_amplitude_hpa * np.cos(2 * np.pi * days / 4.0)
    pressure_noise = np.random.normal(0, 0.15, size=num_steps)
    pressure = config.mean_pressure_hpa + synoptic_pressure_wave + semi_diurnal_tide + pressure_noise

    # 3. Base Relative Humidity (%) physically derived from dew point and temperature
    # Dew point varies slowly
    dew_point_base = config.mean_dew_point_c + 2.0 * np.sin(2 * np.pi * days / 2.0) + np.random.normal(0, 0.3, size=num_steps)
    # Ensure dew point is physically below temperature by at least 1°C
    dew_point = np.minimum(dew_point_base, temperature - 1.5)
    
    # Calculate physical actual vapor pressure from dew point
    e_actual = calculate_saturation_vapor_pressure(dew_point)
    e_saturation = calculate_saturation_vapor_pressure(temperature)
    humidity = np.clip((e_actual / e_saturation) * 100.0 + np.random.normal(0, 0.5, size=num_steps), 10.0, 98.0)

    # 4. Base Wind Speed (m/s) with Weibull distribution & convective daytime peaks
    diurnal_wind_factor = 1.0 + 0.4 * np.maximum(0, np.sin((hours - 8) * np.pi / 12))
    base_wind = np.random.weibull(config.wind_weibull_k, size=num_steps) * config.wind_weibull_c
    wind_speed = np.clip(base_wind * diurnal_wind_factor, 0.0, 35.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind_speed,
    })

    ground_truth = []

    # Inject realistic real-world anomalies if enabled
    if config.inject_anomalies and num_steps > 200:
        # A. Missing / Sentinel values
        idx_sentinel_1 = int(num_steps * 0.08)
        df.loc[idx_sentinel_1:idx_sentinel_1 + 2, "temperature"] = -999.0
        ground_truth.append({"idx": idx_sentinel_1, "sensor": "temperature", "type": "MISSING", "desc": "Sentinel -999.0"})

        idx_nan_gap = int(num_steps * 0.25)
        df.loc[idx_nan_gap:idx_nan_gap + 4, "pressure"] = np.nan
        ground_truth.append({"idx": idx_nan_gap, "sensor": "pressure", "type": "MISSING", "desc": "Sensor dropout NaN gap"})

        # B. Physics Violations
        # 1. Supersaturated / Out of bounds RH
        idx_rh_sup = int(num_steps * 0.15)
        df.loc[idx_rh_sup, "humidity"] = 118.5
        ground_truth.append({"idx": idx_rh_sup, "sensor": "humidity", "type": "PHYSICS_VIOLATION", "desc": "RH exceeds 100% (118.5%)"})

        # 2. Negative RH
        idx_rh_neg = int(num_steps * 0.45)
        df.loc[idx_rh_neg, "humidity"] = -12.0
        ground_truth.append({"idx": idx_rh_neg, "sensor": "humidity", "type": "PHYSICS_VIOLATION", "desc": "Negative RH (-12%)"})

        # 3. Negative Wind Speed
        idx_wind_neg = int(num_steps * 0.35)
        df.loc[idx_wind_neg, "wind_speed"] = -4.5
        ground_truth.append({"idx": idx_wind_neg, "sensor": "wind_speed", "type": "PHYSICS_VIOLATION", "desc": "Negative wind speed (-4.5 m/s)"})

        # 4. Out of bounds Extreme Wind Speed
        idx_wind_ext = int(num_steps * 0.65)
        df.loc[idx_wind_ext, "wind_speed"] = 108.0
        ground_truth.append({"idx": idx_wind_ext, "sensor": "wind_speed", "type": "PHYSICS_VIOLATION", "desc": "Extreme unphysical wind speed (108 m/s)"})

        # 5. Climatological out-of-bounds Temperature
        idx_temp_ext = int(num_steps * 0.72)
        df.loc[idx_temp_ext, "temperature"] = 78.5
        ground_truth.append({"idx": idx_temp_ext, "sensor": "temperature", "type": "PHYSICS_VIOLATION", "desc": "Climatological high temperature (78.5°C)"})

        # 6. Pressure drop out-of-bounds
        idx_pres_ext = int(num_steps * 0.88)
        df.loc[idx_pres_ext, "pressure"] = 720.0
        ground_truth.append({"idx": idx_pres_ext, "sensor": "pressure", "type": "PHYSICS_VIOLATION", "desc": "Pressure below physical surface limits (720 hPa)"})

        # C. Statistical Anomalies
        # 1. Transient Spike (Temperature glitch)
        idx_temp_spike = int(num_steps * 0.52)
        df.loc[idx_temp_spike, "temperature"] += 14.0
        ground_truth.append({"idx": idx_temp_spike, "sensor": "temperature", "type": "SPIKE", "desc": "Transient +14°C spike"})

        # 2. Transient Spike (Pressure glitch)
        idx_pres_spike = int(num_steps * 0.60)
        df.loc[idx_pres_spike, "pressure"] += 9.5
        ground_truth.append({"idx": idx_pres_spike, "sensor": "pressure", "type": "SPIKE", "desc": "Step spike in pressure (+9.5 hPa)"})

        # 3. Stuck / Frozen Sensor (Flatline)
        idx_stuck = int(num_steps * 0.80)
        stuck_val = df.loc[idx_stuck, "temperature"]
        df.loc[idx_stuck:idx_stuck + 16, "temperature"] = stuck_val
        ground_truth.append({"idx": idx_stuck, "sensor": "temperature", "type": "STUCK", "desc": "Frozen temperature sensor for 16 intervals"})

    gt_df = pd.DataFrame(ground_truth)
    return df, gt_df
