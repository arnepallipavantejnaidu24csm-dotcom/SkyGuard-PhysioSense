"""
Physics-based consistency checks and thermodynamic calculations for AWS sensor data.

Implements:
1. Saturation Vapor Pressure calculation (Magnus-Tetens formula)
2. Actual Vapor Pressure calculation
3. Dew Point Temperature derivation
4. Absolute and Specific Humidity calculations
5. T-P-RH consistency validation (Dew point <= Temperature, e < P, RH bounds, supersaturation)
6. Wind speed physical bounds checking (0-50 m/s nominal range, negative value checks, extreme bounds)
7. Climatological range plausibility checks for Temperature and Barometric Pressure
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class PhysicsThresholds:
    """Configurable physical and climatological thresholds for AWS sensors."""
    # Temperature limits (°C)
    min_temp_c: float = -50.0
    max_temp_c: float = 60.0
    
    # Pressure limits (hPa / mbar)
    min_pressure_hpa: float = 800.0
    max_pressure_hpa: float = 1090.0
    
    # Relative Humidity limits (%)
    min_rh_percent: float = 0.0
    max_rh_percent: float = 100.0
    rh_tolerance_percent: float = 2.0  # Allow up to 102% for sensor calibration margin
    
    # Wind speed limits (m/s)
    min_wind_speed_mps: float = 0.0
    max_wind_speed_typical_mps: float = 50.0
    max_wind_speed_extreme_mps: float = 90.0  # Category 5 hurricane / tornado threshold
    
    # Dew point tolerance (°C)
    dew_point_tolerance_c: float = 0.5  # Dew point > T + tolerance is physically invalid
    
    # Absolute humidity limits (g/m^3)
    max_absolute_humidity_gm3: float = 60.0


def calculate_saturation_vapor_pressure(
    temp_c: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate saturation vapor pressure e_s (hPa) using the Magnus-Tetens approximation.
    Valid for temperatures from -45°C to +60°C over liquid water.

    Formula:
        e_s(T) = 6.112 * exp((17.67 * T) / (T + 243.5))

    Args:
        temp_c: Dry bulb temperature in Celsius.

    Returns:
        Saturation vapor pressure in hPa.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    es = 6.112 * np.exp((17.67 * t) / (t + 243.5))
    if isinstance(temp_c, pd.Series):
        return pd.Series(es, index=temp_c.index, name="saturation_vapor_pressure_hpa")
    if np.isscalar(temp_c):
        return float(es)
    return es


def calculate_actual_vapor_pressure(
    temp_c: Union[float, np.ndarray, pd.Series],
    rh_percent: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate actual vapor pressure e (hPa) from temperature and relative humidity.

    Formula:
        e = e_s(T) * (RH / 100)

    Args:
        temp_c: Dry bulb temperature in Celsius.
        rh_percent: Relative humidity in percentage (0-100).

    Returns:
        Actual vapor pressure in hPa.
    """
    es = calculate_saturation_vapor_pressure(temp_c)
    rh = np.asarray(rh_percent, dtype=np.float64)
    # Clip negative RH to 0 for vapor pressure calculation
    rh_clipped = np.clip(rh, 0.0, None)
    e = np.asarray(es) * (rh_clipped / 100.0)
    if isinstance(temp_c, pd.Series):
        return pd.Series(e, index=temp_c.index, name="actual_vapor_pressure_hpa")
    if isinstance(rh_percent, pd.Series):
        return pd.Series(e, index=rh_percent.index, name="actual_vapor_pressure_hpa")
    if np.isscalar(temp_c) and np.isscalar(rh_percent):
        return float(e)
    return e


def calculate_dew_point(
    temp_c: Union[float, np.ndarray, pd.Series],
    rh_percent: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate dew point temperature T_d (°C) using the Magnus formula inversion.

    Formula:
        gamma(T, RH) = (17.67 * T) / (243.5 + T) + ln(RH / 100)
        T_d = (243.5 * gamma) / (17.67 - gamma)

    Args:
        temp_c: Dry bulb temperature in Celsius.
        rh_percent: Relative humidity in percentage.

    Returns:
        Dew point temperature in Celsius.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_percent, dtype=np.float64)
    
    # Avoid log of zero or negative
    rh_safe = np.clip(rh / 100.0, 1e-4, 1.5)
    gamma = (17.67 * t) / (243.5 + t) + np.log(rh_safe)
    td = (243.5 * gamma) / (17.67 - gamma)
    
    if isinstance(temp_c, pd.Series):
        return pd.Series(td, index=temp_c.index, name="dew_point_c")
    if isinstance(rh_percent, pd.Series):
        return pd.Series(td, index=rh_percent.index, name="dew_point_c")
    if np.isscalar(temp_c) and np.isscalar(rh_percent):
        return float(td)
    return td


def calculate_absolute_humidity(
    temp_c: Union[float, np.ndarray, pd.Series],
    actual_vapor_pressure_hpa: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate absolute humidity (water vapor density in g/m^3).

    Formula:
        rho_v = (216.7 * e) / (T + 273.15)

    Args:
        temp_c: Dry bulb temperature in Celsius.
        actual_vapor_pressure_hpa: Actual vapor pressure in hPa.

    Returns:
        Absolute humidity in g/m^3.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    e = np.asarray(actual_vapor_pressure_hpa, dtype=np.float64)
    ah = (216.7 * e) / (t + 273.15)
    if isinstance(temp_c, pd.Series):
        return pd.Series(ah, index=temp_c.index, name="absolute_humidity_gm3")
    if np.isscalar(temp_c) and np.isscalar(actual_vapor_pressure_hpa):
        return float(ah)
    return ah


def validate_t_p_rh_physics(
    temp_c: pd.Series,
    pressure_hpa: pd.Series,
    rh_percent: pd.Series,
    thresholds: Optional[PhysicsThresholds] = None
) -> Dict[str, pd.Series]:
    """
    Validate thermodynamic and physical consistency among Temperature, Pressure, and RH.

    Performs 6 primary checks:
    1. RH lower and upper limits (RH >= 0% and RH <= 100% + tolerance)
    2. Dew Point <= Dry Bulb Temperature (Td <= T + tolerance)
    3. Actual Vapor Pressure strictly less than Atmospheric Pressure (e < P)
    4. Absolute Humidity plausibility (0 <= AH <= max_AH)
    5. Temperature climatological bounds (min_T <= T <= max_T)
    6. Pressure climatological bounds (min_P <= P <= max_P)

    Returns:
        Dictionary of boolean Series where True indicates an anomaly / physical violation.
    """
    if thresholds is None:
        thresholds = PhysicsThresholds()

    # Calculate derived thermodynamic quantities
    es = calculate_saturation_vapor_pressure(temp_c)
    e = calculate_actual_vapor_pressure(temp_c, rh_percent)
    td = calculate_dew_point(temp_c, rh_percent)
    ah = calculate_absolute_humidity(temp_c, e)

    # 1. RH Bounds
    rh_negative = rh_percent < thresholds.min_rh_percent
    rh_supersaturated = rh_percent > (thresholds.max_rh_percent + thresholds.rh_tolerance_percent)
    rh_out_of_bounds = rh_negative | rh_supersaturated

    # 2. Dew point consistency: Dew point cannot exceed dry bulb temperature
    dew_point_exceeds_temp = td > (temp_c + thresholds.dew_point_tolerance_c)

    # 3. Vapor pressure cannot exceed ambient total pressure
    vapor_exceeds_pressure = e >= pressure_hpa

    # 4. Absolute humidity plausible range
    ah_unphysical = (ah < 0.0) | (ah > thresholds.max_absolute_humidity_gm3)

    # 5. Temperature climatological range
    temp_out_of_bounds = (temp_c < thresholds.min_temp_c) | (temp_c > thresholds.max_temp_c)

    # 6. Pressure climatological range
    pressure_out_of_bounds = (pressure_hpa < thresholds.min_pressure_hpa) | (pressure_hpa > thresholds.max_pressure_hpa)

    # Combined T-P-RH consistency violation
    t_p_rh_inconsistent = (
        rh_out_of_bounds |
        dew_point_exceeds_temp |
        vapor_exceeds_pressure |
        ah_unphysical |
        temp_out_of_bounds |
        pressure_out_of_bounds
    )

    return {
        "rh_out_of_bounds": rh_out_of_bounds.fillna(False),
        "dew_point_exceeds_temp": dew_point_exceeds_temp.fillna(False),
        "vapor_exceeds_pressure": vapor_exceeds_pressure.fillna(False),
        "ah_unphysical": ah_unphysical.fillna(False),
        "temp_out_of_bounds": temp_out_of_bounds.fillna(False),
        "pressure_out_of_bounds": pressure_out_of_bounds.fillna(False),
        "t_p_rh_inconsistent": t_p_rh_inconsistent.fillna(False),
        # Derived values for inspection
        "_dew_point_c": td,
        "_actual_vapor_pressure_hpa": e,
        "_saturation_vapor_pressure_hpa": es,
        "_absolute_humidity_gm3": ah,
    }


def validate_wind_speed_bounds(
    wind_speed_mps: pd.Series,
    thresholds: Optional[PhysicsThresholds] = None
) -> Dict[str, pd.Series]:
    """
    Validate wind speed against physical limits and typical AWS bounds.

    Performs:
    1. Negative wind speed check (WS < 0 m/s is physically impossible)
    2. Typical range check (0 <= WS <= 50 m/s typical meteorological range)
    3. Extreme physical limit check (WS > 90 m/s impossible or catastrophic sensor failure)

    Returns:
        Dictionary of boolean Series where True indicates an anomaly.
    """
    if thresholds is None:
        thresholds = PhysicsThresholds()

    wind_negative = wind_speed_mps < thresholds.min_wind_speed_mps
    wind_exceeds_typical = wind_speed_mps > thresholds.max_wind_speed_typical_mps
    wind_exceeds_extreme = wind_speed_mps > thresholds.max_wind_speed_extreme_mps
    
    wind_invalid = wind_negative | wind_exceeds_extreme

    return {
        "wind_negative": wind_negative.fillna(False),
        "wind_exceeds_typical": wind_exceeds_typical.fillna(False),
        "wind_exceeds_extreme": wind_exceeds_extreme.fillna(False),
        "wind_invalid": wind_invalid.fillna(False),
    }


class PhysicsQualityChecker:
    """High-level physics quality control checker for comprehensive AWS multi-sensor validation."""

    def __init__(self, thresholds: Optional[PhysicsThresholds] = None):
        self.thresholds = thresholds or PhysicsThresholds()

    def check_all(
        self,
        df: pd.DataFrame,
        temp_col: str = "temperature",
        pressure_col: str = "pressure",
        rh_col: str = "humidity",
        wind_col: str = "wind_speed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all physical consistency checks across standard AWS variables.

        Args:
            df: DataFrame containing sensor columns.
            temp_col: Column name for Temperature (°C).
            pressure_col: Column name for Pressure (hPa).
            rh_col: Column name for Relative Humidity (%).
            wind_col: Column name for Wind Speed (m/s).

        Returns:
            Tuple of (flags_df, derived_thermodynamics_df).
        """
        t_p_rh_results = validate_t_p_rh_physics(
            temp_c=df[temp_col],
            pressure_hpa=df[pressure_col],
            rh_percent=df[rh_col],
            thresholds=self.thresholds
        )

        wind_results = validate_wind_speed_bounds(
            wind_speed_mps=df[wind_col],
            thresholds=self.thresholds
        )

        # Separate flags and derived quantities
        flags = {
            "flag_temp_out_of_bounds": t_p_rh_results["temp_out_of_bounds"],
            "flag_pressure_out_of_bounds": t_p_rh_results["pressure_out_of_bounds"],
            "flag_rh_out_of_bounds": t_p_rh_results["rh_out_of_bounds"],
            "flag_dew_point_exceeds_temp": t_p_rh_results["dew_point_exceeds_temp"],
            "flag_vapor_exceeds_pressure": t_p_rh_results["vapor_exceeds_pressure"],
            "flag_ah_unphysical": t_p_rh_results["ah_unphysical"],
            "flag_t_p_rh_physics_violation": t_p_rh_results["t_p_rh_inconsistent"],
            "flag_wind_negative": wind_results["wind_negative"],
            "flag_wind_exceeds_typical": wind_results["wind_exceeds_typical"],
            "flag_wind_exceeds_extreme": wind_results["wind_exceeds_extreme"],
            "flag_wind_physics_violation": wind_results["wind_invalid"],
        }
        flags_df = pd.DataFrame(flags, index=df.index)

        derived = {
            "dew_point_c": t_p_rh_results["_dew_point_c"],
            "actual_vapor_pressure_hpa": t_p_rh_results["_actual_vapor_pressure_hpa"],
            "saturation_vapor_pressure_hpa": t_p_rh_results["_saturation_vapor_pressure_hpa"],
            "absolute_humidity_gm3": t_p_rh_results["_absolute_humidity_gm3"],
        }
        derived_df = pd.DataFrame(derived, index=df.index)

        return flags_df, derived_df
