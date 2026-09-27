"""
Physics-Informed Rule Engine with Hierarchical Severity Mapping.

Rules:
1. HIGH Severity (CRITICAL):
   - Violates fundamental thermodynamic relationships:
     * Dew point exceeds dry-bulb temperature (Td > T)
     * Water vapor pressure exceeds atmospheric ambient pressure (e >= P)
     * Physical relative humidity violation (RH < 0% or RH > 105% unphysical supersaturation)
2. MEDIUM Severity (WARNING):
   - Exceeds instrument / sensor operational specifications:
     * Negative wind speed (WS < 0 m/s)
     * Extreme wind speed exceeding 50 m/s specification bound
     * Climatological / hardware temperature limits (-45°C to 55°C)
     * Climatological / hardware pressure limits (850 hPa to 1080 hPa)
3. LOW Severity (INFO):
   - Statistical deviations & operational glitches:
     * Transient rate-of-change step spike
     * Frozen / stuck sensor (zero variance flatline)
     * Statistical distribution excursion (Z-Score / IQR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from ..physics_checks import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_dew_point,
)


@dataclass
class RuleViolation:
    """Individual rule violation record."""
    rule_id: str
    rule_name: str
    severity: str        # 'HIGH', 'MEDIUM', 'LOW'
    alert_level: str     # 'CRITICAL', 'WARNING', 'INFO'
    affected_sensors: List[str]
    confidence: float    # 0.0 - 1.0
    description: str
    recommended_action: str


class PhysicsRuleEngine:
    """Evaluates physical and thermodynamic constraints on AWS multi-sensor readings."""

    def __init__(
        self,
        min_temp_spec: float = -45.0,
        max_temp_spec: float = 55.0,
        min_pressure_spec: float = 850.0,
        max_pressure_spec: float = 1080.0,
        max_wind_spec: float = 50.0,
        dew_point_tolerance_c: float = 0.5,
    ):
        self.min_temp_spec = min_temp_spec
        self.max_temp_spec = max_temp_spec
        self.min_pressure_spec = min_pressure_spec
        self.max_pressure_spec = max_pressure_spec
        self.max_wind_spec = max_wind_spec
        self.dew_point_tolerance_c = dew_point_tolerance_c

    def evaluate_row(
        self,
        temp_c: float,
        pressure_hpa: float,
        rh_percent: float,
        wind_speed_mps: float,
        stat_flags: Optional[Dict[str, bool]] = None
    ) -> List[RuleViolation]:
        """
        Evaluate a single multi-sensor observation vector across all physics rules.
        """
        violations: List[RuleViolation] = []

        # Handle NaNs
        has_t = not np.isnan(temp_c)
        has_p = not np.isnan(pressure_hpa)
        has_rh = not np.isnan(rh_percent)
        has_ws = not np.isnan(wind_speed_mps)

        # -------------------------------------------------------------
        # 1. HIGH SEVERITY (CRITICAL): Thermodynamic Violations
        # -------------------------------------------------------------
        if has_t and has_rh:
            # Saturation & Actual vapor pressure
            e = calculate_actual_vapor_pressure(temp_c, rh_percent)
            td = calculate_dew_point(temp_c, rh_percent)

            # Rule 1.1: Dew Point cannot exceed Dry Bulb Temperature
            if td > (temp_c + self.dew_point_tolerance_c):
                violations.append(RuleViolation(
                    rule_id="THERMO_TD_EXCEEDS_T",
                    rule_name="Dew Point Exceeds Temperature",
                    severity="HIGH",
                    alert_level="CRITICAL",
                    affected_sensors=["temperature", "humidity"],
                    confidence=0.98,
                    description=f"Thermodynamic violation: Calculated Dew Point ({td:.1f}°C) exceeds Dry Bulb Temp ({temp_c:.1f}°C)",
                    recommended_action="Emergency calibration required: Inspect hygrometer sensing element and aspirator fan",
                ))

            # Rule 1.2: Vapor pressure cannot exceed ambient pressure
            if has_p and e >= pressure_hpa:
                violations.append(RuleViolation(
                    rule_id="THERMO_VAPOR_EXCEEDS_PRESSURE",
                    rule_name="Vapor Pressure Exceeds Ambient Pressure",
                    severity="HIGH",
                    alert_level="CRITICAL",
                    affected_sensors=["pressure", "humidity"],
                    confidence=0.99,
                    description=f"Thermodynamic violation: Water vapor pressure ({e:.1f} hPa) exceeds ambient barometric pressure ({pressure_hpa:.1f} hPa)",
                    recommended_action="Isolate barometer and RH sensor channels; check for sensor chamber flooding",
                ))

            # Rule 1.3: Unphysical Relative Humidity
            if rh_percent < 0.0 or rh_percent > 105.0:
                violations.append(RuleViolation(
                    rule_id="THERMO_RH_OUT_OF_BOUNDS",
                    rule_name="Unphysical Relative Humidity",
                    severity="HIGH",
                    alert_level="CRITICAL",
                    affected_sensors=["humidity"],
                    confidence=0.97,
                    description=f"Relative Humidity ({rh_percent:.1f}%) is outside physical limits [0%, 105%]",
                    recommended_action="Recalibrate or replace capacitive humidity sensor element; inspect for condensation or saline coating",
                ))

        # -------------------------------------------------------------
        # 2. MEDIUM SEVERITY (WARNING): Sensor Specification Breaches
        # -------------------------------------------------------------
        if has_ws:
            if wind_speed_mps < 0.0:
                violations.append(RuleViolation(
                    rule_id="SPEC_WIND_NEGATIVE",
                    rule_name="Negative Wind Speed",
                    severity="MEDIUM",
                    alert_level="WARNING",
                    affected_sensors=["wind_speed"],
                    confidence=0.95,
                    description=f"Negative wind speed measured ({wind_speed_mps:.2f} m/s)",
                    recommended_action="Inspect anemometer wiring, zero-offset calibration, and ADC voltage reference",
                ))
            elif wind_speed_mps > self.max_wind_spec:
                violations.append(RuleViolation(
                    rule_id="SPEC_WIND_EXCEEDS_RANGE",
                    rule_name="Wind Speed Exceeds Typical Specification",
                    severity="MEDIUM",
                    alert_level="WARNING",
                    affected_sensors=["wind_speed"],
                    confidence=0.88,
                    description=f"Wind speed ({wind_speed_mps:.1f} m/s) exceeds typical 50 m/s sensor specification",
                    recommended_action="Verify severe weather event / gust recording; inspect anemometer bearing for mechanical damage",
                ))

        if has_t:
            if temp_c < self.min_temp_spec or temp_c > self.max_temp_spec:
                violations.append(RuleViolation(
                    rule_id="SPEC_TEMP_EXCEEDS_RANGE",
                    rule_name="Temperature Outside Operating Limits",
                    severity="MEDIUM",
                    alert_level="WARNING",
                    affected_sensors=["temperature"],
                    confidence=0.92,
                    description=f"Temperature ({temp_c:.1f}°C) exceeds operating specifications [{self.min_temp_spec}°C, {self.max_temp_spec}°C]",
                    recommended_action="Inspect radiation shield ventilation and RTD/thermistor wiring integrity",
                ))

        if has_p:
            if pressure_hpa < self.min_pressure_spec or pressure_hpa > self.max_pressure_spec:
                violations.append(RuleViolation(
                    rule_id="SPEC_PRESSURE_EXCEEDS_RANGE",
                    rule_name="Pressure Outside Operating Limits",
                    severity="MEDIUM",
                    alert_level="WARNING",
                    affected_sensors=["pressure"],
                    confidence=0.92,
                    description=f"Pressure ({pressure_hpa:.1f} hPa) exceeds station operating specifications [{self.min_pressure_spec} hPa, {self.max_pressure_spec} hPa]",
                    recommended_action="Inspect barometric pressure port for insect blockage or static port leakage",
                ))

        # -------------------------------------------------------------
        # 3. LOW SEVERITY (INFO): Statistical & Transient Outliers
        # -------------------------------------------------------------
        if stat_flags:
            for ch, is_flagged in stat_flags.items():
                if is_flagged:
                    if "spike" in ch.lower():
                        violations.append(RuleViolation(
                            rule_id=f"STAT_SPIKE_{ch.upper()}",
                            rule_name="Transient Rate-of-Change Spike",
                            severity="LOW",
                            alert_level="INFO",
                            affected_sensors=[ch.replace("flag_", "").replace("_spike", "")],
                            confidence=0.75,
                            description=f"Rapid rate-of-change spike detected in {ch}",
                            recommended_action="Apply median smoothing filter; monitor channel for recurring transient spikes",
                        ))
                    elif "stuck" in ch.lower():
                        violations.append(RuleViolation(
                            rule_id=f"STAT_FLATLINE_{ch.upper()}",
                            rule_name="Sensor Output Flatline",
                            severity="LOW",
                            alert_level="WARNING",
                            affected_sensors=[ch.replace("flag_", "").replace("_stuck", "")],
                            confidence=0.82,
                            description=f"Frozen sensor: reading remained constant with zero variance",
                            recommended_action="Check sensor power line and communication bus (SDI-12 / Modbus / RS-485)",
                        ))
                    elif "outlier" in ch.lower():
                        violations.append(RuleViolation(
                            rule_id=f"STAT_OUTLIER_{ch.upper()}",
                            rule_name="Statistical Distribution Outlier",
                            severity="LOW",
                            alert_level="INFO",
                            affected_sensors=[ch.replace("flag_", "").replace("_outlier", "")],
                            confidence=0.68,
                            description=f"Reading is an excursion beyond normal statistical distribution",
                            recommended_action="Flag for automated quality control review; cross-check with nearby weather stations",
                        ))

        return violations

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        stat_flags_df: Optional[pd.DataFrame] = None
    ) -> List[List[RuleViolation]]:
        """
        Evaluate physics rules across all rows of a DataFrame.
        """
        all_violations = []
        n_rows = len(df)

        for i in range(n_rows):
            t_val = df["temperature"].iloc[i] if "temperature" in df.columns else np.nan
            p_val = df["pressure"].iloc[i] if "pressure" in df.columns else np.nan
            rh_val = df["humidity"].iloc[i] if "humidity" in df.columns else np.nan
            ws_val = df["wind_speed"].iloc[i] if "wind_speed" in df.columns else np.nan

            stat_flags = None
            if stat_flags_df is not None and i < len(stat_flags_df):
                stat_flags = stat_flags_df.iloc[i].to_dict()

            v_list = self.evaluate_row(
                temp_c=t_val,
                pressure_hpa=p_val,
                rh_percent=rh_val,
                wind_speed_mps=ws_val,
                stat_flags=stat_flags,
            )
            all_violations.append(v_list)

        return all_violations
