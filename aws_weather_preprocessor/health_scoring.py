"""
Sensor Health Scoring and Predictive Maintenance System for AWS Instruments.

Implements:
1. Multi-metric sensor tracking:
   - Data completeness rate (% received vs expected)
   - Bias drift rate (bias change over time from Kalman filter or reference)
   - Noise level (variance/std of measurement residuals)
   - Calibration status (elapsed time vs calibration interval)
2. Health Score model (0 - 100%) with hierarchical degradation grades
3. Predictive Maintenance forecasting (Remaining Useful Life / RUL in days)
4. Sensor degradation flagging in anomaly assessment
5. Health Dashboard generator & prioritized maintenance alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta


@dataclass
class SensorProfile:
    """Hardware specification and calibration parameters for a sensor."""
    sensor_name: str
    instrument_model: str
    unit: str
    last_calibration_date: str            # ISO date string e.g. "2026-01-01"
    calibration_interval_days: int = 365   # Standard 1-year calibration interval
    max_allowable_bias: float = 0.8        # Max allowed bias before out-of-spec
    max_allowable_noise_std: float = 0.5   # Max allowed residual noise std
    min_completeness_threshold: float = 95.0
    degradation_score_threshold: float = 60.0  # Below this score, sensor is flagged degraded


@dataclass
class SensorHealthMetrics:
    """Computed health and performance metrics for a single instrument."""
    sensor_name: str
    instrument_model: str
    total_expected: int
    total_received: int
    completeness_pct: float
    current_bias: float
    drift_rate_per_day: float
    noise_std: float
    noise_variance: float
    days_since_calibration: float
    calibration_status: str               # 'OPTIMAL', 'DUE_SOON', 'EXPIRED', 'OUT_OF_SPEC'
    anomaly_incidents: int
    # Sub-scores (0 - 100)
    completeness_subscore: float
    drift_subscore: float
    noise_subscore: float
    calibration_subscore: float
    # Overall Score & Status
    health_score: float                   # 0.0 - 100.0%
    health_grade: str                     # 'EXCELLENT', 'GOOD', 'FAIR', 'DEGRADED', 'CRITICAL'
    is_degraded: bool
    # Predictive Maintenance
    predicted_rul_days: float             # Remaining Useful Life in days
    maintenance_status: str               # 'NORMAL', 'CALIBRATION_RECOMMENDED', 'MAINTENANCE_DUE', 'URGENT_ACTION_REQUIRED'
    recommended_action: str


@dataclass
class MaintenanceAlert:
    """Actionable maintenance alert record."""
    timestamp: Any
    sensor_name: str
    instrument_model: str
    urgency: str                          # 'URGENT', 'HIGH', 'MEDIUM', 'LOW'
    trigger_reason: str
    current_health_score: float
    predicted_rul_days: float
    action_instructions: str


class SensorHealthScoringSystem:
    """
    Automated Sensor Health Tracking, Scoring, and Predictive Maintenance Engine.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, SensorProfile]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        # Default sensor hardware profiles
        self.profiles = profiles or {
            "temperature": SensorProfile(
                sensor_name="temperature",
                instrument_model="Pt100 4-Wire RTD Class 1/10 DIN",
                unit="°C",
                last_calibration_date="2026-03-01",
                calibration_interval_days=365,
                max_allowable_bias=0.6,
                max_allowable_noise_std=0.35,
            ),
            "pressure": SensorProfile(
                sensor_name="pressure",
                instrument_model="Vaisala PTB330 Digital Barometer",
                unit="hPa",
                last_calibration_date="2026-01-15",
                calibration_interval_days=365,
                max_allowable_bias=1.2,
                max_allowable_noise_std=0.25,
            ),
            "humidity": SensorProfile(
                sensor_name="humidity",
                instrument_model="Rotronic HygroMet4 Thin-Film Capacitive",
                unit="%",
                last_calibration_date="2026-04-10",
                calibration_interval_days=180,
                max_allowable_bias=3.5,
                max_allowable_noise_std=1.8,
            ),
            "wind_speed": SensorProfile(
                sensor_name="wind_speed",
                instrument_model="Gill WindObserver II Ultrasonic Anemometer",
                unit="m/s",
                last_calibration_date="2025-11-01",
                calibration_interval_days=365,
                max_allowable_bias=0.8,
                max_allowable_noise_std=0.6,
            ),
        }

        # Multi-criteria scoring weights (sum to 1.0)
        self.weights = weights or {
            "completeness": 0.25,
            "drift": 0.35,
            "noise": 0.25,
            "calibration": 0.15,
        }

    def evaluate_sensor_health(
        self,
        sensor_name: str,
        raw_series: pd.Series,
        bias_series: Optional[pd.Series] = None,
        residual_series: Optional[pd.Series] = None,
        anomaly_flags_series: Optional[pd.Series] = None,
        current_date: Optional[str] = None,
        dt_minutes: float = 5.0,
    ) -> SensorHealthMetrics:
        """
        Compute full health metrics, 0-100% score, and RUL for an individual sensor.
        """
        profile = self.profiles.get(
            sensor_name,
            SensorProfile(
                sensor_name=sensor_name,
                instrument_model="Standard AWS Sensor",
                unit="",
                last_calibration_date="2026-01-01",
            )
        )

        total_expected = len(raw_series)
        # Non-null, non-sentinel valid readings
        valid_mask = ~raw_series.isna() & (raw_series > -900) & (raw_series < 9000)
        total_received = int(valid_mask.sum())
        completeness_pct = (total_received / total_expected * 100.0) if total_expected > 0 else 0.0

        # 1. Bias & Drift Tracking
        if bias_series is not None and not bias_series.dropna().empty:
            b_clean = bias_series.dropna()
            current_bias = float(b_clean.iloc[-1])
            
            # Compute drift rate per day (slope of bias over time)
            num_points = len(b_clean)
            if num_points > 10:
                time_days = np.linspace(0, (num_points * dt_minutes) / 1440.0, num_points)
                drift_slope, _ = np.polyfit(time_days, b_clean.values, 1)
                drift_rate_per_day = float(drift_slope)
            else:
                drift_rate_per_day = 0.0
        else:
            current_bias = 0.0
            drift_rate_per_day = 0.0

        # 2. Residual Noise Level
        if residual_series is not None and not residual_series.dropna().empty:
            res_clean = residual_series.dropna()
            noise_std = float(res_clean.std())
            noise_var = float(res_clean.var())
        else:
            # Fallback high-pass filter: differencing
            diff_clean = raw_series[valid_mask].diff().dropna()
            noise_std = float(diff_clean.std() / np.sqrt(2)) if not diff_clean.empty else 0.0
            noise_var = float(noise_std ** 2)

        # 3. Calibration Status & Elapsed Days
        curr_dt = pd.to_datetime(current_date or "2026-09-24")
        cal_dt = pd.to_datetime(profile.last_calibration_date)
        days_since_cal = max(0.0, (curr_dt - cal_dt).total_seconds() / 86400.0)

        cal_limit = profile.calibration_interval_days
        if days_since_cal > cal_limit:
            cal_status = "EXPIRED"
        elif days_since_cal > (cal_limit * 0.85):
            cal_status = "DUE_SOON"
        elif abs(current_bias) > profile.max_allowable_bias:
            cal_status = "OUT_OF_SPEC"
        else:
            cal_status = "OPTIMAL"

        # 4. Anomaly Incidents
        anom_count = int(anomaly_flags_series.sum()) if anomaly_flags_series is not None else 0

        # 5. Compute Sub-scores (0 - 100)
        # Completeness sub-score: steep drop below min_completeness_threshold (e.g. 95%)
        if completeness_pct >= profile.min_completeness_threshold:
            s_comp = 90.0 + 10.0 * ((completeness_pct - profile.min_completeness_threshold) / (100.0 - profile.min_completeness_threshold + 1e-6))
        else:
            s_comp = max(0.0, (completeness_pct / profile.min_completeness_threshold) * 90.0)

        # Drift sub-score: penalty based on fraction of max allowable bias reached
        bias_ratio = min(2.0, abs(current_bias) / (profile.max_allowable_bias + 1e-6))
        s_drift = max(0.0, 100.0 * (1.0 - (bias_ratio / 1.5)))

        # Noise sub-score: penalty based on residual noise standard deviation
        noise_ratio = min(2.0, noise_std / (profile.max_allowable_noise_std + 1e-6))
        s_noise = max(0.0, 100.0 * (1.0 - (noise_ratio / 1.6)))

        # Calibration sub-score: penalty as days approach/exceed calibration interval
        cal_ratio = days_since_cal / (cal_limit + 1e-6)
        if cal_ratio <= 1.0:
            s_cal = 100.0 * (1.0 - 0.4 * cal_ratio)  # 100% -> 60% over valid year
        else:
            s_cal = max(0.0, 60.0 - 40.0 * (cal_ratio - 1.0))

        # 6. Composite Health Score Calculation
        w = self.weights
        health_raw = (
            w["completeness"] * s_comp +
            w["drift"] * s_drift +
            w["noise"] * s_noise +
            w["calibration"] * s_cal
        )

        # Penalty for repeated anomaly incidents (e.g. -1 point per physical violation)
        anom_penalty = min(20.0, anom_count * 1.5)
        health_score = float(np.clip(health_raw - anom_penalty, 0.0, 100.0))

        # Health Grade Mapping
        if health_score >= 90.0:
            health_grade = "EXCELLENT"
        elif health_score >= 75.0:
            health_grade = "GOOD"
        elif health_score >= 60.0:
            health_grade = "FAIR"
        elif health_score >= 40.0:
            health_grade = "DEGRADED"
        else:
            health_grade = "CRITICAL"

        is_degraded = health_score < profile.degradation_score_threshold

        # 7. Predictive Maintenance Forecasting (Remaining Useful Life / RUL in Days)
        rul_candidates = []

        # A. Drift-based RUL projection: time until bias hits max_allowable_bias
        remaining_bias_margin = max(0.0, profile.max_allowable_bias - abs(current_bias))
        if abs(drift_rate_per_day) > 1e-4:
            rul_drift = remaining_bias_margin / abs(drift_rate_per_day)
            rul_candidates.append(rul_drift)
        else:
            rul_candidates.append(365.0)

        # B. Calibration expiry RUL
        rul_cal = max(0.0, cal_limit - days_since_cal)
        rul_candidates.append(rul_cal)

        # C. Overall RUL (limiting factor)
        predicted_rul_days = float(min(rul_candidates))

        # Maintenance Status & Actions
        if health_score < 40.0 or abs(current_bias) >= profile.max_allowable_bias or predicted_rul_days <= 7.0:
            maint_status = "URGENT_ACTION_REQUIRED"
            rec_action = f"URGENT: Immediate field inspection & sensor replacement / full recalibration required for {sensor_name}."
        elif is_degraded or predicted_rul_days <= 30.0 or cal_status == "EXPIRED":
            maint_status = "MAINTENANCE_DUE"
            rec_action = f"Schedule maintenance within {int(predicted_rul_days)} days: Recalibrate zero/span and clean sensor aperture."
        elif cal_status == "DUE_SOON" or health_score < 75.0:
            maint_status = "CALIBRATION_RECOMMENDED"
            rec_action = f"Calibration inspection recommended: Monitor drift trend ({drift_rate_per_day:+.3f} {profile.unit}/day)."
        else:
            maint_status = "NORMAL"
            rec_action = "Sensor operating within optimal specifications. Continue routine monitoring."

        return SensorHealthMetrics(
            sensor_name=sensor_name,
            instrument_model=profile.instrument_model,
            total_expected=total_expected,
            total_received=total_received,
            completeness_pct=round(completeness_pct, 2),
            current_bias=round(current_bias, 3),
            drift_rate_per_day=round(drift_rate_per_day, 4),
            noise_std=round(noise_std, 3),
            noise_variance=round(noise_var, 4),
            days_since_calibration=round(days_since_cal, 1),
            calibration_status=cal_status,
            anomaly_incidents=anom_count,
            completeness_subscore=round(s_comp, 1),
            drift_subscore=round(s_drift, 1),
            noise_subscore=round(s_noise, 1),
            calibration_subscore=round(s_cal, 1),
            health_score=round(health_score, 1),
            health_grade=health_grade,
            is_degraded=is_degraded,
            predicted_rul_days=round(predicted_rul_days, 1),
            maintenance_status=maint_status,
            recommended_action=rec_action,
        )

    def evaluate_all(
        self,
        raw_df: pd.DataFrame,
        bias_df: Optional[pd.DataFrame] = None,
        residuals_df: Optional[pd.DataFrame] = None,
        flags_df: Optional[pd.DataFrame] = None,
        current_date: Optional[str] = None,
        dt_minutes: float = 5.0,
    ) -> Dict[str, SensorHealthMetrics]:
        """
        Evaluate health across all weather station sensors.
        """
        metrics_dict = {}
        for ch in self.profiles.keys():
            if ch in raw_df.columns:
                raw_s = raw_df[ch]
                
                # Extract matching bias series
                bias_s = None
                if bias_df is not None:
                    if f"{ch}_bias" in bias_df.columns:
                        bias_s = bias_df[f"{ch}_bias"]
                    elif f"{ch}_true_bias" in bias_df.columns:
                        bias_s = bias_df[f"{ch}_true_bias"]

                # Extract matching residuals
                res_s = None
                if residuals_df is not None:
                    if f"{ch}_post_fit_residual" in residuals_df.columns:
                        res_s = residuals_df[f"{ch}_post_fit_residual"]
                    elif f"{ch}_innovation" in residuals_df.columns:
                        res_s = residuals_df[f"{ch}_innovation"]

                # Extract anomaly flags
                flag_s = None
                if flags_df is not None:
                    flag_cols = [c for c in flags_df.columns if ch in c and "flag_" in c]
                    if flag_cols:
                        flag_s = flags_df[flag_cols].any(axis=1)

                m = self.evaluate_sensor_health(
                    sensor_name=ch,
                    raw_series=raw_s,
                    bias_series=bias_s,
                    residual_series=res_s,
                    anomaly_flags_series=flag_s,
                    current_date=current_date,
                    dt_minutes=dt_minutes,
                )
                metrics_dict[ch] = m

        return metrics_dict

    def generate_maintenance_alerts(
        self,
        health_metrics: Dict[str, SensorHealthMetrics],
        timestamp: Optional[Any] = None
    ) -> List[MaintenanceAlert]:
        """
        Compile prioritized maintenance alerts for degraded sensors.
        """
        alerts: List[MaintenanceAlert] = []
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for s_name, m in health_metrics.items():
            if m.maintenance_status == "URGENT_ACTION_REQUIRED":
                alerts.append(MaintenanceAlert(
                    timestamp=ts,
                    sensor_name=s_name,
                    instrument_model=m.instrument_model,
                    urgency="URGENT",
                    trigger_reason=f"Health Score Critical ({m.health_score}%) | Bias ({m.current_bias:+.2f}) exceeds limits | RUL: {m.predicted_rul_days} days",
                    current_health_score=m.health_score,
                    predicted_rul_days=m.predicted_rul_days,
                    action_instructions=m.recommended_action,
                ))
            elif m.maintenance_status == "MAINTENANCE_DUE":
                alerts.append(MaintenanceAlert(
                    timestamp=ts,
                    sensor_name=s_name,
                    instrument_model=m.instrument_model,
                    urgency="HIGH",
                    trigger_reason=f"Sensor Degraded ({m.health_score}%) | RUL: {m.predicted_rul_days} days | Drift: {m.drift_rate_per_day:+.3f}/day",
                    current_health_score=m.health_score,
                    predicted_rul_days=m.predicted_rul_days,
                    action_instructions=m.recommended_action,
                ))
            elif m.maintenance_status == "CALIBRATION_RECOMMENDED":
                alerts.append(MaintenanceAlert(
                    timestamp=ts,
                    sensor_name=s_name,
                    instrument_model=m.instrument_model,
                    urgency="MEDIUM",
                    trigger_reason=f"Calibration interval expiring ({m.days_since_calibration} days elapsed) | Status: {m.calibration_status}",
                    current_health_score=m.health_score,
                    predicted_rul_days=m.predicted_rul_days,
                    action_instructions=m.recommended_action,
                ))

        # Sort by urgency
        urgency_rank = {"URGENT": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
        alerts.sort(key=lambda a: urgency_rank.get(a.urgency, 5))
        return alerts

    def generate_dashboard_markdown(
        self,
        health_metrics: Dict[str, SensorHealthMetrics],
        station_id: str = "AWS-STATION-01"
    ) -> str:
        """
        Generate rich Markdown Health Dashboard with visual health indicators.
        """
        md = []
        md.append(f"# 🏥 AWS Sensor Health & Maintenance Dashboard ({station_id})")
        md.append("")
        md.append("## 1. Station Sensor Health Overview")
        md.append("")
        md.append("| Sensor | Instrument Model | Health Score | Grade | Completeness | Bias Drift Rate | Noise Level | RUL (Days) | Maintenance Status |")
        md.append("|---|---|---|---|---|---|---|---|---|")

        for s_name, m in health_metrics.items():
            # Create visual progress bar
            filled = int(m.health_score / 10)
            bar = "█" * filled + "░" * (10 - filled)
            
            # Grade badge
            if m.health_grade == "EXCELLENT":
                grade_icon = f"🟢 **{m.health_grade}**"
            elif m.health_grade == "GOOD":
                grade_icon = f"🟢 **{m.health_grade}**"
            elif m.health_grade == "FAIR":
                grade_icon = f"🟡 **{m.health_grade}**"
            elif m.health_grade == "DEGRADED":
                grade_icon = f"🟠 **{m.health_grade}**"
            else:
                grade_icon = f"🔴 **{m.health_grade}**"

            md.append(
                f"| `{s_name.capitalize()}` | {m.instrument_model} | `[{bar}]` **{m.health_score}%** | "
                f"{grade_icon} | {m.completeness_pct}% | `{m.drift_rate_per_day:+.3f}`/day | "
                f"σ={m.noise_std:.2f} | **{m.predicted_rul_days:.0f} d** | `{m.maintenance_status}` |"
            )

        md.append("")
        md.append("## 2. Sensor-by-Sensor Detailed Diagnostics")
        md.append("")
        for s_name, m in health_metrics.items():
            md.append(f"### 📡 {s_name.capitalize()} ({m.instrument_model})")
            md.append(f"- **Overall Health Score**: `{m.health_score}%` ({m.health_grade})")
            md.append(f"- **Degradation Status**: {'⚠️ **DEGRADED SENSOR FLAGGED**' if m.is_degraded else '✅ **NORMAL OPERATION**'}")
            md.append(f"- **Data Completeness**: `{m.completeness_pct}%` ({m.total_received}/{m.total_expected} readings received)")
            md.append(f"- **Current Estimated Bias**: `{m.current_bias:+.3f}` (Drift Rate: `{m.drift_rate_per_day:+.4f}`/day)")
            md.append(f"- **Residual Noise Std**: `{m.noise_std:.3f}` (Variance: `{m.noise_variance:.4f}`)")
            md.append(f"- **Calibration Status**: `{m.calibration_status}` ({m.days_since_calibration:.0f} days since last calibration)")
            md.append(f"- **Predicted Remaining Useful Life (RUL)**: **{m.predicted_rul_days:.1f} days**")
            md.append(f"- **Action Required**: {m.recommended_action}")
            md.append("")

        alerts = self.generate_maintenance_alerts(health_metrics)
        md.append("## 3. Prioritized Maintenance Alerts Queue")
        if alerts:
            md.append("| Priority | Sensor | Trigger Diagnostic | Remaining Life | Technician Action |")
            md.append("|---|---|---|---|---|")
            for a in alerts:
                if a.urgency == "URGENT":
                    u_icon = "🚨 **URGENT**"
                elif a.urgency == "HIGH":
                    u_icon = "⚠️ **HIGH**"
                else:
                    u_icon = "ℹ️ **MEDIUM**"
                md.append(f"| {u_icon} | `{a.sensor_name.capitalize()}` | {a.trigger_reason} | **{a.predicted_rul_days:.0f} days** | {a.action_instructions} |")
        else:
            md.append("✅ *All sensors operating within healthy tolerance. No maintenance alerts active.*")
        md.append("")

        return "\n".join(md)
