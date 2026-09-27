"""
Sensor Auto-Correction, Historical Context Tracking, and Real-Time Alerting Module.

Implements:
1. Automated bias correction with physical bounding
2. Multi-level dynamic confidence interval derivation (95% and 99% CIs)
3. Actionable alerting with historical context (recurrence frequency & pattern memory)
4. Full dataset exports (CSV and JSON) with corrected values, biases, bounds, and flags
5. Real-time alert and health dashboard generator
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from .kalman_tracker import AtmosphericStateKalmanFilter, KalmanOutput
from .health_scoring import SensorHealthScoringSystem, SensorHealthMetrics
from .anomaly_detector import AWSHybridAnomalyDetector, AnomalyDetectionEvent, HybridDetectionResult


@dataclass
class ActionableAlert:
    """Actionable alert with historical recurrence context and technician instructions."""
    alert_id: str
    timestamp: Any
    anomaly_type: str
    severity: str                     # 'CRITICAL', 'WARNING', 'INFO'
    affected_parameters: List[str]
    raw_reading: str
    corrected_estimate: str
    confidence_score: float           # 0.0 - 1.0
    recommended_action: str
    historical_context: str           # Recurrence context: "Detected 3 times in past 48h"
    sensor_health_grade: str          # 'EXCELLENT', 'GOOD', 'FAIR', 'DEGRADED', 'CRITICAL'


class HistoricalAnomalyTracker:
    """Tracks historical anomaly incidents to provide temporal recurrence context."""

    def __init__(self, lookback_window_hours: float = 72.0):
        self.lookback_window_hours = lookback_window_hours
        self.history: List[Dict[str, Any]] = []

    def log_and_query_context(
        self,
        timestamp: Any,
        anomaly_type: str,
        affected_sensors: List[str]
    ) -> str:
        """
        Record current anomaly and return historical recurrence summary.
        """
        curr_dt = pd.to_datetime(timestamp)
        cutoff_dt = curr_dt - timedelta(hours=self.lookback_window_hours)

        # Count previous occurrences matching anomaly type or sensor in lookback window
        prev_matches = [
            h for h in self.history
            if h["dt"] >= cutoff_dt and (
                h["type"] == anomaly_type or
                any(s in h["sensors"] for s in affected_sensors)
            )
        ]

        count = len(prev_matches)
        
        # Format historical context message
        if count == 0:
            context_msg = "First occurrence in observation window (Isolated incident)"
        elif count == 1:
            prev_time_str = prev_matches[-1]["dt"].strftime("%m-%d %H:%M")
            context_msg = f"Similar event previously recorded at {prev_time_str} (2nd occurrence)"
        else:
            context_msg = f"Recurring pattern: Detected {count + 1} times in past {int(self.lookback_window_hours)}h (Investigate persistent fault)"

        # Record into history
        self.history.append({
            "dt": curr_dt,
            "type": anomaly_type,
            "sensors": affected_sensors,
        })

        return context_msg


class SensorAutoCorrector:
    """
    Auto-corrects sensor readings by removing estimated bias and deriving confidence intervals.
    """

    @staticmethod
    def correct_series(
        raw_series: pd.Series,
        bias_series: pd.Series,
        std_series: pd.Series,
        sensor_name: str
    ) -> Dict[str, pd.Series]:
        """
        Apply bias subtraction, calculate 95% and 99% CIs, and enforce physical bounds.
        """
        raw = raw_series.astype(float)
        bias = bias_series.fillna(0.0).astype(float)
        sigma = std_series.fillna(0.1).astype(float)

        # Corrected value = raw - estimated_bias
        corrected = raw - bias

        # Physical clamping
        if sensor_name == "humidity":
            corrected = np.clip(corrected, 0.0, 100.0)
        elif sensor_name == "wind_speed":
            corrected = np.clip(corrected, 0.0, None)
        elif sensor_name == "pressure":
            corrected = np.clip(corrected, 800.0, 1100.0)
        elif sensor_name == "temperature":
            corrected = np.clip(corrected, -50.0, 60.0)

        # 95% Confidence Interval (z = 1.960)
        ci95_lower = corrected - 1.960 * sigma
        ci95_upper = corrected + 1.960 * sigma

        # 99% Confidence Interval (z = 2.576)
        ci99_lower = corrected - 2.576 * sigma
        ci99_upper = corrected + 2.576 * sigma

        # Re-clamp CIs for physically bounded variables
        if sensor_name == "humidity":
            ci95_lower = np.clip(ci95_lower, 0.0, 100.0)
            ci95_upper = np.clip(ci95_upper, 0.0, 100.0)
            ci99_lower = np.clip(ci99_lower, 0.0, 100.0)
            ci99_upper = np.clip(ci99_upper, 0.0, 100.0)
        elif sensor_name == "wind_speed":
            ci95_lower = np.clip(ci95_lower, 0.0, None)
            ci95_upper = np.clip(ci95_upper, 0.0, None)
            ci99_lower = np.clip(ci99_lower, 0.0, None)
            ci99_upper = np.clip(ci99_upper, 0.0, None)

        return {
            f"{sensor_name}_corrected": pd.Series(corrected, index=raw_series.index).round(2),
            f"{sensor_name}_bias_est": pd.Series(bias, index=raw_series.index).round(3),
            f"{sensor_name}_uncertainty_std": pd.Series(sigma, index=raw_series.index).round(3),
            f"{sensor_name}_ci95_lower": pd.Series(ci95_lower, index=raw_series.index).round(2),
            f"{sensor_name}_ci95_upper": pd.Series(ci95_upper, index=raw_series.index).round(2),
            f"{sensor_name}_ci99_lower": pd.Series(ci99_lower, index=raw_series.index).round(2),
            f"{sensor_name}_ci99_upper": pd.Series(ci99_upper, index=raw_series.index).round(2),
        }


@dataclass
class CorrectionAndAlertingResult:
    """Comprehensive output from the Correction & Alerting Module."""
    corrected_dataset_df: pd.DataFrame
    actionable_alerts: List[ActionableAlert]
    alerts_df: pd.DataFrame
    health_metrics: Dict[str, SensorHealthMetrics]
    dashboard_markdown: str
    dashboard_text: str

    def export_csv(self, output_filepath: str) -> None:
        """Export corrected dataset to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        self.corrected_dataset_df.to_csv(output_filepath, index=False)

    def export_json(self, output_filepath: str) -> None:
        """Export corrected dataset to JSON format."""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        self.corrected_dataset_df.to_json(output_filepath, orient="records", date_format="iso", indent=2)

    def export_alerts_csv(self, output_filepath: str) -> None:
        """Export alerts log to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        self.alerts_df.to_csv(output_filepath, index=False)


class AWSCorrectionAndAlertingPipeline:
    """
    Unified end-to-end pipeline orchestrating Auto-Correction, Confidence Intervals,
    Historical Anomaly Tracking, Sensor Health Scoring, and Real-Time Alert Dashboards.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        lookback_history_hours: float = 72.0,
    ):
        self.channels = channels or ["temperature", "pressure", "humidity", "wind_speed"]
        self.kalman_filter = AtmosphericStateKalmanFilter()
        self.health_system = SensorHealthScoringSystem()
        self.anomaly_detector = AWSHybridAnomalyDetector(channels=self.channels)
        self.history_tracker = HistoricalAnomalyTracker(lookback_window_hours=lookback_history_hours)
        self.auto_corrector = SensorAutoCorrector()

    def process(
        self,
        raw_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        timestamp_col: str = "timestamp",
        station_id: str = "AWS-PRIMARY-STATION-01"
    ) -> CorrectionAndAlertingResult:
        """
        Execute full correction, inference, uncertainty bounds, alert tracking, and dashboard pipeline.
        """
        n_rows = len(raw_df)
        timestamps = raw_df[timestamp_col] if timestamp_col in raw_df.columns else raw_df.index

        # 1. State-Space Kalman Filtering for Bias & Uncertainty Tracking
        kf_output = self.kalman_filter.filter_dataframe(
            df=raw_df,
            timestamp_col=timestamp_col,
            reference_df=reference_df
        )

        # 2. Multi-Metric Sensor Health Scoring
        health_metrics = self.health_system.evaluate_all(
            raw_df=raw_df,
            bias_df=kf_output.bias_estimates_df,
            residuals_df=kf_output.residuals_df,
        )

        # 3. Hybrid Anomaly Detection (Physics + LSTM + Isolation Forest)
        hybrid_result = self.anomaly_detector.detect(raw_df, timestamp_col=timestamp_col)

        # 4. Auto-Correction & Confidence Intervals (95% & 99%)
        corrected_dict = {}
        if timestamp_col in raw_df.columns:
            corrected_dict[timestamp_col] = raw_df[timestamp_col]

        for ch in self.channels:
            if ch in raw_df.columns:
                raw_s = raw_df[ch]
                bias_s = kf_output.bias_estimates_df.get(f"{ch}_bias", pd.Series(0.0, index=raw_df.index))
                # Total uncertainty std = sqrt(P_true + P_bias)
                true_std = kf_output.uncertainty_df.get(f"{ch}_true_std", pd.Series(0.2, index=raw_df.index))
                bias_std = kf_output.uncertainty_df.get(f"{ch}_bias_std", pd.Series(0.2, index=raw_df.index))
                total_std = np.sqrt(true_std ** 2 + bias_std ** 2)

                # Raw column preserved
                corrected_dict[f"{ch}_raw"] = raw_s
                
                # Auto-corrected and confidence bands
                ch_corrected_dict = self.auto_corrector.correct_series(
                    raw_series=raw_s,
                    bias_series=bias_s,
                    std_series=total_std,
                    sensor_name=ch
                )
                corrected_dict.update(ch_corrected_dict)

        # Append anomaly detection flags and severity to dataset
        corrected_dict["is_anomaly"] = hybrid_result.summary_df["is_anomaly"]
        corrected_dict["anomaly_severity"] = hybrid_result.summary_df["severity"]
        corrected_dict["anomaly_type"] = hybrid_result.summary_df["anomaly_type"]
        corrected_dict["anomaly_confidence"] = hybrid_result.summary_df["confidence"]

        corrected_df = pd.DataFrame(corrected_dict)

        # 5. Actionable Alert Generation with Historical Context
        actionable_alerts: List[ActionableAlert] = []
        alert_counter = 1

        for i in range(n_rows):
            event: AnomalyDetectionEvent = hybrid_result.events[i]
            if event.anomaly_detected and event.severity in ["CRITICAL", "WARNING", "INFO"]:
                ts_val = timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i]
                
                # Query historical context
                hist_context = self.history_tracker.log_and_query_context(
                    timestamp=ts_val,
                    anomaly_type=event.primary_anomaly_type,
                    affected_sensors=event.affected_sensors
                )

                # Format raw and corrected readings
                raw_parts = []
                corr_parts = []
                primary_sensor = event.affected_sensors[0] if event.affected_sensors else self.channels[0]
                health_grade = health_metrics.get(primary_sensor, None)
                grade_str = health_grade.health_grade if health_grade else "UNKNOWN"

                for s in event.affected_sensors:
                    if s in raw_df.columns:
                        r_val = raw_df[s].iloc[i]
                        c_val = corrected_df[f"{s}_corrected"].iloc[i] if f"{s}_corrected" in corrected_df else r_val
                        raw_parts.append(f"{s}={r_val:.2f}")
                        corr_parts.append(f"{s}={c_val:.2f}")

                raw_reading_str = ", ".join(raw_parts) if raw_parts else "N/A"
                corr_reading_str = ", ".join(corr_parts) if corr_parts else "N/A"

                alert = ActionableAlert(
                    alert_id=f"ALT-{alert_counter:04d}",
                    timestamp=ts_val,
                    anomaly_type=event.primary_anomaly_type,
                    severity=event.severity,
                    affected_parameters=event.affected_sensors,
                    raw_reading=raw_reading_str,
                    corrected_estimate=corr_reading_str,
                    confidence_score=round(event.confidence, 3),
                    recommended_action=event.recommended_action,
                    historical_context=hist_context,
                    sensor_health_grade=grade_str,
                )
                actionable_alerts.append(alert)
                alert_counter += 1

        # Alerts DataFrame
        if actionable_alerts:
            alerts_df = pd.DataFrame([asdict(a) for a in actionable_alerts])
        else:
            alerts_df = pd.DataFrame(columns=[
                "alert_id", "timestamp", "anomaly_type", "severity", "affected_parameters",
                "raw_reading", "corrected_estimate", "confidence_score", "recommended_action",
                "historical_context", "sensor_health_grade"
            ])

        # 6. Real-Time Alert Dashboard Generator
        dashboard_md = self._build_dashboard_markdown(
            station_id=station_id,
            total_records=n_rows,
            alerts=actionable_alerts,
            health_metrics=health_metrics,
            latest_corrected_row=corrected_df.iloc[-1] if n_rows > 0 else None
        )

        dashboard_text = self._build_dashboard_text(
            station_id=station_id,
            alerts=actionable_alerts,
            health_metrics=health_metrics,
        )

        return CorrectionAndAlertingResult(
            corrected_dataset_df=corrected_df,
            actionable_alerts=actionable_alerts,
            alerts_df=alerts_df,
            health_metrics=health_metrics,
            dashboard_markdown=dashboard_md,
            dashboard_text=dashboard_text,
        )

    def _build_dashboard_markdown(
        self,
        station_id: str,
        total_records: int,
        alerts: List[ActionableAlert],
        health_metrics: Dict[str, SensorHealthMetrics],
        latest_corrected_row: Optional[pd.Series] = None
    ) -> str:
        """Build rich Markdown real-time alert dashboard."""
        md = []
        md.append(f"# 🛰️ Real-Time AWS Sensor Alert & Auto-Correction Dashboard")
        md.append(f"**Station ID**: `{station_id}` | **Status**: `ACTIVE MONITORING` | **Records Processed**: `{total_records:,}`")
        md.append("")

        # 1. Live Weather State with Auto-Corrected Values & Confidence Intervals
        md.append("## 1. Live Meteorological Telemetry (Auto-Corrected)")
        if latest_corrected_row is not None:
            md.append("| Sensor Parameter | Raw Sensor Value | Estimated Bias | Auto-Corrected Value | 95% Confidence Interval | 99% Confidence Interval | Sensor Health |")
            md.append("|---|---|---|---|---|---|---|")
            for ch in self.channels:
                if f"{ch}_raw" in latest_corrected_row:
                    raw_val = latest_corrected_row[f"{ch}_raw"]
                    bias_val = latest_corrected_row[f"{ch}_bias_est"]
                    corr_val = latest_corrected_row[f"{ch}_corrected"]
                    ci95 = f"[{latest_corrected_row[f'{ch}_ci95_lower']:.2f}, {latest_corrected_row[f'{ch}_ci95_upper']:.2f}]"
                    ci99 = f"[{latest_corrected_row[f'{ch}_ci99_lower']:.2f}, {latest_corrected_row[f'{ch}_ci99_upper']:.2f}]"
                    
                    hm = health_metrics.get(ch)
                    h_grade = f"{hm.health_score}% ({hm.health_grade})" if hm else "N/A"
                    unit = "°C" if ch == "temperature" else "hPa" if ch == "pressure" else "%" if ch == "humidity" else "m/s"

                    md.append(f"| **{ch.capitalize()}** | `{raw_val:.2f} {unit}` | `{bias_val:+.3f} {unit}` | **`{corr_val:.2f} {unit}`** | `{ci95}` | `{ci99}` | `{h_grade}` |")
            md.append("")

        # 2. Active Actionable Alerts Queue
        md.append("## 2. Active Actionable Alerts Queue")
        crit_count = sum(1 for a in alerts if a.severity == "CRITICAL")
        warn_count = sum(1 for a in alerts if a.severity == "WARNING")
        info_count = sum(1 for a in alerts if a.severity == "INFO")

        md.append(f"- **Critical Alerts**: `🔴 {crit_count}` | **Warning Alerts**: `🟡 {warn_count}` | **Info Alerts**: `🔵 {info_count}`")
        md.append("")

        if alerts:
            md.append("| Alert ID | Timestamp | Severity | Anomaly Type & Parameter | Raw vs Corrected | Confidence | Historical Recurrence Context | Recommended Action |")
            md.append("|---|---|---|---|---|---|---|---|")
            for a in alerts[:20]:
                sev_badge = "🔴 **CRITICAL**" if a.severity == "CRITICAL" else "🟡 **WARNING**" if a.severity == "WARNING" else "🔵 **INFO**"
                params_str = ", ".join(a.affected_parameters)
                md.append(
                    f"| `{a.alert_id}` | `{a.timestamp}` | {sev_badge} | **{a.anomaly_type}** (`{params_str}`) | "
                    f"Raw: `{a.raw_reading}`<br>Corr: `{a.corrected_estimate}` | `{a.confidence_score * 100:.1f}%` | "
                    f"*{a.historical_context}* | **{a.recommended_action}** |"
                )
            if len(alerts) > 20:
                md.append(f"\n*(Showing top 20 of {len(alerts)} active alerts)*\n")
        else:
            md.append("✅ *Zero active alerts. All instruments operating nominally.*")
        md.append("")

        # 3. Sensor Health & Predictive Maintenance Summary
        md.append("## 3. Sensor Health Diagnostics & RUL Forecast")
        md.append("| Sensor | Model | Health Score | RUL (Days) | Status | Maintenance Action |")
        md.append("|---|---|---|---|---|---|")
        for s_name, m in health_metrics.items():
            filled = int(m.health_score / 10)
            bar = "█" * filled + "░" * (10 - filled)
            md.append(f"| `{s_name.capitalize()}` | {m.instrument_model} | `[{bar}]` **{m.health_score}%** | **{m.predicted_rul_days:.0f} d** | `{m.maintenance_status}` | {m.recommended_action} |")
        md.append("")

        return "\n".join(md)

    def _build_dashboard_text(
        self,
        station_id: str,
        alerts: List[ActionableAlert],
        health_metrics: Dict[str, SensorHealthMetrics]
    ) -> str:
        """Build Plaintext console dashboard."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"       REAL-TIME AWS SENSOR AUTO-CORRECTION & ALERT DASHBOARD [{station_id}]")
        lines.append("=" * 80)
        lines.append(" SENSOR HEALTH SUMMARY:")
        for s_name, m in health_metrics.items():
            lines.append(f"  • {s_name.capitalize():<12}: Health={m.health_score:>5.1f}% [{m.health_grade:<8}] | Drift={m.drift_rate_per_day:+.4f}/day | RUL={m.predicted_rul_days:>4.0f}d | Status={m.maintenance_status}")
        lines.append("-" * 80)
        lines.append(f" ACTIVE ALERTS ({len(alerts)} total):")
        if alerts:
            for a in alerts[:6]:
                lines.append(f"  [{a.severity:<8}] {a.alert_id} @ {a.timestamp} | {a.anomaly_type}")
                lines.append(f"    -> Readings: Raw=[{a.raw_reading}] | Corrected=[{a.corrected_estimate}] | Conf={a.confidence_score * 100:.0f}%")
                lines.append(f"    -> History:  {a.historical_context}")
                lines.append(f"    -> Action:   {a.recommended_action}")
                lines.append("")
        else:
            lines.append("  No active alerts.")
        lines.append("=" * 80)
        return "\n".join(lines)
