"""
Demonstration Runner for the Sensor Health Scoring & Predictive Maintenance System.

Executes:
1. Ingestion of multi-sensor AWS weather time-series with Kalman filter bias tracking & residuals
2. Comprehensive multi-metric sensor performance evaluation (Completeness, Drift Rate, Noise Level, Calibration Status)
3. 0-100% Health Score assignment & Degradation Grading
4. Predictive Maintenance forecasting (Remaining Useful Life / RUL in days)
5. Degradation flagging in anomaly assessment
6. Markdown Health Dashboard & Prioritized Maintenance Alerts export
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import asdict

from aws_weather_preprocessor import (
    AWSDataPreprocessor,
    AtmosphericStateKalmanFilter,
    SensorHealthScoringSystem,
    generate_drift_scenario,
    DriftExperimentResult,
)


def main():
    print("=" * 80)
    print(" [AUTOMATIC WEATHER STATION (AWS) SENSOR HEALTH SCORING & PREDICTIVE MAINTENANCE]")
    print("=" * 80)

    # 1. Simulate 7-Day AWS Multi-Sensor Time-Series with Realistic Degradation
    print("\n[Step 1/5] Simulating 7-day AWS multi-sensor operation with degradation modes...")
    # Inject linear drift on temperature, step shift on humidity, packet loss on wind speed
    exp: DriftExperimentResult = generate_drift_scenario(
        scenario_type="multi_fault",
        duration_days=7.0,
        frequency_minutes=5,
        random_seed=42,
    )
    raw_df = exp.observed_drift_df.copy()

    # Add realistic packet loss on wind speed (5% missing readings)
    np.random.seed(123)
    dropout_idx = np.random.choice(len(raw_df), size=int(len(raw_df) * 0.04), replace=False)
    raw_df.loc[dropout_idx, "wind_speed"] = np.nan

    print(f" -> Ingested {len(raw_df):,} records across 4 primary meteorological instruments.")

    # 2. Run Kalman Filter for Continuous Bias & Residual Tracking
    print("\n[Step 2/5] Running State-Space Kalman Filter for bias and residual estimation...")
    kf = AtmosphericStateKalmanFilter()
    kf_output = kf.filter_dataframe(raw_df, reference_df=exp.reference_checks_df)
    print(" -> Kalman Filter tracking complete!")

    # 3. Initialize Health Scoring System
    print("\n[Step 3/5] Computing Multi-Metric Sensor Health Scores (0 - 100%)...")
    health_system = SensorHealthScoringSystem()
    health_metrics = health_system.evaluate_all(
        raw_df=raw_df,
        bias_df=kf_output.bias_estimates_df,
        residuals_df=kf_output.residuals_df,
        current_date="2026-09-24",
        dt_minutes=5.0,
    )

    # 4. Display Health Summary Table
    print("\n" + "=" * 80)
    print(" SENSOR HEALTH DIAGNOSTICS & MAINTENANCE STATUS")
    print("=" * 80)
    summary_rows = []
    for s_name, m in health_metrics.items():
        summary_rows.append({
            "Sensor": s_name.capitalize(),
            "Model": m.instrument_model,
            "Health %": f"{m.health_score}%",
            "Grade": m.health_grade,
            "Completeness": f"{m.completeness_pct}%",
            "Drift/Day": f"{m.drift_rate_per_day:+.4f}",
            "Noise Std": f"{m.noise_std:.3f}",
            "Cal Status": m.calibration_status,
            "RUL (Days)": f"{m.predicted_rul_days:.0f} d",
            "Maint Status": m.maintenance_status,
        })
    summary_table = pd.DataFrame(summary_rows).set_index("Sensor")
    print(summary_table.to_string())

    # 5. Prioritized Maintenance Alerts
    print("\n[Step 4/5] Evaluating Predictive Maintenance Alerts...")
    alerts = health_system.generate_maintenance_alerts(health_metrics, timestamp="2026-09-24 12:00:00")
    if alerts:
        for i, a in enumerate(alerts, 1):
            print(f"\n [ALERT #{i} - {a.urgency}] Sensor: {a.sensor_name.upper()} ({a.instrument_model})")
            print(f"   * Trigger: {a.trigger_reason}")
            print(f"   * Health Score: {a.current_health_score}% | Predicted Remaining Life: {a.predicted_rul_days:.0f} days")
            print(f"   * Action Required: {a.action_instructions}")
    else:
        print(" -> No urgent maintenance alerts active.")

    # 6. Export Artifacts
    print("\n[Step 5/5] Exporting Health Dashboard & Maintenance Reports to 'output/'...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Metrics CSV
    metrics_records = [asdict(m) for m in health_metrics.values()]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = os.path.join(output_dir, "sensor_health_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f" -> Health metrics exported to: {metrics_csv}")

    # Alerts CSV
    if alerts:
        alerts_records = [asdict(a) for a in alerts]
        alerts_df = pd.DataFrame(alerts_records)
    else:
        alerts_df = pd.DataFrame(columns=["timestamp", "sensor_name", "instrument_model", "urgency", "trigger_reason", "current_health_score", "predicted_rul_days", "action_instructions"])
    alerts_csv = os.path.join(output_dir, "sensor_maintenance_alerts.csv")
    alerts_df.to_csv(alerts_csv, index=False)
    print(f" -> Maintenance alerts exported to: {alerts_csv}")

    # Markdown Dashboard
    dashboard_md = health_system.generate_dashboard_markdown(health_metrics, station_id="AWS-STATION-NORTH-01")
    dashboard_path = os.path.join(output_dir, "sensor_health_dashboard.md")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_md)
    print(f" -> Markdown Health Dashboard exported to: {dashboard_path}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] Sensor Health Scoring & Predictive Maintenance Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
