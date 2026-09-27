"""
Demonstration Runner for the Sensor Auto-Correction, Uncertainty Bounding, and Alerting System.

Executes:
1. State-Space Kalman Filter bias tracking & physical auto-correction
2. Multi-level Confidence Interval estimation (95% and 99% CIs)
3. Actionable Alert Generation with Root Cause & Historical Recurrence Context
4. Dataset Exports (CSV & JSON format)
5. Real-Time Alert & Sensor Health Dashboard
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from aws_weather_preprocessor import (
    AWSCorrectionAndAlertingPipeline,
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


def main():
    print("=" * 80)
    print(" [AWS SENSOR AUTO-CORRECTION, UNCERTAINTY BOUNDING & ACTIONABLE ALERTING]")
    print("=" * 80)

    # 1. Generate 7-day Multi-Sensor AWS Dataset with Real-World Anomalies and Drift
    print("\n[Step 1/5] Ingesting 7-day AWS multi-sensor time-series (2,016 readings)...")
    config = SyntheticDataConfig(
        start_time="2026-09-01 00:00:00",
        duration_days=7.0,
        frequency_minutes=5,
        inject_anomalies=True,
        random_seed=42,
    )
    raw_df, _ = generate_synthetic_aws_data(config)
    print(f" -> Ingested {len(raw_df):,} records for station AWS-PRIMARY-01.")

    # 2. Run Integrated Correction & Alerting Pipeline
    print("\n[Step 2/5] Initializing & Executing Correction and Alerting Pipeline...")
    pipeline = AWSCorrectionAndAlertingPipeline(lookback_history_hours=48.0)
    result = pipeline.process(
        raw_df=raw_df,
        station_id="AWS-PRIMARY-MET-01",
        timestamp_col="timestamp",
    )
    print(" -> Pipeline processing complete!")

    # 3. Display Real-Time Console Dashboard
    print("\n[Step 3/5] Real-Time Alert Dashboard Summary:")
    print(result.dashboard_text)

    # 4. Display Sample Auto-Corrected Telemetry with Confidence Intervals
    print("\n[Step 4/5] Sample Auto-Corrected Readings (with 95% & 99% Confidence Intervals):")
    cols_to_show = [
        "timestamp",
        "temperature_raw", "temperature_bias_est", "temperature_corrected", "temperature_ci95_lower", "temperature_ci95_upper",
        "pressure_raw", "pressure_bias_est", "pressure_corrected", "pressure_ci95_lower", "pressure_ci95_upper",
    ]
    sample_corrected = result.corrected_dataset_df[cols_to_show].head(5)
    print(sample_corrected.to_string(index=False))

    # 5. Export Datasets & Dashboard Artifacts
    print("\n[Step 5/5] Exporting outputs to 'output/'...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # A. Corrected Dataset CSV
    csv_path = os.path.join(output_dir, "aws_corrected_dataset.csv")
    result.export_csv(csv_path)
    print(f" -> Corrected dataset CSV exported to: {csv_path}")

    # B. Corrected Dataset JSON
    json_path = os.path.join(output_dir, "aws_corrected_dataset.json")
    result.export_json(json_path)
    print(f" -> Corrected dataset JSON exported to: {json_path}")

    # C. Actionable Alerts CSV Log
    alerts_csv_path = os.path.join(output_dir, "aws_actionable_alerts_log.csv")
    result.export_alerts_csv(alerts_csv_path)
    print(f" -> Actionable alerts log ({len(result.actionable_alerts)} alerts) exported to: {alerts_csv_path}")

    # D. Markdown Real-Time Dashboard
    dashboard_md_path = os.path.join(output_dir, "aws_realtime_alert_dashboard.md")
    with open(dashboard_md_path, "w", encoding="utf-8") as f:
        f.write(result.dashboard_markdown)
    print(f" -> Markdown Real-Time Dashboard exported to: {dashboard_md_path}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] All auto-correction, alerting, and export tasks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
