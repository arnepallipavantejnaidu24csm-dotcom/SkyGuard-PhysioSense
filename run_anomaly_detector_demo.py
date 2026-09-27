"""
Demonstration and Experiment Runner for the Hybrid Anomaly Detection Module.

Ensembles:
1. Isolation Forest (Unsupervised point & multi-variate detector)
2. PyTorch LSTM Autoencoder (Temporal pattern & sequence dynamics error)
3. Physics-informed rules (Thermodynamic High severity, Sensor Spec Medium severity, Statistical Low severity)

Generates:
- Multi-level Alerts: CRITICAL, WARNING, INFO, NORMAL
- Calibrated Confidence Scores (0.0 - 1.0)
- Actionable Maintenance Recommendations
"""

import os
import numpy as np
import pandas as pd

from aws_weather_preprocessor import (
    AWSHybridAnomalyDetector,
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


def main():
    print("=" * 80)
    print(" [HYBRID ANOMALY DETECTION ENGINE (ISOLATION FOREST + LSTM AUTOENCODER + PHYSICS RULES)]")
    print("=" * 80)

    # 1. Generate Synthetic Normal Baseline Training Data
    print("\n[Step 1/4] Generating 7-day normal operational baseline time-series (2,016 readings)...")
    train_config = SyntheticDataConfig(
        start_time="2026-08-01 00:00:00",
        duration_days=7.0,
        frequency_minutes=5,
        inject_anomalies=False,
        random_seed=101,
    )
    normal_train_df, _ = generate_synthetic_aws_data(train_config)
    print(f" -> Generated {len(normal_train_df):,} baseline records.")

    # 2. Train Hybrid Detector
    print("\n[Step 2/4] Training Hybrid Anomaly Detection Ensemble...")
    detector = AWSHybridAnomalyDetector(
        channels=["temperature", "pressure", "humidity", "wind_speed"],
        lstm_seq_len=12,
        lstm_epochs=20,
        iforest_contamination=0.04,
    )
    detector.train(normal_train_df)

    # 3. Generate Test Dataset with Diverse Injected Anomalies
    print("\n[Step 3/4] Generating test dataset with injected multi-level anomalies...")
    test_config = SyntheticDataConfig(
        start_time="2026-09-01 00:00:00",
        duration_days=7.0,
        frequency_minutes=5,
        inject_anomalies=True,
        random_seed=202,
    )
    test_df, gt_df = generate_synthetic_aws_data(test_config)
    print(f" -> Generated {len(test_df):,} test records with {len(gt_df)} ground-truth anomaly events.")

    # 4. Run Hybrid Anomaly Detection
    print("\n[Step 4/4] Executing Hybrid Inference & Alert Synthesis...")
    result = detector.detect(test_df)
    print(" -> Hybrid detection complete!")

    # 5. Display Alert Breakdown Summary
    print("\n" + "=" * 80)
    print(" MULTI-LEVEL ALERT SUMMARY")
    print("=" * 80)
    
    severity_counts = result.summary_df["severity"].value_counts().to_dict()
    total_records = len(result.summary_df)
    total_anomalies = len(result.alerts_df)

    print(f" Total Records Evaluated: {total_records:,}")
    print(f" Total Anomaly Alerts:    {total_anomalies:,} ({total_anomalies / total_records * 100:.2f}%)")
    print(f"   * CRITICAL Alerts:     {severity_counts.get('CRITICAL', 0):,}")
    print(f"   * WARNING Alerts:      {severity_counts.get('WARNING', 0):,}")
    print(f"   * INFO Alerts:         {severity_counts.get('INFO', 0):,}")
    print(f"   * NORMAL Readings:     {severity_counts.get('NORMAL', 0):,}")

    # Top sample alerts
    print("\n SAMPLE FLAGGED ALERTS (Top 10):")
    sample_alerts = result.alerts_df[["timestamp", "severity", "confidence", "anomaly_type", "recommended_action"]].head(10)
    print(sample_alerts.to_string(index=False))

    # 6. Export Results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Full results CSV
    full_csv = os.path.join(output_dir, "hybrid_anomaly_detection_results.csv")
    result.summary_df.to_csv(full_csv, index=False)
    print(f"\n -> Full detection results exported to: {full_csv}")

    # Alerts only CSV
    alerts_csv = os.path.join(output_dir, "hybrid_anomaly_alerts.csv")
    result.alerts_df.to_csv(alerts_csv, index=False)
    print(f" -> Flagged alerts table exported to: {alerts_csv}")

    # Generate Markdown Report
    report_md_path = os.path.join(output_dir, "hybrid_anomaly_detection_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# 🛡️ Hybrid AWS Anomaly Detection & Alert Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Records Analyzed**: `{total_records:,}` readings\n")
        f.write(f"- **Total Anomalies Flagged**: `{total_anomalies:,}` incidents ({total_anomalies / total_records * 100:.2f}%)\n")
        f.write(f"- **Critical Alerts**: `{severity_counts.get('CRITICAL', 0):,}`\n")
        f.write(f"- **Warning Alerts**: `{severity_counts.get('WARNING', 0):,}`\n")
        f.write(f"- **Info Alerts**: `{severity_counts.get('INFO', 0):,}`\n\n")

        f.write("## 2. Multi-Model Detection Hierarchy\n")
        f.write("| Component | Detection Scope | Severity Mapping |\n")
        f.write("|---|---|---|\n")
        f.write("| **Physics Rule Engine** | Thermodynamic laws ($T_d \\le T, e < P, RH \\le 100\\%$) | **HIGH** $\\to$ `CRITICAL` |\n")
        f.write("| **Physics Rule Engine** | Sensor specifications ($WS \\ge 0$, operating ranges) | **MEDIUM** $\\to$ `WARNING` |\n")
        f.write("| **LSTM Autoencoder** | Temporal sequence reconstruction & pattern distortion | **MEDIUM/HIGH** $\\to$ `WARNING`/`CRITICAL` |\n")
        f.write("| **Isolation Forest** | Multi-variate point & cross-channel correlation outliers | **MEDIUM** $\\to$ `WARNING` |\n")
        f.write("| **Statistical Rules** | Transient rate-of-change spikes & flatlines | **LOW** $\\to$ `INFO`/`WARNING` |\n\n")

        f.write("## 3. Flagged Anomaly Incidents & Recommended Actions\n\n")
        sample_md = result.alerts_df.head(25)[["timestamp", "severity", "confidence", "anomaly_type", "affected_sensors", "recommended_action"]]
        f.write(sample_md.to_markdown(index=False))
        if len(result.alerts_df) > 25:
            f.write(f"\n\n*(Showing first 25 of {len(result.alerts_df):,} flagged alerts)*\n")

    print(f" -> Markdown Anomaly Detection Report exported to: {report_md_path}")
    print("\n" + "=" * 80)
    print(" [SUCCESS] Hybrid Anomaly Detection completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
