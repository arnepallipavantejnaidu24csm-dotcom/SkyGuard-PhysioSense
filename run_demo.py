"""
Demonstration and CLI runner for the AWS Sensor Data Preprocessing Pipeline.

Executes:
1. Synthetic multi-sensor AWS weather dataset generation with diurnal dynamics
2. Physics-based consistency checks (T-P-RH thermodynamic relationships & Wind speed bounds)
3. Statistical anomaly & outlier detection (IQR, Rolling Z-score, Spikes, Stuck Sensors, Missing/Sentinels)
4. Data cleaning, imputation, and flag matrix construction
5. Data Quality Report generation and file exports
"""

import os
import sys
import pandas as pd
from datetime import datetime

from aws_weather_preprocessor import (
    AWSDataPreprocessor,
    PreprocessingConfig,
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


def main():
    print("=" * 80)
    print(" [AWS SENSOR DATA PREPROCESSING & QUALITY CONTROL PIPELINE]")
    print("=" * 80)
    
    # 1. Generate Synthetic AWS Weather Sensor Data
    print("\n[Step 1/5] Generating 7-day high-resolution (5-minute) synthetic AWS time-series...")
    synth_config = SyntheticDataConfig(
        start_time="2026-09-01 00:00:00",
        duration_days=7.0,
        frequency_minutes=5,
        inject_anomalies=True,
        random_seed=42,
    )
    raw_df, ground_truth_df = generate_synthetic_aws_data(synth_config)
    print(f" -> Generated {len(raw_df):,} records with realistic diurnal cycles and injected anomalies.")
    print(f" -> Columns: {list(raw_df.columns)}")
    print(f" -> Injected Anomaly Events count: {len(ground_truth_df)}")

    # 2. Configure Preprocessor
    print("\n[Step 2/5] Initializing AWS Data Preprocessor with Physics & Statistical Rules...")
    preprocessor = AWSDataPreprocessor(
        config=PreprocessingConfig(
            imputation_strategy="interpolate",
            max_impute_gap_periods=6,
            include_derived_thermodynamics=True,
        )
    )

    # 3. Execute Preprocessing Pipeline
    print("\n[Step 3/5] Executing Preprocessing Pipeline...")
    result = preprocessor.process(raw_df)
    print(" -> Processing complete!")

    # 4. Display Quality Report Summary
    print("\n[Step 4/5] Data Quality Report Summary:")
    print(result.report.to_text())

    # 5. Export Output Artifacts
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Step 5/5] Exporting preprocessed datasets, flags, and reports to '{output_dir}/'...")

    # A. Raw synthetic data
    raw_path = os.path.join(output_dir, "sample_raw_aws_data.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f" -> Raw data exported to: {raw_path}")

    # B. Cleaned data
    cleaned_path = os.path.join(output_dir, "cleaned_aws_data.csv")
    result.cleaned_df.to_csv(cleaned_path, index=False)
    print(f" -> Cleaned dataset exported to: {cleaned_path}")

    # C. Anomaly flags
    flags_path = os.path.join(output_dir, "aws_anomaly_flags.csv")
    result.flags_df.to_csv(flags_path, index=False)
    print(f" -> Granular anomaly flags exported to: {flags_path}")

    # D. Unified complete dataset (Raw + Clean + Derived Thermodynamics + Flags)
    full_path = os.path.join(output_dir, "aws_preprocessed_full.csv")
    full_df = result.get_full_dataset()
    full_df.to_csv(full_path, index=False)
    print(f" -> Full unified dataset exported to: {full_path}")

    # E. Granular anomaly incident log
    anom_log_path = os.path.join(output_dir, "anomalies_detected_log.csv")
    anom_df = result.report.get_anomalies_df()
    anom_df.to_csv(anom_log_path, index=False)
    print(f" -> Granular anomaly locations log ({len(anom_df)} items) exported to: {anom_log_path}")

    # F. Markdown report
    report_path = os.path.join(output_dir, "data_quality_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.report.to_markdown())
    print(f" -> Markdown Quality Report exported to: {report_path}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] All preprocessing tasks and exports completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
