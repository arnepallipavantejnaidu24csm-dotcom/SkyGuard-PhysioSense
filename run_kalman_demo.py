"""
Demonstration and Experiment Runner for the State-Space Kalman Filter.

Runs:
1. Multi-instrument sensor drift simulation scenarios (Linear ramp, sudden calibration step, Brownian drift)
2. State-Space Kalman Filtering tracking true atmospheric states vs instrument biases
3. Residual and innovation tracking, uncertainty estimation (95% confidence intervals)
4. Comprehensive comparison tables and file exports
"""

import os
import numpy as np
import pandas as pd

from aws_weather_preprocessor import (
    AtmosphericStateKalmanFilter,
    KalmanFilterConfig,
    generate_drift_scenario,
    DriftExperimentResult,
)


def run_drift_experiment(scenario_name: str, scenario_type: str, duration_days: float = 5.0):
    print("\n" + "=" * 80)
    print(f" EXPERIMENT: {scenario_name.upper()} (Duration: {duration_days} days)")
    print("=" * 80)

    # 1. Generate Synthetic Sensor Drift Scenario
    print(f"[1/4] Simulating AWS true atmospheric dynamics & injecting '{scenario_type}' sensor drift...")
    exp: DriftExperimentResult = generate_drift_scenario(
        scenario_type=scenario_type,
        duration_days=duration_days,
        frequency_minutes=5,
        random_seed=42,
    )
    n_samples = len(exp.observed_drift_df)
    print(f" -> Generated {n_samples:,} observations at 5-minute sampling.")

    # 2. Run Kalman Filter
    print("[2/4] Running State-Space Kalman Filter...")
    kf = AtmosphericStateKalmanFilter(
        config=KalmanFilterConfig(
            channels=["temperature", "pressure", "humidity", "wind_speed"],
            process_noise_weather={"temperature": 0.04, "pressure": 0.02, "humidity": 0.12, "wind_speed": 0.18},
            process_noise_bias={"temperature": 0.003, "pressure": 0.002, "humidity": 0.006, "wind_speed": 0.003},
            measurement_noise={"temperature": 0.25, "pressure": 0.20, "humidity": 1.0, "wind_speed": 0.40},
        )
    )

    output = kf.filter_dataframe(
        df=exp.observed_drift_df,
        reference_df=exp.reference_checks_df
    )
    print(" -> Kalman Filtering complete!")

    # 3. Evaluate Performance Metrics
    print("[3/4] Performance Evaluation & Bias Estimation Results:")
    eval_rows = []
    for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
        true_val_series = exp.true_states_df[ch]
        obs_val_series = exp.observed_drift_df[ch]
        est_val_series = output.state_estimates_df[f"{ch}_true"]
        
        true_bias_series = exp.true_biases_df[f"{ch}_true_bias"]
        est_bias_series = output.bias_estimates_df[f"{ch}_bias"]
        
        # Errors
        raw_rmse = np.sqrt(((obs_val_series - true_val_series) ** 2).mean())
        filtered_rmse = np.sqrt(((est_val_series - true_val_series) ** 2).mean())
        bias_rmse = np.sqrt(((est_bias_series - true_bias_series) ** 2).mean())
        
        eval_rows.append({
            "Sensor Channel": ch.capitalize(),
            "Raw Obs RMSE": f"{raw_rmse:.3f}",
            "KF True State RMSE": f"{filtered_rmse:.3f}",
            "RMSE Reduction": f"{(1 - filtered_rmse / raw_rmse) * 100:.1f}%",
            "True Final Bias": f"{true_bias_series.iloc[-1]:.3f}",
            "Estimated Bias": f"{est_bias_series.iloc[-1]:.3f}",
            "Bias Tracking RMSE": f"{bias_rmse:.3f}",
            "Residual Mean": f"{output.summary_metrics.get(f'{ch}_residual_mean', 0.0):.3f}",
            "Residual Std": f"{output.summary_metrics.get(f'{ch}_residual_std', 0.0):.3f}",
        })

    eval_df = pd.DataFrame(eval_rows).set_index("Sensor Channel")
    print(eval_df.to_string())

    # 4. Export Artifacts
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = f"kalman_{scenario_type}"
    # Combined dataset: True, Observed, Estimated True, Estimated Bias, Residuals, 95% CI
    combined_export = pd.DataFrame({
        "timestamp": exp.observed_drift_df["timestamp"],
        # Temperature
        "temp_true_ground_truth": exp.true_states_df["temperature"],
        "temp_observed_drifted": exp.observed_drift_df["temperature"],
        "temp_true_estimated": output.state_estimates_df["temperature_true"],
        "temp_true_ci95_lower": output.full_output_df["temperature_true_ci95_lower"],
        "temp_true_ci95_upper": output.full_output_df["temperature_true_ci95_upper"],
        "temp_bias_ground_truth": exp.true_biases_df["temperature_true_bias"],
        "temp_bias_estimated": output.bias_estimates_df["temperature_bias"],
        "temp_bias_ci95_lower": output.full_output_df["temperature_bias_ci95_lower"],
        "temp_bias_ci95_upper": output.full_output_df["temperature_bias_ci95_upper"],
        "temp_innovation": output.residuals_df["temperature_innovation"],
        "temp_post_fit_residual": output.residuals_df["temperature_post_fit_residual"],
        # Pressure
        "pres_true_ground_truth": exp.true_states_df["pressure"],
        "pres_observed_drifted": exp.observed_drift_df["pressure"],
        "pres_true_estimated": output.state_estimates_df["pressure_true"],
        "pres_bias_ground_truth": exp.true_biases_df["pressure_true_bias"],
        "pres_bias_estimated": output.bias_estimates_df["pressure_bias"],
        "pres_post_fit_residual": output.residuals_df["pressure_post_fit_residual"],
        # Humidity
        "rh_true_ground_truth": exp.true_states_df["humidity"],
        "rh_observed_drifted": exp.observed_drift_df["humidity"],
        "rh_true_estimated": output.state_estimates_df["humidity_true"],
        "rh_bias_ground_truth": exp.true_biases_df["humidity_true_bias"],
        "rh_bias_estimated": output.bias_estimates_df["humidity_bias"],
        "rh_post_fit_residual": output.residuals_df["humidity_post_fit_residual"],
        # Wind Speed
        "wind_true_ground_truth": exp.true_states_df["wind_speed"],
        "wind_observed_drifted": exp.observed_drift_df["wind_speed"],
        "wind_true_estimated": output.state_estimates_df["wind_speed_true"],
        "wind_bias_ground_truth": exp.true_biases_df["wind_speed_true_bias"],
        "wind_bias_estimated": output.bias_estimates_df["wind_speed_bias"],
        "wind_post_fit_residual": output.residuals_df["wind_speed_post_fit_residual"],
        "nis": output.residuals_df["nis"],
    })

    csv_path = os.path.join(output_dir, f"{prefix}_tracking_results.csv")
    combined_export.to_csv(csv_path, index=False)
    print(f" -> Detailed tracking results exported to: {csv_path}")

    return eval_df, combined_export


def main():
    print("=" * 80)
    print(" [STATE-SPACE KALMAN FILTER SENSOR BIAS & TRUE STATE TRACKING]")
    print("=" * 80)

    # Run multiple benchmark scenarios
    scenarios = [
        ("Linear Ramp Sensor Drift", "linear_ramp", 5.0),
        ("Sudden Calibration Step Offset", "step_jump", 5.0),
        ("Stochastic Brownian Walk Drift", "brownian_drift", 5.0),
    ]

    all_summaries = {}
    for name, stype, duration in scenarios:
        summary_df, _ = run_drift_experiment(name, stype, duration)
        all_summaries[name] = summary_df

    # Write summary markdown report
    report_path = os.path.join("output", "kalman_state_space_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🛰️ Atmospheric State-Space Kalman Filter & Sensor Bias Tracking Report\n\n")
        f.write("## 1. Formulation & Architecture\n")
        f.write("- **State Vector**: Partitioned into True Atmospheric State $[x_w, \\dot{x}_w]$ and Sensor Biases $[b_s]$.\n")
        f.write("- **State Dynamics**: Continuous white-noise acceleration kinematic model for atmospheric variables and random-walk diffusion for instrument drift.\n")
        f.write("- **Measurement Model**: $\\mathbf{z}_k = \\mathbf{H} \\mathbf{x}_k + \\mathbf{v}_k$, where $z_i = x_{true, i} + b_i + v_i$.\n")
        f.write("- **Residual Tracking**: Computes pre-fit innovations $\\mathbf{y}_k = \\mathbf{z}_k - \\mathbf{H}\\hat{\\mathbf{x}}_{k|k-1}$, post-fit residuals $\\mathbf{e}_k = \\mathbf{z}_k - \\mathbf{H}\\hat{\\mathbf{x}}_{k|k}$, and Normalized Innovation Squared (NIS).\n\n")

        f.write("## 2. Benchmark Scenario Performance\n\n")
        for sname, s_df in all_summaries.items():
            f.write(f"### {sname}\n\n")
            f.write(s_df.to_markdown(index=True))
            f.write("\n\n")

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Kalman Filter experiments completed. Report written to {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
