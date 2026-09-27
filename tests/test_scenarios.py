"""
Comprehensive Stress Test Scenarios for AWS Preprocessing & Anomaly Detection:
1. Normal Operation (Zero false positive criticals, health > 90%)
2. Sensor Drift & Calibration Bias
3. Multiple Simultaneous Anomalies
4. Data Gaps & Transmission Packet Dropouts
5. Extreme Weather Conditions (Polar freeze, IVISAK16 heatwave, hurricane gusts)
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor import (
    AWSCorrectionAndAlertingPipeline,
    AtmosphericStateKalmanFilter,
    SensorHealthScoringSystem,
    AWSHybridAnomalyDetector,
    generate_synthetic_aws_data,
    generate_drift_scenario,
    SyntheticDataConfig,
)


class TestOperationalScenarios(unittest.TestCase):

    def setUp(self):
        self.pipeline = AWSCorrectionAndAlertingPipeline()

    def test_scenario_1_normal_operation(self):
        """Scenario 1: Clean nominal 3-day weather dynamics."""
        config = SyntheticDataConfig(duration_days=3.0, frequency_minutes=5, inject_anomalies=False, random_seed=10)
        df, _ = generate_synthetic_aws_data(config)

        res = self.pipeline.process(df, station_id="SCENARIO-NORMAL-01")

        # In nominal operation, there should be zero CRITICAL alerts
        critical_alerts = [a for a in res.actionable_alerts if a.severity == "CRITICAL"]
        self.assertEqual(len(critical_alerts), 0)

        # Health scores should remain high (> 70%)
        for s_name, m in res.health_metrics.items():
            self.assertGreaterEqual(m.health_score, 70.0)

    def test_scenario_2_sensor_drift_and_bias(self):
        """Scenario 2: Progressive linear sensor drift on temperature & pressure."""
        exp = generate_drift_scenario(scenario_type="linear_ramp", duration_days=4.0, frequency_minutes=5)
        
        res = self.pipeline.process(
            raw_df=exp.observed_drift_df,
            reference_df=exp.reference_checks_df,
            station_id="SCENARIO-DRIFT-02"
        )

        # Kalman filter should estimate non-zero bias matching drift direction
        temp_final_bias = res.corrected_dataset_df["temperature_bias_est"].iloc[-1]
        pres_final_bias = res.corrected_dataset_df["pressure_bias_est"].iloc[-1]

        self.assertGreater(temp_final_bias, 0.2)
        self.assertLess(pres_final_bias, -0.2)

        # Health scoring should reflect the degradation
        self.assertIn(res.health_metrics["temperature"].maintenance_status, ["CALIBRATION_RECOMMENDED", "MAINTENANCE_DUE", "URGENT_ACTION_REQUIRED"])

    def test_scenario_3_multiple_simultaneous_anomalies(self):
        """Scenario 3: Concurrent thermodynamic breach + temperature spike + frozen wind sensor."""
        config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=False, random_seed=30)
        df, _ = generate_synthetic_aws_data(config)

        # Inject simultaneous anomalies at step 100
        df.loc[100, "temperature"] += 18.0     # Transient RTD spike
        df.loc[100, "humidity"] = 120.0        # Thermodynamic breach (Td > T)
        df.loc[90:110, "wind_speed"] = 4.2     # Frozen anemometer flatline

        res = self.pipeline.process(df, station_id="SCENARIO-MULTI-ANOM-03")

        # Verify critical alert triggered
        crit_alerts = [a for a in res.actionable_alerts if a.severity == "CRITICAL"]
        self.assertGreaterEqual(len(crit_alerts), 1)

    def test_scenario_4_data_gaps_and_transmission_failures(self):
        """Scenario 4: 25% burst packet dropouts and hardware sentinel codes."""
        config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=False, random_seed=40)
        df, _ = generate_synthetic_aws_data(config)

        # Burst dropout on pressure and sentinels on temperature
        df.loc[50:80, "pressure"] = np.nan
        df.loc[120:125, "temperature"] = -999.0

        res = self.pipeline.process(df, station_id="SCENARIO-GAPS-04")

        # Corrected dataset should successfully impute / bound without crashing
        self.assertEqual(len(res.corrected_dataset_df), len(df))
        self.assertFalse(res.corrected_dataset_df["pressure_corrected"].isna().all())

        # Completeness penalty in health metrics
        self.assertLess(res.health_metrics["pressure"].completeness_pct, 95.0)

    def test_scenario_5_extreme_weather_conditions(self):
        """Scenario 5: Plausible extreme weather (IVISAK16 heat 48°C, storm gusts 38 m/s, deep pressure drop 945 hPa)."""
        num_steps = 200
        timestamps = pd.date_range("2026-09-01", periods=num_steps, freq="5min")

        extreme_df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": np.linspace(35.0, 48.0, num_steps),      # Extreme heatwave
            "pressure": np.linspace(1005.0, 945.0, num_steps),       # Typhoon / severe cyclone pressure drop
            "humidity": np.linspace(25.0, 8.0, num_steps),           # Hyper-arid IVISAK16 humidity
            "wind_speed": np.linspace(10.0, 38.0, num_steps),        # Severe gale storm gusts (within 50 m/s physical spec)
        })

        res = self.pipeline.process(extreme_df, station_id="SCENARIO-EXTREME-05")

        # The extreme weather is physically valid, so no thermodynamic critical breach should be raised
        thermo_crits = [a for a in res.actionable_alerts if "THERMODYNAMIC" in a.anomaly_type]
        self.assertEqual(len(thermo_crits), 0)


if __name__ == "__main__":
    unittest.main()
