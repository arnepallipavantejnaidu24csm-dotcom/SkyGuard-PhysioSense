"""
Unit tests for the Correction & Alerting Module.
"""

import os
import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.correction_alerting import (
    HistoricalAnomalyTracker,
    SensorAutoCorrector,
    ActionableAlert,
    AWSCorrectionAndAlertingPipeline,
)
from aws_weather_preprocessor.synthetic_data import (
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


class TestHistoricalAnomalyTracker(unittest.TestCase):

    def test_recurrence_context_tracking(self):
        tracker = HistoricalAnomalyTracker(lookback_window_hours=48.0)

        # 1st incident
        c1 = tracker.log_and_query_context(
            timestamp="2026-09-01 10:00:00",
            anomaly_type="RATE_OF_CHANGE_SPIKE",
            affected_sensors=["temperature"]
        )
        self.assertIn("First occurrence", c1)

        # 2nd incident 2 hours later
        c2 = tracker.log_and_query_context(
            timestamp="2026-09-01 12:00:00",
            anomaly_type="RATE_OF_CHANGE_SPIKE",
            affected_sensors=["temperature"]
        )
        self.assertIn("2nd occurrence", c2)

        # 3rd incident
        c3 = tracker.log_and_query_context(
            timestamp="2026-09-01 15:00:00",
            anomaly_type="RATE_OF_CHANGE_SPIKE",
            affected_sensors=["temperature"]
        )
        self.assertIn("Recurring pattern", c3)
        self.assertIn("3 times", c3)


class TestSensorAutoCorrector(unittest.TestCase):

    def test_auto_correction_and_ci(self):
        raw = pd.Series([25.0, 26.0, 24.0])
        bias = pd.Series([1.5, 1.5, 1.5])
        sigma = pd.Series([0.2, 0.2, 0.2])

        res = SensorAutoCorrector.correct_series(
            raw_series=raw,
            bias_series=bias,
            std_series=sigma,
            sensor_name="temperature"
        )

        # Corrected values: 25 - 1.5 = 23.5
        self.assertEqual(res["temperature_corrected"].iloc[0], 23.5)
        
        # 95% CI: 23.5 +/- 1.96 * 0.2 = [23.11, 23.89]
        self.assertAlmostEqual(res["temperature_ci95_lower"].iloc[0], 23.11, places=2)
        self.assertAlmostEqual(res["temperature_ci95_upper"].iloc[0], 23.89, places=2)

        # 99% CI: 23.5 +/- 2.576 * 0.2 = [22.98, 24.02]
        self.assertAlmostEqual(res["temperature_ci99_lower"].iloc[0], 22.98, places=2)
        self.assertAlmostEqual(res["temperature_ci99_upper"].iloc[0], 24.02, places=2)

    def test_humidity_bounding(self):
        # Negative corrected humidity should be clipped to 0
        raw = pd.Series([2.0, 102.0])
        bias = pd.Series([5.0, -5.0])
        sigma = pd.Series([1.0, 1.0])

        res = SensorAutoCorrector.correct_series(
            raw_series=raw,
            bias_series=bias,
            std_series=sigma,
            sensor_name="humidity"
        )

        # 2 - 5 = -3 -> clamped to 0.0
        self.assertEqual(res["humidity_corrected"].iloc[0], 0.0)
        # 102 - (-5) = 107 -> clamped to 100.0
        self.assertEqual(res["humidity_corrected"].iloc[1], 100.0)
        self.assertGreaterEqual(res["humidity_ci95_lower"].iloc[0], 0.0)
        self.assertLessEqual(res["humidity_ci95_upper"].iloc[1], 100.0)


class TestCorrectionAndAlertingPipeline(unittest.TestCase):

    def test_end_to_end_pipeline(self):
        config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=10, inject_anomalies=True, random_seed=42)
        raw_df, _ = generate_synthetic_aws_data(config)

        pipeline = AWSCorrectionAndAlertingPipeline()
        result = pipeline.process(raw_df, station_id="TEST-STATION-01")

        self.assertIsNotNone(result.corrected_dataset_df)
        self.assertIsNotNone(result.actionable_alerts)
        self.assertIsNotNone(result.health_metrics)

        # Verify corrected columns exist
        for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
            self.assertIn(f"{ch}_raw", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_corrected", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_bias_est", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci95_lower", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci95_upper", result.corrected_dataset_df.columns)

        # Verify export functions
        os.makedirs("output/test_export", exist_ok=True)
        csv_file = "output/test_export/test_corrected.csv"
        json_file = "output/test_export/test_corrected.json"
        
        result.export_csv(csv_file)
        result.export_json(json_file)

        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(json_file))
        self.assertGreater(os.path.getsize(csv_file), 100)
        self.assertGreater(os.path.getsize(json_file), 100)

        # Verify dashboard markdown
        self.assertIn("Real-Time AWS Sensor Alert", result.dashboard_markdown)
        self.assertIn("Live Meteorological Telemetry", result.dashboard_markdown)


if __name__ == "__main__":
    unittest.main()
