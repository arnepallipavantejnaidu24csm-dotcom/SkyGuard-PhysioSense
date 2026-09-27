"""
End-to-End Integration Tests connecting all modules:
Raw Telemetry -> Physics QC -> Kalman Tracking -> Hybrid Anomaly Detection -> Health Scoring -> Auto-Correction -> Alert Generation.
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor import (
    AWSDataPreprocessor,
    AtmosphericStateKalmanFilter,
    AWSHybridAnomalyDetector,
    SensorHealthScoringSystem,
    AWSCorrectionAndAlertingPipeline,
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


class TestEndToEndPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # 3-day test data with anomalies
        config = SyntheticDataConfig(duration_days=3.0, frequency_minutes=5, inject_anomalies=True, random_seed=42)
        self.raw_df, self.gt_df = generate_synthetic_aws_data(config)

    def test_full_integrated_pipeline_execution(self):
        pipeline = AWSCorrectionAndAlertingPipeline()
        result = pipeline.process(self.raw_df, station_id="AWS-INTEGRATION-TEST-01")

        # 1. Verify Corrected Dataset
        self.assertIsNotNone(result.corrected_dataset_df)
        self.assertEqual(len(result.corrected_dataset_df), len(self.raw_df))
        
        # Check presence of raw, bias, corrected, 95% and 99% CI columns
        for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
            self.assertIn(f"{ch}_raw", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_corrected", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_bias_est", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci95_lower", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci95_upper", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci99_lower", result.corrected_dataset_df.columns)
            self.assertIn(f"{ch}_ci99_upper", result.corrected_dataset_df.columns)

        # 2. Verify Actionable Alerts
        self.assertIsInstance(result.actionable_alerts, list)
        if result.actionable_alerts:
            a0 = result.actionable_alerts[0]
            self.assertTrue(hasattr(a0, "alert_id"))
            self.assertTrue(hasattr(a0, "severity"))
            self.assertTrue(hasattr(a0, "recommended_action"))
            self.assertTrue(hasattr(a0, "historical_context"))

        # 3. Verify Health Scoring
        self.assertIn("temperature", result.health_metrics)
        self.assertIn("pressure", result.health_metrics)
        self.assertIn("humidity", result.health_metrics)
        self.assertIn("wind_speed", result.health_metrics)

        for s_name, hm in result.health_metrics.items():
            self.assertGreaterEqual(hm.health_score, 0.0)
            self.assertLessEqual(hm.health_score, 100.0)
            self.assertGreaterEqual(hm.predicted_rul_days, 0.0)

        # 4. Verify Dashboard Markdown Generation
        self.assertIn("Real-Time AWS Sensor Alert", result.dashboard_markdown)
        self.assertIn("AWS-INTEGRATION-TEST-01", result.dashboard_markdown)


if __name__ == "__main__":
    unittest.main()
