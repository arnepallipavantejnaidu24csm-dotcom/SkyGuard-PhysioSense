"""
Unit tests for the Sensor Health Scoring and Predictive Maintenance System.
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.health_scoring import (
    SensorHealthScoringSystem,
    SensorProfile,
    SensorHealthMetrics,
    MaintenanceAlert,
)


class TestSensorHealthScoring(unittest.TestCase):

    def setUp(self):
        self.health_system = SensorHealthScoringSystem()

    def test_healthy_sensor_evaluation(self):
        # 1000 clean temperature readings
        n = 1000
        raw_s = pd.Series(20.0 + np.random.normal(0, 0.1, n))
        bias_s = pd.Series(np.zeros(n))
        res_s = pd.Series(np.random.normal(0, 0.05, n))

        metrics = self.health_system.evaluate_sensor_health(
            sensor_name="temperature",
            raw_series=raw_s,
            bias_series=bias_s,
            residual_series=res_s,
            current_date="2026-04-01",
        )

        self.assertEqual(metrics.sensor_name, "temperature")
        self.assertEqual(metrics.completeness_pct, 100.0)
        self.assertGreaterEqual(metrics.health_score, 85.0)
        self.assertFalse(metrics.is_degraded)
        self.assertIn(metrics.health_grade, ["EXCELLENT", "GOOD"])
        self.assertEqual(metrics.maintenance_status, "NORMAL")

    def test_degraded_drifting_sensor(self):
        # Temperature drifting +1.5°C with high residual noise
        n = 1000
        time_days = np.linspace(0, 5.0, n)
        true_bias = 0.3 * time_days  # Drifts from 0 to 1.5°C (exceeds max_allowable_bias of 0.6°C)
        raw_s = pd.Series(20.0 + true_bias + np.random.normal(0, 0.4, n))
        bias_s = pd.Series(true_bias)
        res_s = pd.Series(np.random.normal(0, 0.4, n))

        metrics = self.health_system.evaluate_sensor_health(
            sensor_name="temperature",
            raw_series=raw_s,
            bias_series=bias_s,
            residual_series=res_s,
            current_date="2026-09-01",
        )

        self.assertLess(metrics.health_score, 60.0)
        self.assertTrue(metrics.is_degraded)
        self.assertIn(metrics.health_grade, ["DEGRADED", "CRITICAL"])
        self.assertIn(metrics.maintenance_status, ["MAINTENANCE_DUE", "URGENT_ACTION_REQUIRED"])
        self.assertLess(metrics.predicted_rul_days, 15.0)

    def test_missing_data_packet_loss_penalty(self):
        n = 1000
        raw_s = pd.Series(np.random.normal(20, 1, n))
        # Drop 30% of data
        raw_s.iloc[200:500] = np.nan

        metrics = self.health_system.evaluate_sensor_health(
            sensor_name="temperature",
            raw_series=raw_s,
        )

        self.assertAlmostEqual(metrics.completeness_pct, 70.0, delta=1.0)
        self.assertLess(metrics.completeness_subscore, 70.0)

    def test_maintenance_alerts_generation(self):
        # Create dictionary of metrics
        healthy_metrics = SensorHealthMetrics(
            sensor_name="pressure",
            instrument_model="Vaisala PTB330",
            total_expected=1000,
            total_received=1000,
            completeness_pct=100.0,
            current_bias=0.05,
            drift_rate_per_day=0.001,
            noise_std=0.08,
            noise_variance=0.0064,
            days_since_calibration=60,
            calibration_status="OPTIMAL",
            anomaly_incidents=0,
            completeness_subscore=100.0,
            drift_subscore=95.0,
            noise_subscore=95.0,
            calibration_subscore=90.0,
            health_score=94.5,
            health_grade="EXCELLENT",
            is_degraded=False,
            predicted_rul_days=250.0,
            maintenance_status="NORMAL",
            recommended_action="Normal monitoring",
        )

        critical_metrics = SensorHealthMetrics(
            sensor_name="temperature",
            instrument_model="Pt100 RTD",
            total_expected=1000,
            total_received=850,
            completeness_pct=85.0,
            current_bias=1.45,
            drift_rate_per_day=0.25,
            noise_std=0.65,
            noise_variance=0.4225,
            days_since_calibration=370,
            calibration_status="EXPIRED",
            anomaly_incidents=5,
            completeness_subscore=70.0,
            drift_subscore=10.0,
            noise_subscore=20.0,
            calibration_subscore=30.0,
            health_score=32.0,
            health_grade="CRITICAL",
            is_degraded=True,
            predicted_rul_days=0.0,
            maintenance_status="URGENT_ACTION_REQUIRED",
            recommended_action="URGENT: Replace sensor",
        )

        alerts = self.health_system.generate_maintenance_alerts({
            "pressure": healthy_metrics,
            "temperature": critical_metrics,
        })

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].sensor_name, "temperature")
        self.assertEqual(alerts[0].urgency, "URGENT")

        # Test dashboard markdown generation
        dashboard_md = self.health_system.generate_dashboard_markdown({
            "pressure": healthy_metrics,
            "temperature": critical_metrics,
        })
        self.assertIn("AWS Sensor Health & Maintenance Dashboard", dashboard_md)
        self.assertIn("Pt100 RTD", dashboard_md)


if __name__ == "__main__":
    unittest.main()
