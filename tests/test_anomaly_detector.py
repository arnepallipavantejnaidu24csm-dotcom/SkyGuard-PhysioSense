"""
Unit tests for the Hybrid Anomaly Detection Module (Isolation Forest, LSTM Autoencoder, Physics Rules).
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.anomaly_detector import (
    PhysicsRuleEngine,
    RuleViolation,
    AWSIsolationForestDetector,
    AWSLSTMAutoencoderDetector,
    AWSHybridAnomalyDetector,
)
from aws_weather_preprocessor.synthetic_data import (
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


class TestPhysicsRuleEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PhysicsRuleEngine()

    def test_high_severity_thermodynamic_violation(self):
        # T=20°C, RH=120% -> Dew point exceeds dry-bulb temperature & RH > 105%
        violations = self.engine.evaluate_row(
            temp_c=20.0,
            pressure_hpa=1013.25,
            rh_percent=120.0,
            wind_speed_mps=5.0
        )
        severities = [v.severity for v in violations]
        alert_levels = [v.alert_level for v in violations]

        self.assertIn("HIGH", severities)
        self.assertIn("CRITICAL", alert_levels)
        # Check recommended action exists
        high_v = next(v for v in violations if v.severity == "HIGH")
        self.assertIn("calibration", high_v.recommended_action.lower())

    def test_medium_severity_sensor_spec_breach(self):
        # Negative wind speed
        violations = self.engine.evaluate_row(
            temp_c=20.0,
            pressure_hpa=1013.25,
            rh_percent=50.0,
            wind_speed_mps=-5.0
        )
        severities = [v.severity for v in violations]
        alert_levels = [v.alert_level for v in violations]

        self.assertIn("MEDIUM", severities)
        self.assertIn("WARNING", alert_levels)
        med_v = next(v for v in violations if v.severity == "MEDIUM")
        self.assertEqual(med_v.rule_id, "SPEC_WIND_NEGATIVE")

    def test_low_severity_statistical_glitch(self):
        # Normal physics values, but statistical spike flag passed
        violations = self.engine.evaluate_row(
            temp_c=20.0,
            pressure_hpa=1013.25,
            rh_percent=50.0,
            wind_speed_mps=5.0,
            stat_flags={"flag_temperature_spike": True}
        )
        severities = [v.severity for v in violations]
        alert_levels = [v.alert_level for v in violations]

        self.assertIn("LOW", severities)
        self.assertIn("INFO", alert_levels)


class TestIsolationForestDetector(unittest.TestCase):

    def test_training_and_confidence_scoring(self):
        # Generate 2 days of normal data
        config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=False)
        normal_df, _ = generate_synthetic_aws_data(config)

        detector = AWSIsolationForestDetector(contamination=0.05, n_estimators=50)
        detector.fit(normal_df)

        # Create test set with extreme outlier
        test_df = normal_df.copy()
        test_df.loc[50, "temperature"] += 35.0  # Massive outlier

        res = detector.predict(test_df)
        self.assertEqual(len(res.is_anomaly), len(test_df))
        self.assertTrue(res.is_anomaly.iloc[50])
        self.assertGreater(res.anomaly_confidence.iloc[50], 0.70)


class TestLSTMAutoencoderDetector(unittest.TestCase):

    def test_lstm_reconstruction_and_detection(self):
        # Generate 2 days of normal data
        config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=False)
        normal_df, _ = generate_synthetic_aws_data(config)

        lstm_detector = AWSLSTMAutoencoderDetector(
            seq_len=6,
            hidden_dim=16,
            latent_dim=8,
            epochs=5,
            batch_size=32,
        )
        lstm_detector.fit(normal_df)

        # Inject sudden jump into test set
        test_df = normal_df.copy()
        test_df.loc[80:85, "pressure"] += 20.0  # Massive step jump

        res = lstm_detector.predict(test_df)
        self.assertIsNotNone(res.reconstruction_error)
        self.assertGreater(res.reconstruction_error.iloc[83], res.threshold)


class TestHybridAnomalyDetector(unittest.TestCase):

    def test_end_to_end_hybrid_detection(self):
        # Generate normal baseline for training
        train_config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=False, random_seed=10)
        train_df, _ = generate_synthetic_aws_data(train_config)

        # Generate test dataset with realistic injected anomalies
        test_config = SyntheticDataConfig(duration_days=2.0, frequency_minutes=5, inject_anomalies=True, random_seed=20)
        test_df, _ = generate_synthetic_aws_data(test_config)

        hybrid = AWSHybridAnomalyDetector(lstm_seq_len=6, lstm_epochs=5)
        hybrid.train(train_df)

        result = hybrid.detect(test_df)

        self.assertIsNotNone(result.summary_df)
        self.assertIsNotNone(result.alerts_df)
        self.assertFalse(result.alerts_df.empty)

        # Verify output columns
        expected_cols = ["timestamp", "is_anomaly", "anomaly_type", "severity", "confidence", "recommended_action"]
        for col in expected_cols:
            self.assertIn(col, result.summary_df.columns)

        # Verify severities present
        severities = set(result.alerts_df["severity"])
        self.assertTrue("CRITICAL" in severities or "WARNING" in severities)

        # Verify confidence in [0, 1]
        self.assertTrue((result.summary_df["confidence"] >= 0.0).all())
        self.assertTrue((result.summary_df["confidence"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
