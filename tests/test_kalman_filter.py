"""
Unit tests for the Atmospheric State Kalman Filter and Sensor Drift Tracker.
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.kalman_tracker import (
    AtmosphericStateKalmanFilter,
    KalmanFilterConfig,
    KalmanOutput,
)
from aws_weather_preprocessor.drift_scenarios import (
    generate_drift_scenario,
    DriftExperimentResult,
)


class TestKalmanFilterCore(unittest.TestCase):

    def setUp(self):
        self.config = KalmanFilterConfig(
            channels=["temperature", "pressure", "humidity", "wind_speed"]
        )
        self.kf = AtmosphericStateKalmanFilter(self.config)

    def test_initialization_and_dimensions(self):
        obs = {"temperature": 20.0, "pressure": 1013.25, "humidity": 60.0, "wind_speed": 5.0}
        self.kf.initialize_state(obs)

        # 4 channels * 2 (value + rate) + 4 biases = 12 state dimensions
        self.assertEqual(self.kf.dim_x, 12)
        self.assertEqual(self.kf.x.shape, (12, 1))
        self.assertEqual(self.kf.P.shape, (12, 12))

        # Check initial values
        self.assertAlmostEqual(self.kf.x[0, 0], 20.0)      # Temp true
        self.assertAlmostEqual(self.kf.x[1, 0], 0.0)       # Temp rate
        self.assertAlmostEqual(self.kf.x[8, 0], 0.0)       # Temp bias

    def test_transition_and_observation_matrices(self):
        dt = 5.0
        F = self.kf.build_transition_matrix(dt)
        H = self.kf.build_observation_matrix()
        Q = self.kf.build_process_noise_matrix(dt)
        R = self.kf.build_measurement_noise_matrix()

        self.assertEqual(F.shape, (12, 12))
        self.assertEqual(H.shape, (4, 12))
        self.assertEqual(Q.shape, (12, 12))
        self.assertEqual(R.shape, (4, 4))

        # Check F kinematics: x_next = x + dt * v
        self.assertEqual(F[0, 1], dt)
        self.assertEqual(F[2, 3], dt)

        # Check H mapping: z = x_true + bias
        # For channel 0 (temp): col 0 is temp_true, col 8 is temp_bias
        self.assertEqual(H[0, 0], 1.0)
        self.assertEqual(H[0, 8], 1.0)
        self.assertEqual(H[0, 1], 0.0)

    def test_prediction_and_update_cycle(self):
        obs = {"temperature": 25.0, "pressure": 1010.0, "humidity": 50.0, "wind_speed": 4.0}
        self.kf.initialize_state(obs)

        # Predict
        x_pred, P_pred = self.kf.predict(dt=5.0)
        self.assertTrue(np.all(np.diag(P_pred) > 0))

        # Update
        new_obs = {"temperature": 25.2, "pressure": 1009.8, "humidity": 51.0, "wind_speed": 4.2}
        res = self.kf.update(new_obs)

        self.assertIn("innovations", res)
        self.assertIn("post_fit_residuals", res)
        self.assertIn("nis", res)
        self.assertTrue(res["measurement_updated"])
        
        # Post-fit residuals should be smaller than innovations
        self.assertLess(abs(res["post_fit_residuals"]["temperature"]), abs(res["innovations"]["temperature"]) + 0.1)

    def test_missing_observation_handling(self):
        obs = {"temperature": 22.0, "pressure": 1015.0, "humidity": 55.0, "wind_speed": 3.0}
        self.kf.initialize_state(obs)
        self.kf.predict(dt=5.0)

        # Missing pressure and humidity (NaNs)
        partial_obs = {"temperature": 22.1, "pressure": np.nan, "humidity": np.nan, "wind_speed": 3.1}
        res = self.kf.update(partial_obs)

        self.assertTrue(res["measurement_updated"])
        self.assertFalse(np.isnan(res["innovations"]["temperature"]))
        self.assertTrue(np.isnan(res["innovations"]["pressure"]))


class TestDriftTrackingScenarios(unittest.TestCase):

    def test_linear_drift_tracking(self):
        # Generate 4-day synthetic experiment with linear drift
        exp = generate_drift_scenario(
            scenario_type="linear_ramp",
            duration_days=4.0,
            frequency_minutes=5,
            random_seed=42
        )

        kf = AtmosphericStateKalmanFilter()
        output = kf.filter_dataframe(
            df=exp.observed_drift_df,
            reference_df=exp.reference_checks_df
        )

        self.assertIsNotNone(output.state_estimates_df)
        self.assertIsNotNone(output.bias_estimates_df)
        self.assertIsNotNone(output.residuals_df)

        # Verify estimated true temperature tracks close to ground truth
        final_idx = len(exp.true_states_df) - 1
        true_temp_end = exp.true_states_df["temperature"].iloc[final_idx]
        obs_temp_end = exp.observed_drift_df["temperature"].iloc[final_idx]
        est_temp_end = output.state_estimates_df["temperature_true"].iloc[final_idx]

        # The estimated true temperature should be significantly closer to ground truth than raw observed
        err_raw = abs(obs_temp_end - true_temp_end)
        err_est = abs(est_temp_end - true_temp_end)
        self.assertLess(err_est, err_raw)

    def test_step_jump_tracking(self):
        exp = generate_drift_scenario(
            scenario_type="step_jump",
            duration_days=4.0,
            frequency_minutes=5,
            random_seed=42
        )

        kf = AtmosphericStateKalmanFilter()
        output = kf.filter_dataframe(
            df=exp.observed_drift_df,
            reference_df=exp.reference_checks_df
        )

        # Verify residuals exist and are finite
        self.assertFalse(output.residuals_df["temperature_post_fit_residual"].isna().all())
        self.assertLess(abs(output.summary_metrics["temperature_residual_mean"]), 1.5)


if __name__ == "__main__":
    unittest.main()
