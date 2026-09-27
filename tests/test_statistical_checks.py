"""
Unit tests for statistical quality control checks (IQR, Rolling Z-Score, Spikes, Flatlines, Sentinels).
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.statistical_checks import (
    detect_missing_values,
    detect_iqr_outliers,
    detect_rolling_zscore_outliers,
    detect_step_spikes,
    detect_stuck_sensors,
    StatisticalQualityChecker,
    StatisticalThresholds,
)


class TestStatisticalQC(unittest.TestCase):

    def test_sentinel_and_nan_detection(self):
        s = pd.Series([15.0, -999.0, np.nan, 22.0, -9999.0, 999.0, np.inf])
        missing = detect_missing_values(s)
        self.assertEqual(list(missing), [False, True, True, False, True, True, True])

    def test_iqr_outlier_detection(self):
        np.random.seed(42)
        normal_data = np.random.normal(loc=20.0, scale=1.5, size=150)
        normal_data[50] = 75.0   # Extreme high outlier
        normal_data[100] = -35.0 # Extreme low outlier
        s = pd.Series(normal_data)

        is_outlier, lower, upper = detect_iqr_outliers(s, multiplier=2.0)
        self.assertTrue(is_outlier.iloc[50])
        self.assertTrue(is_outlier.iloc[100])
        self.assertFalse(is_outlier.iloc[0])

    def test_rolling_robust_zscore(self):
        # Sine diurnal pattern with a transient anomaly
        x = np.linspace(0, 4*np.pi, 200)
        y = 20.0 + 8.0 * np.sin(x) + np.random.normal(0, 0.2, 200)
        y[100] += 12.0  # Statistical outlier on top of diurnal curve

        s = pd.Series(y)
        outliers = detect_rolling_zscore_outliers(s, window=24, threshold=3.5, use_robust_mad=True)
        self.assertTrue(outliers.iloc[100])
        self.assertFalse(outliers.iloc[20])

    def test_step_spike_detection(self):
        # Transient single-step glitch that returns to baseline
        s = pd.Series([20.0, 20.2, 20.1, 35.0, 20.3, 20.2])
        spikes = detect_step_spikes(s, max_step=5.0)
        self.assertTrue(spikes.iloc[3])
        self.assertFalse(spikes.iloc[1])

    def test_stuck_sensor_flatline(self):
        # 10 identical values
        s = pd.Series([20.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0])
        stuck = detect_stuck_sensors(s, window=8)
        self.assertTrue(stuck.iloc[5])
        self.assertTrue(stuck.iloc[9])
        self.assertFalse(stuck.iloc[0])


if __name__ == "__main__":
    unittest.main()
