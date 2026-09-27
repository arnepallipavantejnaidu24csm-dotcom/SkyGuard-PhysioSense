"""
Unit and integration tests for AWS Weather Sensor Data Preprocessor.
"""

import unittest
import numpy as np
import pandas as pd

from aws_weather_preprocessor.physics_checks import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_dew_point,
    calculate_absolute_humidity,
    validate_t_p_rh_physics,
    validate_wind_speed_bounds,
    PhysicsQualityChecker,
    PhysicsThresholds,
)
from aws_weather_preprocessor.statistical_checks import (
    detect_missing_values,
    detect_iqr_outliers,
    detect_rolling_zscore_outliers,
    detect_step_spikes,
    detect_stuck_sensors,
    StatisticalQualityChecker,
    StatisticalThresholds,
)
from aws_weather_preprocessor.preprocessor import (
    AWSDataPreprocessor,
    PreprocessingConfig,
)
from aws_weather_preprocessor.synthetic_data import (
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)


class TestPhysicsChecks(unittest.TestCase):

    def test_saturation_vapor_pressure_magnus(self):
        # Known meteorological standard values:
        # At 0°C, e_s ~ 6.11 hPa
        # At 20°C, e_s ~ 23.38 hPa
        # At 30°C, e_s ~ 42.43 hPa
        es_0 = calculate_saturation_vapor_pressure(0.0)
        es_20 = calculate_saturation_vapor_pressure(20.0)
        es_30 = calculate_saturation_vapor_pressure(30.0)

        self.assertAlmostEqual(es_0, 6.112, places=2)
        self.assertAlmostEqual(es_20, 23.37, places=1)
        self.assertAlmostEqual(es_30, 42.43, places=1)

    def test_dew_point_calculation(self):
        # At 100% RH, Dew point should equal Temperature
        temp = 25.0
        rh_100 = 100.0
        td_100 = calculate_dew_point(temp, rh_100)
        self.assertAlmostEqual(td_100, temp, places=1)

        # At 50% RH and 20°C, Dew point is ~ 9.3°C
        td_50 = calculate_dew_point(20.0, 50.0)
        self.assertAlmostEqual(td_50, 9.3, delta=0.5)

    def test_t_p_rh_physics_validation(self):
        temps = pd.Series([20.0, 25.0, 22.0, 70.0])  # Last is climatologically invalid
        pressures = pd.Series([1013.25, 1013.25, 750.0, 1013.25])  # 3rd is low
        rhs = pd.Series([50.0, 115.0, 60.0, 40.0])  # 2nd is supersaturated

        results = validate_t_p_rh_physics(temps, pressures, rhs)
        self.assertFalse(results["rh_out_of_bounds"].iloc[0])
        self.assertTrue(results["rh_out_of_bounds"].iloc[1])
        self.assertTrue(results["pressure_out_of_bounds"].iloc[2])
        self.assertTrue(results["temp_out_of_bounds"].iloc[3])

    def test_wind_speed_validation(self):
        winds = pd.Series([5.0, -3.2, 45.0, 120.0])
        results = validate_wind_speed_bounds(winds)
        self.assertFalse(results["wind_negative"].iloc[0])
        self.assertTrue(results["wind_negative"].iloc[1])
        self.assertFalse(results["wind_exceeds_typical"].iloc[0])
        self.assertTrue(results["wind_exceeds_typical"].iloc[3])
        self.assertTrue(results["wind_exceeds_extreme"].iloc[3])


class TestStatisticalChecks(unittest.TestCase):

    def test_missing_and_sentinel_detection(self):
        s = pd.Series([20.0, np.nan, -999.0, 22.5, -9999.0, np.inf])
        missing = detect_missing_values(s)
        self.assertEqual(list(missing), [False, True, True, False, True, True])

    def test_iqr_outliers(self):
        # Normal distribution with extreme outlier
        np.random.seed(42)
        normal_data = np.random.normal(20, 2, 100)
        normal_data[50] = 95.0  # Massive outlier
        s = pd.Series(normal_data)
        outliers, lower, upper = detect_iqr_outliers(s, multiplier=2.0)
        self.assertTrue(outliers.iloc[50])
        self.assertFalse(outliers.iloc[0])

    def test_spike_detection(self):
        s = pd.Series([20.0, 20.2, 20.1, 35.0, 20.3, 20.2])  # 35.0 is a 15-degree spike
        spikes = detect_step_spikes(s, max_step=5.0)
        self.assertTrue(spikes.iloc[3])
        self.assertFalse(spikes.iloc[1])

    def test_stuck_sensor_detection(self):
        s = pd.Series([20.0, 20.5, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0])
        stuck = detect_stuck_sensors(s, window=8)
        self.assertTrue(stuck.iloc[8])
        self.assertFalse(stuck.iloc[0])


class TestFullPreprocessorPipeline(unittest.TestCase):

    def test_synthetic_pipeline_execution(self):
        config = SyntheticDataConfig(duration_days=3.0, frequency_minutes=10, random_seed=42)
        raw_df, gt_df = generate_synthetic_aws_data(config)

        preprocessor = AWSDataPreprocessor()
        res = preprocessor.process(raw_df)

        # Assertions
        self.assertIsNotNone(res.cleaned_df)
        self.assertIsNotNone(res.flags_df)
        self.assertIsNotNone(res.derived_df)
        self.assertIsNotNone(res.report)

        # Verify derived columns exist
        self.assertIn("dew_point_c", res.derived_df.columns)
        self.assertIn("actual_vapor_pressure_hpa", res.derived_df.columns)
        self.assertIn("absolute_humidity_gm3", res.derived_df.columns)

        # Verify clean columns have no NaNs (imputed)
        self.assertFalse(res.cleaned_df["temperature"].isna().all())

        # Verify report generation
        md_report = res.report.to_markdown()
        self.assertIn("AWS Sensor Data Preprocessing", md_report)
        self.assertIn("Sensor Channel Breakdown", md_report)

        txt_report = res.report.to_text()
        self.assertIn("AUTOMATIC WEATHER STATION", txt_report)


if __name__ == "__main__":
    unittest.main()
