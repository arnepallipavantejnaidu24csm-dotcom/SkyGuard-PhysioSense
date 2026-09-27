"""
Unit tests for physics-based consistency checks and thermodynamic calculations.
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


class TestThermodynamicCalculations(unittest.TestCase):

    def test_saturation_vapor_pressure_clausius_clapeyron(self):
        # Known standards: at 0°C -> ~6.11 hPa, at 20°C -> ~23.38 hPa, at 40°C -> ~73.8 hPa
        self.assertAlmostEqual(calculate_saturation_vapor_pressure(0.0), 6.112, places=2)
        self.assertAlmostEqual(calculate_saturation_vapor_pressure(20.0), 23.37, delta=0.2)
        self.assertAlmostEqual(calculate_saturation_vapor_pressure(40.0), 73.8, delta=0.5)

    def test_dew_point_psychrometric_relationship(self):
        # When RH = 100%, Dew Point must equal Temperature
        td_100 = calculate_dew_point(25.0, 100.0)
        self.assertAlmostEqual(td_100, 25.0, places=1)

        # When RH is 50% at 20°C, Dew point ~ 9.3°C
        td_50 = calculate_dew_point(20.0, 50.0)
        self.assertAlmostEqual(td_50, 9.3, delta=0.5)

    def test_actual_vapor_pressure_bounds(self):
        e = calculate_actual_vapor_pressure(20.0, 50.0)
        es = calculate_saturation_vapor_pressure(20.0)
        self.assertAlmostEqual(e, 0.5 * es, places=2)

    def test_absolute_humidity_plausibility(self):
        e = calculate_actual_vapor_pressure(20.0, 50.0)
        ah = calculate_absolute_humidity(20.0, e)
        # At 20°C and 50% RH, absolute humidity is ~ 8.6 g/m^3
        self.assertGreater(ah, 5.0)
        self.assertLess(ah, 15.0)


class TestPhysicsConsistencyRules(unittest.TestCase):

    def test_thermodynamic_violation_detection(self):
        # Case 1: Dew Point > Temperature (RH > 105%)
        # Case 2: Vapor Pressure > Atmospheric Pressure
        # Case 3: Climatological bounds breach
        temps = pd.Series([20.0, 25.0, 15.0, 75.0])
        pressures = pd.Series([1013.25, 1013.25, 10.0, 1013.25])
        rhs = pd.Series([50.0, 120.0, 60.0, 30.0])

        flags = validate_t_p_rh_physics(temps, pressures, rhs)
        self.assertFalse(flags["rh_out_of_bounds"].iloc[0])
        self.assertTrue(flags["rh_out_of_bounds"].iloc[1])
        self.assertTrue(flags["vapor_exceeds_pressure"].iloc[2])
        self.assertTrue(flags["temp_out_of_bounds"].iloc[3])
        self.assertTrue(flags["t_p_rh_inconsistent"].iloc[1])

    def test_wind_speed_boundaries(self):
        winds = pd.Series([5.0, -2.5, 48.0, 115.0])
        flags = validate_wind_speed_bounds(winds)
        self.assertFalse(flags["wind_negative"].iloc[0])
        self.assertTrue(flags["wind_negative"].iloc[1])
        self.assertFalse(flags["wind_exceeds_typical"].iloc[0])
        self.assertTrue(flags["wind_exceeds_typical"].iloc[3])
        self.assertTrue(flags["wind_exceeds_extreme"].iloc[3])


if __name__ == "__main__":
    unittest.main()
