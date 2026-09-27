"""
Main AWS Sensor Data Preprocessing Pipeline.

Orchestrates:
1. Multi-sensor time-series ingestion and timestamp indexing
2. Physics-based consistency checking (T-P-RH relationships, wind bounds)
3. Statistical outlier & missing value identification (IQR, Z-Score, Spikes, Flatlines)
4. Comprehensive Anomaly Flagging
5. Data Cleaning and Imputation (Linear interpolation, forward fill, or NaN masking)
6. Automated Data Quality Reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .physics_checks import (
    PhysicsThresholds,
    PhysicsQualityChecker,
    calculate_dew_point,
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_absolute_humidity,
)
from .statistical_checks import (
    StatisticalThresholds,
    StatisticalQualityChecker,
)
from .quality_reporter import (
    DataQualityReport,
    QualityReportGenerator,
)


@dataclass
class PreprocessingConfig:
    """Configuration options for the AWS Data Preprocessor pipeline."""
    # Column mapping
    timestamp_col: str = "timestamp"
    temp_col: str = "temperature"
    pressure_col: str = "pressure"
    rh_col: str = "humidity"
    wind_col: str = "wind_speed"

    # Physics and statistical settings
    physics_thresholds: PhysicsThresholds = field(default_factory=PhysicsThresholds)
    statistical_thresholds: StatisticalThresholds = field(default_factory=StatisticalThresholds)

    # Imputation strategy: 'interpolate' (time/linear), 'ffill', 'bfill', 'none' (leave as NaN)
    imputation_strategy: str = "interpolate"
    max_impute_gap_periods: int = 6  # Max consecutive missing periods to impute (e.g. 30 mins for 5-min data)
    
    # Append derived thermodynamic parameters to cleaned dataset
    include_derived_thermodynamics: bool = True


@dataclass
class PreprocessingResult:
    """Container for preprocessing results, cleaned datasets, flags, and reports."""
    cleaned_df: pd.DataFrame
    flags_df: pd.DataFrame
    derived_df: pd.DataFrame
    report: DataQualityReport
    raw_df: pd.DataFrame

    def get_full_dataset(self) -> pd.DataFrame:
        """Combine raw data, flags, derived parameters, and clean data into a unified DataFrame."""
        combined = self.raw_df.copy()
        
        # Add clean columns
        for col in self.cleaned_df.columns:
            if col != "timestamp":
                combined[f"{col}_cleaned"] = self.cleaned_df[col]
        
        # Add derived thermodynamic features
        for col in self.derived_df.columns:
            combined[col] = self.derived_df[col]

        # Add flags
        for col in self.flags_df.columns:
            combined[col] = self.flags_df[col]

        return combined


class AWSDataPreprocessor:
    """
    Production-grade preprocessor for Automatic Weather Station (AWS) sensor data.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.physics_checker = PhysicsQualityChecker(thresholds=self.config.physics_thresholds)
        self.stat_checker = StatisticalQualityChecker(thresholds=self.config.statistical_thresholds)

    def process(self, df: pd.DataFrame) -> PreprocessingResult:
        """
        Execute full quality control, anomaly flagging, data cleaning, and reporting pipeline.

        Args:
            df: Input DataFrame containing AWS time-series readings.

        Returns:
            PreprocessingResult containing cleaned dataset, anomaly flags, derived thermodynamics, and report.
        """
        raw_df = df.copy()
        temp_c = self.config.temp_col
        pres_c = self.config.pressure_col
        rh_c = self.config.rh_col
        wind_c = self.config.wind_col
        ts_c = self.config.timestamp_col

        # Ensure datetime format for timestamps if present
        if ts_c in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df[ts_c]):
            raw_df[ts_c] = pd.to_datetime(raw_df[ts_c])

        # 1. Run Physics-based Checks
        physics_flags, derived_df = self.physics_checker.check_all(
            df=raw_df,
            temp_col=temp_c,
            pressure_col=pres_c,
            rh_col=rh_c,
            wind_col=wind_c
        )

        # 2. Run Statistical Checks (IQR, Rolling Z-score, Spikes, Flatlines, Missing/Sentinels)
        channels = [temp_c, pres_c, rh_c, wind_c]
        stat_flags = self.stat_checker.check_all(raw_df, channels=channels)

        # Combine all flags
        flags_df = pd.concat([physics_flags, stat_flags], axis=1)

        # 3. Create Consolidated Per-Sensor and Overall Quality Flags
        for ch in channels:
            if ch in raw_df.columns:
                ch_missing = flags_df.get(f"flag_{ch}_missing", pd.Series(False, index=raw_df.index))
                ch_stat = flags_df.get(f"flag_{ch}_stat_anomaly", pd.Series(False, index=raw_df.index))
                
                ch_physics = pd.Series(False, index=raw_df.index)
                if ch == temp_c:
                    ch_physics = flags_df.get("flag_temp_out_of_bounds", pd.Series(False, index=raw_df.index))
                elif ch == pres_c:
                    ch_physics = flags_df.get("flag_pressure_out_of_bounds", pd.Series(False, index=raw_df.index))
                elif ch == rh_c:
                    ch_physics = (
                        flags_df.get("flag_rh_out_of_bounds", pd.Series(False, index=raw_df.index)) |
                        flags_df.get("flag_dew_point_exceeds_temp", pd.Series(False, index=raw_df.index)) |
                        flags_df.get("flag_vapor_exceeds_pressure", pd.Series(False, index=raw_df.index))
                    )
                elif ch == wind_c:
                    ch_physics = flags_df.get("flag_wind_physics_violation", pd.Series(False, index=raw_df.index))

                # Sensor is invalid if missing, physics violated, or statistical anomaly
                flags_df[f"is_valid_{ch}"] = ~(ch_missing | ch_physics | ch_stat)

        # Overall row-level quality flag
        sensor_valid_cols = [f"is_valid_{ch}" for ch in channels if f"is_valid_{ch}" in flags_df.columns]
        flags_df["is_all_valid"] = flags_df[sensor_valid_cols].all(axis=1)

        # 4. Generate Cleaned Dataset
        cleaned_df = self._clean_and_impute(raw_df, flags_df, channels, ts_c)

        # 5. Generate Data Quality Report
        report = QualityReportGenerator.generate(
            raw_df=raw_df,
            flags_df=flags_df,
            derived_df=derived_df,
            temp_col=temp_c,
            pressure_col=pres_c,
            rh_col=rh_c,
            wind_col=wind_c,
            timestamp_col=ts_c,
        )

        return PreprocessingResult(
            cleaned_df=cleaned_df,
            flags_df=flags_df,
            derived_df=derived_df,
            report=report,
            raw_df=raw_df,
        )

    def _clean_and_impute(
        self,
        raw_df: pd.DataFrame,
        flags_df: pd.DataFrame,
        channels: List[str],
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Clean invalid readings (set to NaN) and optionally impute gaps.
        """
        cleaned_df = pd.DataFrame(index=raw_df.index)
        
        if timestamp_col in raw_df.columns:
            cleaned_df[timestamp_col] = raw_df[timestamp_col]

        for ch in channels:
            if ch not in raw_df.columns:
                continue
            
            # Start with raw series
            series = raw_df[ch].copy()

            # Identify invalid records for this channel
            is_valid_col = f"is_valid_{ch}"
            if is_valid_col in flags_df:
                series[~flags_df[is_valid_col]] = np.nan

            # Apply imputation if configured
            strategy = self.config.imputation_strategy.lower()
            limit = self.config.max_impute_gap_periods

            if strategy == "interpolate":
                # Linear interpolation with max gap limit
                series_imputed = series.interpolate(method="linear", limit=limit, limit_direction="both")
            elif strategy == "ffill":
                series_imputed = series.ffill(limit=limit)
            elif strategy == "bfill":
                series_imputed = series.bfill(limit=limit)
            elif strategy == "none":
                series_imputed = series
            else:
                series_imputed = series.interpolate(method="linear", limit=limit)

            # Round to realistic sensor precision
            if ch in ["temperature", "humidity"]:
                cleaned_df[ch] = series_imputed.round(2)
            elif ch == "pressure":
                cleaned_df[ch] = series_imputed.round(2)
            elif ch == "wind_speed":
                cleaned_df[ch] = series_imputed.clip(lower=0.0).round(2)
            else:
                cleaned_df[ch] = series_imputed

        return cleaned_df
