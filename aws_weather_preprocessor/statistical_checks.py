"""
Statistical quality control and anomaly detection methods for AWS sensor time-series data.

Implements:
1. Missing Value Detection (NaNs, nulls, infinite values, and sentinel codes e.g. -999, -9999)
2. Interquartile Range (IQR) Outlier Detection
3. Rolling / Modified Z-Score Outlier Detection (MAD-based for robustness against diurnal drift)
4. Rate-of-Change / Step Check (Spike Detection)
5. Stuck Sensor / Persistence / Flatline Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class StatisticalThresholds:
    """Configurable thresholds for statistical anomaly detection across sensor channels."""
    # Sentinel missing values
    sentinel_values: List[float] = field(default_factory=lambda: [-999.0, -9999.0, 999.0, 9999.0, -99.0])

    # IQR multiplier (standard boxplot is 1.5, strict is 3.0)
    iqr_multiplier: float = 2.0

    # Rolling Z-score thresholds
    zscore_threshold: float = 3.5
    rolling_window_periods: int = 24  # e.g., 2 hours for 5-min intervals or 24 points

    # Rate of change max allowed step per interval (e.g. per 5-10 min)
    max_step_change: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 5.0,     # °C per time step
        "pressure": 4.0,        # hPa per time step
        "humidity": 25.0,       # % per time step
        "wind_speed": 15.0,     # m/s per time step
    })

    # Stuck sensor window (number of consecutive identical / near-constant readings)
    stuck_window_periods: int = 12  # e.g., 1 hour for 5-min interval
    stuck_variance_epsilon: float = 1e-6


def detect_missing_values(
    series: pd.Series,
    sentinels: Optional[List[float]] = None
) -> pd.Series:
    """
    Detect missing values, including NaNs, Infs, and sentinel missing codes.

    Args:
        series: Time-series sensor data.
        sentinels: List of numeric sentinel codes to treat as missing.

    Returns:
        Boolean Series where True indicates missing.
    """
    if sentinels is None:
        sentinels = [-999.0, -9999.0, 999.0, 9999.0, -99.0]

    is_nan = series.isna() | np.isinf(series)
    is_sentinel = series.isin(sentinels)
    return (is_nan | is_sentinel).rename(f"missing_{series.name}")


def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = 2.0
) -> Tuple[pd.Series, float, float]:
    """
    Detect global outliers using the Interquartile Range (IQR) method.

    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - (multiplier * IQR)
        Upper Bound = Q3 + (multiplier * IQR)

    Args:
        series: Time-series sensor data.
        multiplier: Factor multiplied by IQR (default 2.0).

    Returns:
        Tuple of (boolean Series, lower_bound, upper_bound).
    """
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return pd.Series(False, index=series.index), np.nan, np.nan

    q1 = valid_data.quantile(0.25)
    q3 = valid_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    is_outlier = (series < lower_bound) | (series > upper_bound)
    return is_outlier.fillna(False), float(lower_bound), float(upper_bound)


def detect_rolling_zscore_outliers(
    series: pd.Series,
    window: int = 24,
    threshold: float = 3.5,
    use_robust_mad: bool = True
) -> pd.Series:
    """
    Detect local contextual outliers using a rolling Z-score or rolling MAD (Median Absolute Deviation).
    Rolling statistics effectively account for diurnal weather cycles (diurnal non-stationarity).

    Args:
        series: Time-series sensor data.
        window: Rolling window size (number of observations).
        threshold: Z-score cutoff (typically 3.0 to 4.0).
        use_robust_mad: If True, uses Median and MAD instead of Mean and Standard Deviation.

    Returns:
        Boolean Series where True indicates a rolling statistical outlier.
    """
    min_periods = max(3, window // 4)

    if use_robust_mad:
        # Rolling Median and Rolling MAD
        rolling_median = series.rolling(window=window, center=True, min_periods=min_periods).median()
        rolling_dev = (series - rolling_median).abs()
        rolling_mad = rolling_dev.rolling(window=window, center=True, min_periods=min_periods).median()
        
        # Consistent scale factor for normal distribution: scale ~ 1.4826 * MAD
        scale = 1.4826 * rolling_mad
        # Avoid zero division
        scale = scale.replace(0.0, np.nan)
        zscore = (series - rolling_median).abs() / scale
    else:
        # Standard Rolling Mean and Rolling Std
        rolling_mean = series.rolling(window=window, center=True, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, center=True, min_periods=min_periods).std()
        rolling_std = rolling_std.replace(0.0, np.nan)
        zscore = (series - rolling_mean).abs() / rolling_std

    is_outlier = zscore > threshold
    return is_outlier.fillna(False)


def detect_step_spikes(
    series: pd.Series,
    max_step: float
) -> pd.Series:
    """
    Detect sudden unphysical spikes / rate-of-change jumps between consecutive time steps.

    Formula:
        |x[t] - x[t-1]| > max_step AND |x[t] - x[t+1]| > max_step (with opposite/restoring direction)
        OR single forward jump |x[t] - x[t-1]| > 1.5 * max_step.

    Args:
        series: Time-series sensor data.
        max_step: Maximum plausible change per single sampling interval.

    Returns:
        Boolean Series where True indicates a spike anomaly.
    """
    diff_backward = (series - series.shift(1)).abs()
    diff_forward = (series - series.shift(-1)).abs()
    
    # Transient spike: sudden jump up/down that immediately recovers
    is_transient_spike = (diff_backward > max_step) & (diff_forward > max_step)
    
    # Extreme step jump
    is_extreme_jump = diff_backward > (max_step * 1.75)

    is_spike = is_transient_spike | is_extreme_jump
    return is_spike.fillna(False)


def detect_stuck_sensors(
    series: pd.Series,
    window: int = 12,
    epsilon: float = 1e-6
) -> pd.Series:
    """
    Detect frozen / stuck sensors (flatlines) where readings remain unnaturally constant over a window.

    Args:
        series: Time-series sensor data.
        window: Window size of consecutive observations.
        epsilon: Maximum variance tolerance to be considered stuck.

    Returns:
        Boolean Series where True indicates a stuck sensor reading.
    """
    rolling_var = series.rolling(window=window, min_periods=window).var()
    is_stuck_end = (rolling_var <= epsilon)

    if not is_stuck_end.any():
        return pd.Series(False, index=series.index)

    # Propagate backwards across the window that was stuck
    is_stuck_full = is_stuck_end.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].fillna(False).astype(bool)
    return is_stuck_full


class StatisticalQualityChecker:
    """High-level statistical quality control checker for all AWS sensor channels."""

    def __init__(self, thresholds: Optional[StatisticalThresholds] = None):
        self.thresholds = thresholds or StatisticalThresholds()

    def check_channel(
        self,
        series: pd.Series,
        channel_name: str
    ) -> Dict[str, pd.Series]:
        """
        Run statistical checks on an individual sensor channel.

        Args:
            series: Sensor data time-series.
            channel_name: Name of the channel (e.g. 'temperature', 'pressure', etc.).

        Returns:
            Dictionary of boolean flag Series.
        """
        # 1. Missing values & sentinel codes
        missing = detect_missing_values(series, self.thresholds.sentinel_values)
        
        # Work with sanitized series where sentinels are converted to NaN for statistical calculations
        clean_for_stats = series.copy()
        clean_for_stats[missing] = np.nan

        # 2. IQR Outliers
        iqr_outliers, _, _ = detect_iqr_outliers(
            clean_for_stats,
            multiplier=self.thresholds.iqr_multiplier
        )

        # 3. Rolling Z-Score Outliers
        rolling_z_outliers = detect_rolling_zscore_outliers(
            clean_for_stats,
            window=self.thresholds.rolling_window_periods,
            threshold=self.thresholds.zscore_threshold,
            use_robust_mad=True
        )

        # 4. Step Spikes
        max_step = self.thresholds.max_step_change.get(channel_name, 10.0)
        step_spikes = detect_step_spikes(clean_for_stats, max_step=max_step)

        # 5. Stuck Sensor (Flatline)
        # For wind speed, zero wind is natural (calm), so stuck check should only trigger on non-zero flatlines or very long zeros
        if channel_name == "wind_speed":
            stuck = detect_stuck_sensors(
                clean_for_stats.where(clean_for_stats > 0.1),
                window=self.thresholds.stuck_window_periods * 2,
                epsilon=self.thresholds.stuck_variance_epsilon
            )
        else:
            stuck = detect_stuck_sensors(
                clean_for_stats,
                window=self.thresholds.stuck_window_periods,
                epsilon=self.thresholds.stuck_variance_epsilon
            )

        # Combined statistical anomaly
        combined_stat_anomaly = (iqr_outliers | rolling_z_outliers | step_spikes | stuck) & (~missing)

        return {
            f"flag_{channel_name}_missing": missing,
            f"flag_{channel_name}_iqr_outlier": iqr_outliers,
            f"flag_{channel_name}_zscore_outlier": rolling_z_outliers,
            f"flag_{channel_name}_spike": step_spikes,
            f"flag_{channel_name}_stuck": stuck,
            f"flag_{channel_name}_stat_anomaly": combined_stat_anomaly,
        }

    def check_all(
        self,
        df: pd.DataFrame,
        channels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run statistical checks across all specified sensor columns.

        Args:
            df: DataFrame containing sensor columns.
            channels: List of column names to check. Defaults to df.columns.

        Returns:
            DataFrame containing all statistical flag columns.
        """
        if channels is None:
            channels = [col for col in df.columns if col != "timestamp"]

        all_flags = {}
        for ch in channels:
            if ch in df.columns:
                ch_flags = self.check_channel(df[ch], ch)
                all_flags.update(ch_flags)

        return pd.DataFrame(all_flags, index=df.index)
