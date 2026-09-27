"""
Data Quality Report Generator for AWS Weather Sensor Data.

Generates:
1. Overall summary statistics (completeness, validity, flag counts)
2. Per-sensor quality breakdown (missing, physics violations, statistical outliers)
3. Detailed anomaly index log with timestamps, observed values, and violation reasons
4. Formatted Markdown, Plaintext, and Dictionary/JSON exports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AnomalyRecord:
    """Individual anomaly incident record."""
    timestamp: Any
    sensor: str
    observed_value: Any
    anomaly_category: str  # 'MISSING', 'PHYSICS_VIOLATION', 'STATISTICAL_OUTLIER', 'SPIKE', 'STUCK'
    rule_violated: str
    description: str


class DataQualityReport:
    """Encapsulates data quality metrics, per-channel summaries, and anomaly logs."""

    def __init__(
        self,
        total_records: int,
        time_range: Tuple[Any, Any],
        overall_valid_percent: float,
        overall_missing_percent: float,
        overall_anomaly_percent: float,
        channel_summaries: pd.DataFrame,
        anomalies_log: List[AnomalyRecord],
        physics_violation_counts: Dict[str, int],
        statistical_anomaly_counts: Dict[str, int],
    ):
        self.total_records = total_records
        self.time_range = time_range
        self.overall_valid_percent = overall_valid_percent
        self.overall_missing_percent = overall_missing_percent
        self.overall_anomaly_percent = overall_anomaly_percent
        self.channel_summaries = channel_summaries
        self.anomalies_log = anomalies_log
        self.physics_violation_counts = physics_violation_counts
        self.statistical_anomaly_counts = statistical_anomaly_counts

    def get_anomalies_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return anomalies log as a pandas DataFrame."""
        if not self.anomalies_log:
            return pd.DataFrame(columns=["timestamp", "sensor", "observed_value", "anomaly_category", "rule_violated", "description"])
        
        records = [asdict(a) for a in self.anomalies_log]
        df = pd.DataFrame(records)
        if limit is not None:
            return df.head(limit)
        return df

    def to_markdown(self) -> str:
        """Render the data quality report as a comprehensive Markdown document."""
        md = []
        md.append("# 🌤️ AWS Sensor Data Preprocessing & Quality Report")
        md.append("")
        md.append("## 1. Executive Summary")
        md.append(f"- **Total Records Analyzed**: `{self.total_records:,}` readings")
        md.append(f"- **Time Coverage**: `{self.time_range[0]}` to `{self.time_range[1]}`")
        md.append(f"- **Overall Clean Data Yield**: **{self.overall_valid_percent:.2f}%**")
        md.append(f"- **Total Missing Rate**: **{self.overall_missing_percent:.2f}%**")
        md.append(f"- **Total Anomaly Rate**: **{self.overall_anomaly_percent:.2f}%** (Physics + Statistical)")
        md.append("")

        md.append("## 2. Sensor Channel Breakdown")
        md.append(self.channel_summaries.to_markdown(index=True))
        md.append("")

        md.append("## 3. Physics-Based Consistency Violations")
        if any(cnt > 0 for cnt in self.physics_violation_counts.values()):
            md.append("| Physics Rule Checked | Violations Count | Description |")
            md.append("|---|---|---|")
            for rule, cnt in self.physics_violation_counts.items():
                if cnt > 0:
                    md.append(f"| `{rule}` | **{cnt:,}** | Thermodynamic / physical limit breach |")
        else:
            md.append("✅ *No physics-based consistency violations detected.*")
        md.append("")

        md.append("## 4. Statistical Anomaly & Outlier Breakdown")
        if any(cnt > 0 for cnt in self.statistical_anomaly_counts.values()):
            md.append("| Anomaly Type | Occurrences | Characteristics |")
            md.append("|---|---|---|")
            for cat, cnt in self.statistical_anomaly_counts.items():
                if cnt > 0:
                    md.append(f"| `{cat}` | **{cnt:,}** | Detected via IQR / Z-Score / Step / Flatline |")
        else:
            md.append("✅ *No statistical outliers detected.*")
        md.append("")

        md.append("## 5. Sample Detected Anomaly Locations (Top 25)")
        anom_df = self.get_anomalies_df(limit=25)
        if not anom_df.empty:
            md.append(anom_df.to_markdown(index=False))
            if len(self.anomalies_log) > 25:
                md.append(f"\n*(Showing first 25 of {len(self.anomalies_log):,} total anomaly instances)*")
        else:
            md.append("✅ *Clean dataset with zero anomaly locations to report.*")
        md.append("")

        return "\n".join(md)

    def to_text(self) -> str:
        """Render plain text console output."""
        lines = []
        lines.append("=" * 70)
        lines.append("           AUTOMATIC WEATHER STATION (AWS) QUALITY REPORT")
        lines.append("=" * 70)
        lines.append(f" Total Records:     {self.total_records}")
        lines.append(f" Time Range:        {self.time_range[0]} -> {self.time_range[1]}")
        lines.append(f" Clean Data Yield:  {self.overall_valid_percent:.2f}%")
        lines.append(f" Missing Rate:      {self.overall_missing_percent:.2f}%")
        lines.append(f" Total Anomalies:   {len(self.anomalies_log)} incidents ({self.overall_anomaly_percent:.2f}%)")
        lines.append("-" * 70)
        lines.append(" CHANNEL SUMMARY:")
        lines.append(self.channel_summaries.to_string())
        lines.append("-" * 70)
        lines.append(" SAMPLE ANOMALY LOCATIONS:")
        sample_df = self.get_anomalies_df(limit=10)
        if not sample_df.empty:
            lines.append(sample_df.to_string(index=False))
        else:
            lines.append(" No anomalies found.")
        lines.append("=" * 70)
        return "\n".join(lines)


class QualityReportGenerator:
    """Builds DataQualityReport instances from preprocessing flag tables and raw input data."""

    @staticmethod
    def generate(
        raw_df: pd.DataFrame,
        flags_df: pd.DataFrame,
        derived_df: Optional[pd.DataFrame] = None,
        temp_col: str = "temperature",
        pressure_col: str = "pressure",
        rh_col: str = "humidity",
        wind_col: str = "wind_speed",
        timestamp_col: str = "timestamp",
    ) -> DataQualityReport:
        """
        Generate a comprehensive DataQualityReport.

        Args:
            raw_df: Original input DataFrame.
            flags_df: Boolean flags generated by physics and statistical checks.
            derived_df: Derived thermodynamic variables (dew point, vapor pressure, etc.).
            temp_col, pressure_col, rh_col, wind_col, timestamp_col: Column mappings.
        """
        total_records = len(raw_df)
        
        # Time range extraction
        if timestamp_col in raw_df.columns:
            time_start = raw_df[timestamp_col].iloc[0] if total_records > 0 else "N/A"
            time_end = raw_df[timestamp_col].iloc[-1] if total_records > 0 else "N/A"
            timestamps = raw_df[timestamp_col]
        else:
            time_start = raw_df.index[0] if total_records > 0 else "N/A"
            time_end = raw_df.index[-1] if total_records > 0 else "N/A"
            timestamps = raw_df.index.to_series()

        channels = [temp_col, pressure_col, rh_col, wind_col]
        channel_data = []

        # Build per-channel summary table
        for ch in channels:
            if ch not in raw_df.columns:
                continue
            
            # Missing count
            missing_flag_col = f"flag_{ch}_missing"
            missing_count = flags_df[missing_flag_col].sum() if missing_flag_col in flags_df else raw_df[ch].isna().sum()
            missing_pct = (missing_count / total_records * 100) if total_records > 0 else 0.0

            # Physics violations
            physics_count = 0
            if ch == temp_col:
                physics_count = flags_df.get("flag_temp_out_of_bounds", pd.Series(0, index=raw_df.index)).sum()
            elif ch == pressure_col:
                physics_count = flags_df.get("flag_pressure_out_of_bounds", pd.Series(0, index=raw_df.index)).sum()
            elif ch == rh_col:
                physics_count = (
                    flags_df.get("flag_rh_out_of_bounds", pd.Series(0, index=raw_df.index)) |
                    flags_df.get("flag_dew_point_exceeds_temp", pd.Series(0, index=raw_df.index))
                ).sum()
            elif ch == wind_col:
                physics_count = (
                    flags_df.get("flag_wind_negative", pd.Series(0, index=raw_df.index)) |
                    flags_df.get("flag_wind_exceeds_extreme", pd.Series(0, index=raw_df.index))
                ).sum()

            # Statistical outliers
            stat_iqr = flags_df.get(f"flag_{ch}_iqr_outlier", pd.Series(0, index=raw_df.index)).sum()
            stat_zscore = flags_df.get(f"flag_{ch}_zscore_outlier", pd.Series(0, index=raw_df.index)).sum()
            stat_spike = flags_df.get(f"flag_{ch}_spike", pd.Series(0, index=raw_df.index)).sum()
            stat_stuck = flags_df.get(f"flag_{ch}_stuck", pd.Series(0, index=raw_df.index)).sum()
            stat_total = flags_df.get(f"flag_{ch}_stat_anomaly", pd.Series(0, index=raw_df.index)).sum()

            # Clean / valid readings
            invalid_ch_mask = (
                flags_df.get(f"flag_{ch}_missing", pd.Series(False, index=raw_df.index)) |
                (flags_df.get(f"flag_{ch}_stat_anomaly", pd.Series(False, index=raw_df.index)))
            )
            if ch == temp_col:
                invalid_ch_mask |= flags_df.get("flag_temp_out_of_bounds", pd.Series(False, index=raw_df.index))
            elif ch == pressure_col:
                invalid_ch_mask |= flags_df.get("flag_pressure_out_of_bounds", pd.Series(False, index=raw_df.index))
            elif ch == rh_col:
                invalid_ch_mask |= flags_df.get("flag_rh_out_of_bounds", pd.Series(False, index=raw_df.index))
                invalid_ch_mask |= flags_df.get("flag_dew_point_exceeds_temp", pd.Series(False, index=raw_df.index))
            elif ch == wind_col:
                invalid_ch_mask |= flags_df.get("flag_wind_physics_violation", pd.Series(False, index=raw_df.index))

            valid_count = total_records - invalid_ch_mask.sum()
            valid_pct = (valid_count / total_records * 100) if total_records > 0 else 0.0

            channel_data.append({
                "Sensor": ch.capitalize(),
                "Valid %": f"{valid_pct:.1f}%",
                "Missing": missing_count,
                "Physics Violations": physics_count,
                "IQR Outliers": stat_iqr,
                "Z-Score Outliers": stat_zscore,
                "Spikes": stat_spike,
                "Stuck / Frozen": stat_stuck,
            })

        channel_summaries_df = pd.DataFrame(channel_data).set_index("Sensor")

        # Compile granular anomaly log with exact timestamp locations
        anomalies_log: List[AnomalyRecord] = []

        # 1. Missing values
        for ch in channels:
            m_col = f"flag_{ch}_missing"
            if m_col in flags_df:
                for idx in flags_df[flags_df[m_col]].index:
                    anomalies_log.append(AnomalyRecord(
                        timestamp=timestamps.loc[idx],
                        sensor=ch,
                        observed_value=raw_df.loc[idx, ch] if ch in raw_df else np.nan,
                        anomaly_category="MISSING",
                        rule_violated=f"{ch}_missing_or_sentinel",
                        description=f"Missing reading or sentinel value detected in {ch}",
                    ))

        # 2. Physics violations
        if "flag_dew_point_exceeds_temp" in flags_df:
            for idx in flags_df[flags_df["flag_dew_point_exceeds_temp"]].index:
                t_val = raw_df.loc[idx, temp_col]
                rh_val = raw_df.loc[idx, rh_col]
                td_val = derived_df.loc[idx, "dew_point_c"] if derived_df is not None and "dew_point_c" in derived_df else "N/A"
                td_str = f"{td_val:.2f}°C" if isinstance(td_val, (int, float)) else str(td_val)
                anomalies_log.append(AnomalyRecord(
                    timestamp=timestamps.loc[idx],
                    sensor=f"{temp_col}+{rh_col}",
                    observed_value=f"T={t_val}°C, RH={rh_val}%",
                    anomaly_category="PHYSICS_VIOLATION",
                    rule_violated="dew_point_le_temp",
                    description=f"Thermodynamic violation: Dew Point ({td_str}) exceeds Dry Bulb Temperature ({t_val}°C)",
                ))

        if "flag_vapor_exceeds_pressure" in flags_df:
            for idx in flags_df[flags_df["flag_vapor_exceeds_pressure"]].index:
                anomalies_log.append(AnomalyRecord(
                    timestamp=timestamps.loc[idx],
                    sensor=f"{pressure_col}+{rh_col}",
                    observed_value=raw_df.loc[idx, pressure_col],
                    anomaly_category="PHYSICS_VIOLATION",
                    rule_violated="vapor_pressure_le_ambient",
                    description=f"Water vapor pressure exceeds total atmospheric ambient pressure",
                ))

        if "flag_rh_out_of_bounds" in flags_df:
            for idx in flags_df[flags_df["flag_rh_out_of_bounds"]].index:
                rh_val = raw_df.loc[idx, rh_col]
                anomalies_log.append(AnomalyRecord(
                    timestamp=timestamps.loc[idx],
                    sensor=rh_col,
                    observed_value=rh_val,
                    anomaly_category="PHYSICS_VIOLATION",
                    rule_violated="rh_physical_bounds",
                    description=f"Relative Humidity ({rh_val}%) out of physical bounds [0%, 102%]",
                ))

        if "flag_wind_negative" in flags_df:
            for idx in flags_df[flags_df["flag_wind_negative"]].index:
                ws_val = raw_df.loc[idx, wind_col]
                anomalies_log.append(AnomalyRecord(
                    timestamp=timestamps.loc[idx],
                    sensor=wind_col,
                    observed_value=ws_val,
                    anomaly_category="PHYSICS_VIOLATION",
                    rule_violated="wind_speed_non_negative",
                    description=f"Physically impossible negative wind speed ({ws_val} m/s)",
                ))

        if "flag_wind_exceeds_typical" in flags_df:
            for idx in flags_df[flags_df["flag_wind_exceeds_typical"]].index:
                ws_val = raw_df.loc[idx, wind_col]
                anomalies_log.append(AnomalyRecord(
                    timestamp=timestamps.loc[idx],
                    sensor=wind_col,
                    observed_value=ws_val,
                    anomaly_category="PHYSICS_WARNING" if ws_val <= 90.0 else "PHYSICS_VIOLATION",
                    rule_violated="wind_speed_max_bounds",
                    description=f"Wind speed ({ws_val} m/s) exceeds typical 50 m/s bound",
                ))

        # 3. Statistical anomalies (Spikes, Stuck, Z-Score)
        for ch in channels:
            spike_col = f"flag_{ch}_spike"
            if spike_col in flags_df:
                for idx in flags_df[flags_df[spike_col]].index:
                    val = raw_df.loc[idx, ch]
                    anomalies_log.append(AnomalyRecord(
                        timestamp=timestamps.loc[idx],
                        sensor=ch,
                        observed_value=val,
                        anomaly_category="SPIKE",
                        rule_violated=f"{ch}_rate_of_change_spike",
                        description=f"Unphysical abrupt step change / transient spike in {ch} ({val})",
                    ))

            stuck_col = f"flag_{ch}_stuck"
            if stuck_col in flags_df:
                for idx in flags_df[flags_df[stuck_col]].index:
                    val = raw_df.loc[idx, ch]
                    anomalies_log.append(AnomalyRecord(
                        timestamp=timestamps.loc[idx],
                        sensor=ch,
                        observed_value=val,
                        anomaly_category="STUCK",
                        rule_violated=f"{ch}_sensor_flatline",
                        description=f"Frozen / stuck sensor: zero variance in {ch} over time window ({val})",
                    ))

        # Sort anomaly records chronologically
        try:
            anomalies_log.sort(key=lambda x: str(x.timestamp))
        except Exception:
            pass

        # Counts summaries
        physics_counts = {
            "T-P-RH Thermodynamic Violations": flags_df.get("flag_t_p_rh_physics_violation", pd.Series(0)).sum(),
            "Dew Point > Dry Bulb Temperature": flags_df.get("flag_dew_point_exceeds_temp", pd.Series(0)).sum(),
            "Relative Humidity Out of Bounds": flags_df.get("flag_rh_out_of_bounds", pd.Series(0)).sum(),
            "Vapor Pressure > Atmospheric": flags_df.get("flag_vapor_exceeds_pressure", pd.Series(0)).sum(),
            "Negative Wind Speed": flags_df.get("flag_wind_negative", pd.Series(0)).sum(),
            "Wind Speed Exceeds Typical Range": flags_df.get("flag_wind_exceeds_typical", pd.Series(0)).sum(),
            "Temperature Climatological Breach": flags_df.get("flag_temp_out_of_bounds", pd.Series(0)).sum(),
            "Pressure Climatological Breach": flags_df.get("flag_pressure_out_of_bounds", pd.Series(0)).sum(),
        }

        stat_counts = {
            "Rate-of-Change Spikes": sum(flags_df.get(f"flag_{c}_spike", pd.Series(0)).sum() for c in channels),
            "Rolling Z-Score Outliers": sum(flags_df.get(f"flag_{c}_zscore_outlier", pd.Series(0)).sum() for c in channels),
            "IQR Boxplot Outliers": sum(flags_df.get(f"flag_{c}_iqr_outlier", pd.Series(0)).sum() for c in channels),
            "Stuck / Frozen Sensor Readings": sum(flags_df.get(f"flag_{c}_stuck", pd.Series(0)).sum() for c in channels),
            "Missing / Sentinel Readings": sum(flags_df.get(f"flag_{c}_missing", pd.Series(0)).sum() for c in channels),
        }

        # Overall rates
        all_flag_cols = [c for c in flags_df.columns if c.startswith("flag_") and not c.endswith("_missing")]
        any_anomaly = flags_df[all_flag_cols].any(axis=1) if all_flag_cols else pd.Series(False, index=raw_df.index)
        
        missing_cols = [c for c in flags_df.columns if c.endswith("_missing")]
        any_missing = flags_df[missing_cols].any(axis=1) if missing_cols else pd.Series(False, index=raw_df.index)

        all_clean_rows = (~any_anomaly) & (~any_missing)
        overall_valid_pct = (all_clean_rows.sum() / total_records * 100) if total_records > 0 else 0.0
        overall_missing_pct = (any_missing.sum() / total_records * 100) if total_records > 0 else 0.0
        overall_anomaly_pct = (any_anomaly.sum() / total_records * 100) if total_records > 0 else 0.0

        return DataQualityReport(
            total_records=total_records,
            time_range=(time_start, time_end),
            overall_valid_percent=overall_valid_pct,
            overall_missing_percent=overall_missing_pct,
            overall_anomaly_percent=overall_anomaly_pct,
            channel_summaries=channel_summaries_df,
            anomalies_log=anomalies_log,
            physics_violation_counts=physics_counts,
            statistical_anomaly_counts=stat_counts,
        )
