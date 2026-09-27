"""
Hybrid Multi-Model Anomaly Detection Engine for AWS Weather Sensor Data.

Ensembles:
1. Isolation Forest (Unsupervised multi-variate point & correlation anomaly detector)
2. LSTM Autoencoder (Temporal sequence pattern recognition & reconstruction error)
3. Physics-Informed Rule Engine (Thermodynamic HIGH severity, Sensor Spec MEDIUM severity, Statistical LOW severity)

Outputs:
- Anomaly Type
- Severity Level (CRITICAL / WARNING / INFO)
- Unified Confidence Score (0.0 - 1.0)
- Recommended Action
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from .isolation_forest_detector import AWSIsolationForestDetector, IsolationForestResult
from .lstm_autoencoder import AWSLSTMAutoencoderDetector, LSTMAutoencoderResult
from .physics_rule_engine import PhysicsRuleEngine, RuleViolation


@dataclass
class AnomalyDetectionEvent:
    """Detailed anomaly record output for a single reading."""
    timestamp: Any
    anomaly_detected: bool
    primary_anomaly_type: str
    severity: str             # 'CRITICAL', 'WARNING', 'INFO', 'NORMAL'
    confidence: float         # 0.0 - 1.0
    recommended_action: str
    affected_sensors: List[str]
    description: str
    physics_violations: List[str]
    iforest_flag: bool
    iforest_confidence: float
    lstm_flag: bool
    lstm_confidence: float


@dataclass
class HybridDetectionResult:
    """Complete results from Hybrid Anomaly Detector."""
    events: List[AnomalyDetectionEvent]
    summary_df: pd.DataFrame
    alerts_df: pd.DataFrame
    iforest_result: IsolationForestResult
    lstm_result: LSTMAutoencoderResult
    rule_violations: List[List[RuleViolation]]

    def get_alerts_dataframe(self, min_severity: Optional[str] = None) -> pd.DataFrame:
        """Filter and return alerts table."""
        if self.alerts_df.empty:
            return self.alerts_df
        if min_severity is None:
            return self.alerts_df
        
        severity_order = {"INFO": 1, "WARNING": 2, "CRITICAL": 3}
        min_level = severity_order.get(min_severity.upper(), 1)
        
        mask = self.alerts_df["severity"].map(lambda s: severity_order.get(s, 0)) >= min_level
        return self.alerts_df[mask]


class AWSHybridAnomalyDetector:
    """
    Production-grade hybrid anomaly detection system for Automatic Weather Stations.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        lstm_seq_len: int = 12,
        lstm_epochs: int = 25,
        iforest_contamination: float = 0.04,
        confidence_threshold: float = 0.45,
    ):
        self.channels = channels or ["temperature", "pressure", "humidity", "wind_speed"]
        self.confidence_threshold = confidence_threshold

        # Initialize sub-detectors
        self.iforest = AWSIsolationForestDetector(
            contamination=iforest_contamination,
            n_estimators=150,
            random_state=42,
        )
        self.lstm = AWSLSTMAutoencoderDetector(
            channels=self.channels,
            seq_len=lstm_seq_len,
            hidden_dim=32,
            latent_dim=16,
            epochs=lstm_epochs,
            batch_size=64,
        )
        self.physics_engine = PhysicsRuleEngine()
        self.is_trained = False

    def train(self, normal_df: pd.DataFrame) -> "AWSHybridAnomalyDetector":
        """
        Train both Isolation Forest and LSTM Autoencoder on clean operational baseline time-series.
        """
        print(" -> Fitting Isolation Forest on baseline data...")
        self.iforest.fit(normal_df)

        print(f" -> Training LSTM Autoencoder ({self.lstm.epochs} epochs) on temporal sequence patterns...")
        self.lstm.fit(normal_df)

        self.is_trained = True
        print(" -> Hybrid models successfully trained!")
        return self

    def detect(
        self,
        df: pd.DataFrame,
        stat_flags_df: Optional[pd.DataFrame] = None,
        timestamp_col: str = "timestamp"
    ) -> HybridDetectionResult:
        """
        Run multi-model hybrid inference and synthesize alerts, severities, and recommendations.
        """
        if not self.is_trained:
            self.train(df)

        # 1. Run Isolation Forest
        iforest_res = self.iforest.predict(df)

        # 2. Run LSTM Autoencoder
        lstm_res = self.lstm.predict(df)

        # 3. Run Physics Rule Engine
        rule_violations = self.physics_engine.evaluate_dataframe(df, stat_flags_df=stat_flags_df)

        n_rows = len(df)
        events: List[AnomalyDetectionEvent] = []
        timestamps = df[timestamp_col] if timestamp_col in df.columns else df.index

        for i in range(n_rows):
            ts = timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i]
            
            # Physics rule results
            v_list = rule_violations[i]
            has_high_physics = any(v.severity == "HIGH" for v in v_list)
            has_med_spec = any(v.severity == "MEDIUM" for v in v_list)
            has_low_stat = any(v.severity == "LOW" for v in v_list)

            # ML model outputs
            if_flag = bool(iforest_res.is_anomaly.iloc[i])
            if_conf = float(iforest_res.anomaly_confidence.iloc[i])
            lstm_flag = bool(lstm_res.is_anomaly.iloc[i])
            lstm_conf = float(lstm_res.anomaly_confidence.iloc[i])

            # Determine severity, anomaly type, confidence, and action
            # -------------------------------------------------------------
            # PRIORITY 1: HIGH Physics / Thermodynamic Violations -> CRITICAL
            # -------------------------------------------------------------
            if has_high_physics:
                high_v = next(v for v in v_list if v.severity == "HIGH")
                conf = max(high_v.confidence, 0.95)
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type=f"THERMODYNAMIC_VIOLATION ({high_v.rule_id})",
                    severity="CRITICAL",
                    confidence=conf,
                    recommended_action=high_v.recommended_action,
                    affected_sensors=high_v.affected_sensors,
                    description=high_v.description,
                    physics_violations=[v.rule_id for v in v_list],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            # -------------------------------------------------------------
            # PRIORITY 2: MEDIUM Sensor Specification Breach -> WARNING
            # -------------------------------------------------------------
            elif has_med_spec:
                med_v = next(v for v in v_list if v.severity == "MEDIUM")
                conf = max(med_v.confidence, 0.88)
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type=f"SENSOR_SPEC_BREACH ({med_v.rule_id})",
                    severity="WARNING",
                    confidence=conf,
                    recommended_action=med_v.recommended_action,
                    affected_sensors=med_v.affected_sensors,
                    description=med_v.description,
                    physics_violations=[v.rule_id for v in v_list],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            # -------------------------------------------------------------
            # PRIORITY 3: Multi-Model Consensus (Isolation Forest + LSTM)
            # -------------------------------------------------------------
            elif if_flag and lstm_flag:
                ensemble_conf = float(np.mean([if_conf, lstm_conf]))
                severity = "WARNING"
                
                # Identify dominant affected sensor from LSTM reconstruction error
                lstm_errs = lstm_res.channel_errors_df.iloc[i]
                max_ch = lstm_errs.idxmax().replace("lstm_err_", "")
                
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type="TEMPORAL_MULTIVARIATE_ANOMALY",
                    severity=severity,
                    confidence=ensemble_conf,
                    recommended_action=f"Deep inspection required: Temporal pattern and multi-sensor correlation deviation in {max_ch}",
                    affected_sensors=[max_ch],
                    description=f"Dual ML model consensus: Temporal sequence reconstruction error (LSTM) and multi-variate feature isolation (IForest) flagged anomalous pattern",
                    physics_violations=[],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            # -------------------------------------------------------------
            # PRIORITY 4: Individual ML Model Trigger (High Confidence)
            # -------------------------------------------------------------
            elif lstm_flag and lstm_conf >= 0.70:
                lstm_errs = lstm_res.channel_errors_df.iloc[i]
                max_ch = lstm_errs.idxmax().replace("lstm_err_", "")
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type="TEMPORAL_SEQUENCE_DISTORTION",
                    severity="WARNING",
                    confidence=lstm_conf,
                    recommended_action=f"Inspect time-series trajectory of {max_ch}; check for sudden rate-of-change distortion or sensor drift",
                    affected_sensors=[max_ch],
                    description=f"LSTM Autoencoder reconstruction error exceeded 3-sigma baseline",
                    physics_violations=[],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            elif if_flag and if_conf >= 0.70:
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type="MULTIVARIATE_OUTLIER",
                    severity="WARNING",
                    confidence=if_conf,
                    recommended_action="Inspect multi-sensor correlation matrix; check for abnormal atmospheric coupling",
                    affected_sensors=self.channels,
                    description="Isolation Forest identified rare multi-dimensional feature combination",
                    physics_violations=[],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            # -------------------------------------------------------------
            # PRIORITY 5: LOW Severity Statistical Outlier -> INFO
            # -------------------------------------------------------------
            elif has_low_stat:
                low_v = next(v for v in v_list if v.severity == "LOW")
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=True,
                    primary_anomaly_type=f"STATISTICAL_OUTLIER ({low_v.rule_id})",
                    severity="INFO",
                    confidence=low_v.confidence,
                    recommended_action=low_v.recommended_action,
                    affected_sensors=low_v.affected_sensors,
                    description=low_v.description,
                    physics_violations=[v.rule_id for v in v_list],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            # -------------------------------------------------------------
            # NORMAL OPERATION
            # -------------------------------------------------------------
            else:
                event = AnomalyDetectionEvent(
                    timestamp=ts,
                    anomaly_detected=False,
                    primary_anomaly_type="NORMAL",
                    severity="NORMAL",
                    confidence=float(1.0 - max(if_conf, lstm_conf)),
                    recommended_action="Normal operational routine",
                    affected_sensors=[],
                    description="Nominal sensor operation within physical and statistical bounds",
                    physics_violations=[],
                    iforest_flag=if_flag,
                    iforest_confidence=if_conf,
                    lstm_flag=lstm_flag,
                    lstm_confidence=lstm_conf,
                )

            events.append(event)

        # Build DataFrames
        summary_records = []
        alerts_records = []

        for e in events:
            rec = {
                "timestamp": e.timestamp,
                "is_anomaly": e.anomaly_detected,
                "anomaly_type": e.primary_anomaly_type,
                "severity": e.severity,
                "confidence": round(e.confidence, 3),
                "affected_sensors": ", ".join(e.affected_sensors) if e.affected_sensors else "None",
                "recommended_action": e.recommended_action,
                "description": e.description,
                "iforest_flag": e.iforest_flag,
                "iforest_conf": round(e.iforest_confidence, 3),
                "lstm_flag": e.lstm_flag,
                "lstm_conf": round(e.lstm_confidence, 3),
            }
            summary_records.append(rec)
            if e.anomaly_detected:
                alerts_records.append(rec)

        summary_df = pd.DataFrame(summary_records)
        alerts_df = pd.DataFrame(alerts_records)

        return HybridDetectionResult(
            events=events,
            summary_df=summary_df,
            alerts_df=alerts_df,
            iforest_result=iforest_res,
            lstm_result=lstm_res,
            rule_violations=rule_violations,
        )
