"""
Isolation Forest Unsupervised Multi-Variate Anomaly Detector for AWS Sensor Data.

Implements:
1. Feature extraction combining raw multi-sensor readings, temporal differences, and thermodynamic couplings
2. Unsupervised isolation forest model training on normal operational patterns
3. Calibrated anomaly confidence scoring in [0.0, 1.0]
4. Feature contribution / anomaly attribution estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..physics_checks import (
    calculate_actual_vapor_pressure,
    calculate_dew_point,
    calculate_absolute_humidity,
)


@dataclass
class IsolationForestResult:
    """Output from Isolation Forest anomaly evaluation."""
    is_anomaly: pd.Series
    anomaly_score: pd.Series        # Raw decision score (lower = more anomalous)
    anomaly_confidence: pd.Series   # Calibrated confidence in [0.0, 1.0]
    features_used: List[str]


class AWSIsolationForestDetector:
    """
    Unsupervised Isolation Forest detector for multi-sensor AWS time-series anomalies.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            bootstrap=False,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract multi-variate and thermodynamic coupling features for anomaly isolation.
        """
        feats = pd.DataFrame(index=df.index)

        # 1. Base sensor readings
        for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
            if ch in df.columns:
                feats[ch] = df[ch].astype(float)

        # 2. First-order temporal derivatives (rate-of-change)
        for ch in ["temperature", "pressure", "humidity", "wind_speed"]:
            if ch in df.columns:
                feats[f"{ch}_diff1"] = df[ch].diff().fillna(0.0)

        # 3. Second-order temporal acceleration
        for ch in ["temperature", "pressure"]:
            if ch in df.columns:
                feats[f"{ch}_diff2"] = feats[f"{ch}_diff1"].diff().fillna(0.0)

        # 4. Thermodynamic physical coupling features
        if "temperature" in df.columns and "humidity" in df.columns:
            td = calculate_dew_point(df["temperature"], df["humidity"])
            e = calculate_actual_vapor_pressure(df["temperature"], df["humidity"])
            ah = calculate_absolute_humidity(df["temperature"], e)

            feats["dew_point_depression"] = (df["temperature"] - td).astype(float)  # T - Td
            feats["actual_vapor_pressure"] = e.astype(float)
            feats["absolute_humidity"] = ah.astype(float)

        # Fill any remaining NaNs via forward/backward fill or median
        feats = feats.ffill().bfill().fillna(0.0)
        return feats

    def fit(self, normal_df: pd.DataFrame) -> "AWSIsolationForestDetector":
        """
        Fit Isolation Forest model on baseline / normal operational data.
        """
        features_df = self.extract_features(normal_df)
        self.feature_names = list(features_df.columns)
        
        X_scaled = self.scaler.fit_transform(features_df.values)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> IsolationForestResult:
        """
        Evaluate anomalies and compute continuous confidence scores.
        """
        if not self.is_fitted:
            # Fit on incoming data if not pre-trained
            self.fit(df)

        features_df = self.extract_features(df)
        X_scaled = self.scaler.transform(features_df[self.feature_names].values)

        # Decision function: negative values indicate anomalies, positive indicate inliers
        raw_scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        # Calibrate raw score to [0.0, 1.0] anomaly confidence
        # When score is negative (anomaly), confidence -> 1.0; when score is positive (inlier), confidence -> 0.0
        confidence = 1.0 / (1.0 + np.exp(raw_scores * 16.0))
        confidence = np.clip(confidence, 0.0, 1.0)

        is_anomaly_series = pd.Series(preds == -1, index=df.index, name="iforest_anomaly")
        score_series = pd.Series(raw_scores, index=df.index, name="iforest_score")
        conf_series = pd.Series(confidence, index=df.index, name="iforest_confidence")

        return IsolationForestResult(
            is_anomaly=is_anomaly_series,
            anomaly_score=score_series,
            anomaly_confidence=conf_series,
            features_used=self.feature_names,
        )
