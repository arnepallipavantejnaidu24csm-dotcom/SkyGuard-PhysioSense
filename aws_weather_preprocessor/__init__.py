"""
AWS Weather Sensor Data Preprocessor
A modular Python package for automated quality control, physics-based consistency validation,
statistical anomaly detection, and reporting for Automatic Weather Station (AWS) time-series data.
"""

from .physics_checks import (
    calculate_saturation_vapor_pressure,
    calculate_actual_vapor_pressure,
    calculate_dew_point,
    validate_t_p_rh_physics,
    validate_wind_speed_bounds,
    PhysicsQualityChecker,
)
from .statistical_checks import (
    detect_missing_values,
    detect_iqr_outliers,
    detect_rolling_zscore_outliers,
    detect_step_spikes,
    detect_stuck_sensors,
    StatisticalQualityChecker,
)
from .preprocessor import (
    AWSDataPreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
)
from .quality_reporter import (
    DataQualityReport,
    QualityReportGenerator,
)
from .synthetic_data import (
    generate_synthetic_aws_data,
    SyntheticDataConfig,
)
from .kalman_tracker import (
    AtmosphericStateKalmanFilter,
    KalmanFilterConfig,
    KalmanOutput,
)
from .drift_scenarios import (
    generate_drift_scenario,
    DriftExperimentResult,
)
from .anomaly_detector import (
    AWSIsolationForestDetector,
    IsolationForestResult,
    AWSLSTMAutoencoderDetector,
    LSTMAutoencoderResult,
    PhysicsRuleEngine,
    RuleViolation,
    AWSHybridAnomalyDetector,
    AnomalyDetectionEvent,
    HybridDetectionResult,
)
from .health_scoring import (
    SensorProfile,
    SensorHealthMetrics,
    MaintenanceAlert,
    SensorHealthScoringSystem,
)
from .correction_alerting import (
    ActionableAlert,
    HistoricalAnomalyTracker,
    SensorAutoCorrector,
    CorrectionAndAlertingResult,
    AWSCorrectionAndAlertingPipeline,
)

__all__ = [
    "calculate_saturation_vapor_pressure",
    "calculate_actual_vapor_pressure",
    "calculate_dew_point",
    "validate_t_p_rh_physics",
    "validate_wind_speed_bounds",
    "PhysicsQualityChecker",
    "detect_missing_values",
    "detect_iqr_outliers",
    "detect_rolling_zscore_outliers",
    "detect_step_spikes",
    "detect_stuck_sensors",
    "StatisticalQualityChecker",
    "AWSDataPreprocessor",
    "PreprocessingConfig",
    "PreprocessingResult",
    "DataQualityReport",
    "QualityReportGenerator",
    "generate_synthetic_aws_data",
    "SyntheticDataConfig",
    "AtmosphericStateKalmanFilter",
    "KalmanFilterConfig",
    "KalmanOutput",
    "generate_drift_scenario",
    "DriftExperimentResult",
    "AWSIsolationForestDetector",
    "IsolationForestResult",
    "AWSLSTMAutoencoderDetector",
    "LSTMAutoencoderResult",
    "PhysicsRuleEngine",
    "RuleViolation",
    "AWSHybridAnomalyDetector",
    "AnomalyDetectionEvent",
    "HybridDetectionResult",
    "SensorProfile",
    "SensorHealthMetrics",
    "MaintenanceAlert",
    "SensorHealthScoringSystem",
    "ActionableAlert",
    "HistoricalAnomalyTracker",
    "SensorAutoCorrector",
    "CorrectionAndAlertingResult",
    "AWSCorrectionAndAlertingPipeline",
]

__version__ = "1.0.0"
