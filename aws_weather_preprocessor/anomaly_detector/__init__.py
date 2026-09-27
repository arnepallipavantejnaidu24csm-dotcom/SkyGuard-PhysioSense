"""
Anomaly Detector Sub-package combining:
1. Isolation Forest (Unsupervised point & multi-variate detector)
2. LSTM Autoencoder (Temporal sequence pattern recognition)
3. Physics-informed rules with hierarchical severity (HIGH/MEDIUM/LOW)
4. Multi-level alerts (CRITICAL/WARNING/INFO) with confidence scores and recommended actions
"""

from .isolation_forest_detector import (
    AWSIsolationForestDetector,
    IsolationForestResult,
)
from .lstm_autoencoder import (
    AWSLSTMAutoencoderDetector,
    LSTMAutoencoderResult,
)
from .physics_rule_engine import (
    PhysicsRuleEngine,
    RuleViolation,
)
from .hybrid_detector import (
    AWSHybridAnomalyDetector,
    AnomalyDetectionEvent,
    HybridDetectionResult,
)

__all__ = [
    "AWSIsolationForestDetector",
    "IsolationForestResult",
    "AWSLSTMAutoencoderDetector",
    "LSTMAutoencoderResult",
    "PhysicsRuleEngine",
    "RuleViolation",
    "AWSHybridAnomalyDetector",
    "AnomalyDetectionEvent",
    "HybridDetectionResult",
]
