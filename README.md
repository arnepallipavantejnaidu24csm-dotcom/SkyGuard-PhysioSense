# AWS Weather Sensor Preprocessing & Anomaly Detection Framework

A production-grade Python framework for:
1. **Automated Data Quality Control & Preprocessing** (Physics-based consistency & statistical checks)
2. **State-Space Kalman Filtering** (True atmospheric state tracking vs instrument drift / bias estimation)
3. **Hybrid Machine Learning Anomaly Detection** (Isolation Forest + PyTorch LSTM Autoencoder + Physics-informed rules with multi-level alerts)

---

## 🌟 1. System Architecture

```
├── aws_weather_preprocessor/
│   ├── __init__.py
│   ├── physics_checks.py              # Thermodynamic calculations & physical consistency validation
│   ├── statistical_checks.py          # Missing value, IQR, rolling Z-score, spike, flatline detection
│   ├── preprocessor.py                # Main AWSDataPreprocessor pipeline orchestrator
│   ├── quality_reporter.py            # DataQualityReport generator & anomaly location logger
│   ├── synthetic_data.py              # High-resolution AWS time-series generator with diurnal cycles
│   ├── kalman_tracker.py              # State-Space Kalman Filter for true state vs sensor bias
│   ├── drift_scenarios.py             # Sensor drift & instrument fouling simulation benchmarks
│   └── anomaly_detector/              # Hybrid Machine Learning & Physics-Informed Anomaly Engine
│       ├── __init__.py
│       ├── physics_rule_engine.py     # Hierarchical rule engine (HIGH/MEDIUM/LOW severity)
│       ├── isolation_forest_detector.py # Unsupervised multi-variate & feature isolation detector
│       ├── lstm_autoencoder.py        # PyTorch LSTM Autoencoder for temporal pattern distortion
│       └── hybrid_detector.py         # Multi-model ensemble fusion, confidence scoring & alerts
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py           # Unit tests for QC & thermodynamic equations
│   ├── test_kalman_filter.py          # Unit tests for Kalman filter & drift tracking
│   └── test_anomaly_detector.py       # Unit tests for Isolation Forest, LSTM, and Hybrid engine
├── output/                            # Output artifacts, cleaned CSVs, flags, and reports
│   ├── sample_raw_aws_data.csv
│   ├── cleaned_aws_data.csv
│   ├── aws_anomaly_flags.csv
│   ├── kalman_linear_ramp_tracking_results.csv
│   ├── hybrid_anomaly_detection_results.csv
│   ├── hybrid_anomaly_alerts.csv
│   ├── data_quality_report.md
│   ├── kalman_state_space_report.md
│   └── hybrid_anomaly_detection_report.md
├── run_demo.py                        # Preprocessing demo script
├── run_kalman_demo.py                 # Kalman filtering & sensor drift demo
├── run_anomaly_detector_demo.py       # Hybrid anomaly detection & alert demo
└── README.md
```

---

## 🛡️ 2. Hybrid Anomaly Detection Module

### Detection Hierarchy & Severity Mapping:
| Detection Engine | Scope & Underlying Dynamics | Rule Severity | Alert Level | Confidence Range |
|---|---|---|---|---|
| **Physics Rule Engine** | Fundamental thermodynamic laws ($T_d \le T$, $e < P$, $0 \le RH \le 105\%$) | **HIGH** | `CRITICAL` | $0.95 - 0.99$ |
| **Physics Rule Engine** | Sensor operating specifications ($WS \ge 0$, hardware boundaries) | **MEDIUM** | `WARNING` | $0.88 - 0.95$ |
| **LSTM Autoencoder** | Temporal sequence reconstruction & rate-of-change distortion | **MEDIUM/HIGH** | `WARNING`/`CRITICAL` | $0.70 - 1.00$ |
| **Isolation Forest** | Multi-variate point & cross-channel correlation outliers | **MEDIUM** | `WARNING` | $0.70 - 0.95$ |
| **Statistical Rules** | Transient spikes, sensor flatlines, distribution excursions | **LOW** | `INFO`/`WARNING` | $0.65 - 0.85$ |

### Actionable Maintenance Outputs:
Every detected anomaly produces:
- `primary_anomaly_type`: Exact root-cause classification (e.g. `THERMODYNAMIC_VIOLATION (THERMO_TD_EXCEEDS_T)`).
- `severity`: Alert tier (`CRITICAL`, `WARNING`, `INFO`).
- `confidence`: Calibrated continuous score in $[0.0, 1.0]$.
- `affected_sensors`: List of impacted hardware channels.
- `recommended_action`: Context-aware maintenance instructions (e.g. *"Emergency calibration required: Inspect hygrometer sensing element and aspirator fan"*).

---

## 🛰️ 3. State-Space Kalman Filter (Bias Tracking)

- **Continuous-Discrete State-Space Model**:
  $$\mathbf{x}_k = \begin{bmatrix} \mathbf{x}_{k, \text{weather}} \\ \mathbf{x}_{k, \text{bias}} \end{bmatrix} = \begin{bmatrix} T_k, \dot{T}_k, P_k, \dot{P}_k, RH_k, \dot{RH}_k, WS_k, \dot{WS}_k, b_{T, k}, b_{P, k}, b_{RH, k}, b_{WS, k} \end{bmatrix}^T$$
- **Observation Model**:
  $$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k, \quad z_i = x_{\text{true}, i} + b_i + v_i$$
- **Joseph Form Numerical Stabilization**: Prevents covariance matrix non-positive definiteness.
- **Residual & Innovation Tracking**: Pre-fit innovations $\mathbf{y}_k$, post-fit residuals $\mathbf{e}_k$, Normalized Innovation Squared (NIS), and dynamic $95\%$ confidence bounds ($\pm 1.96 \sigma$).

---

## ⚡ 4. Execution & Demos

### Run All Unit Tests
```bash
python -m unittest discover -s tests
```
*(All 21 unit & integration tests pass with 100% success).*

### Run Quality Control Preprocessing Pipeline
```bash
python run_demo.py
```

### Run State-Space Kalman Filter Drift Experiments
```bash
python run_kalman_demo.py
```

### Run Hybrid ML + Physics Anomaly Detection Pipeline
```bash
python run_anomaly_detector_demo.py
```

### Run Sensor Health Scoring & Predictive Maintenance
```bash
python run_health_scoring_demo.py
```

### Run Auto-Correction, Confidence Bounding & Real-Time Alerting
```bash
python run_correction_alerting_demo.py
```

### Launch Interactive Web Dashboard Server
```bash
python web_dashboard/app.py
```
Open your browser at `http://127.0.0.1:5000/` to access the live dashboard.

---

## 🌐 5. Web Dashboard Features

- **Live Meteorological Telemetry**: Real-time gauge metrics, auto-corrected values, estimated instrument biases, and dynamic $95\%$ / $99\%$ confidence interval bands.
- **Multi-Channel Line Charts**: Dual lines for Raw Observations (dashed) vs. Auto-Corrected Values (solid) with shaded $95\%$ CI uncertainty envelopes.
- **Dual-Axis Synoptic Comparison**: Synchronized multi-variable coupling (Temperature vs. Barometric Pressure tides).
- **Interactive Multi-Station Map**: Leaflet.js network map tracking multiple AWS stations (`Central ISRIKA1-Bheemili`, `Pataparadesipalem Station Observatory`, `Pacific Marine`, `Mojave Solar Array`).
- **Sensor Health Cards with Radial Meters**: $0 - 100\%$ composite scores, Remaining Useful Life (RUL), drift rate per day, residual noise $\sigma$, and calibration status.
- **Actionable Anomaly Alerts Log**: Real-time searchable & filterable table with root cause, historical recurrence memory, and technician instructions.
- **Interactive Anomaly Injection Sandbox**: One-click live injection of temperature spikes, humidity thermodynamic breaches, barometer offsets, and negative wind speeds.
- **One-Click Data Exports**: Export datasets to CSV, JSON, and printable reports.

## SIH Combined Regional Event + Sensor Health Features

The dashboard now combines:
- Regional Event Corroboration with spatial-agreement scoring.
- Regional-event and isolated-sensor-fault scenario simulation.
- Explainable regional-event decision factors and event fingerprint trends.
- Event Replay Timeline for the selected scenario.
- Sensor Health & Fault Isolation table combining health status with spatial consistency.
- Existing live telemetry, Kalman bias tracking, anomaly alerts, map, charts, and sensor-health diagnostics are preserved.

The regional-event buttons are a scenario/simulation layer; the existing backend telemetry and health APIs remain unchanged.
