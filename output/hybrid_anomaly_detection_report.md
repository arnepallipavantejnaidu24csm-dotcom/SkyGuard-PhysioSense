# 🛡️ Hybrid AWS Anomaly Detection & Alert Report

## 1. Executive Summary
- **Total Records Analyzed**: `2,016` readings
- **Total Anomalies Flagged**: `57` incidents (2.83%)
- **Critical Alerts**: `8`
- **Warning Alerts**: `49`
- **Info Alerts**: `0`

## 2. Multi-Model Detection Hierarchy
| Component | Detection Scope | Severity Mapping |
|---|---|---|
| **Physics Rule Engine** | Thermodynamic laws ($T_d \le T, e < P, RH \le 100\%$) | **HIGH** $\to$ `CRITICAL` |
| **Physics Rule Engine** | Sensor specifications ($WS \ge 0$, operating ranges) | **MEDIUM** $\to$ `WARNING` |
| **LSTM Autoencoder** | Temporal sequence reconstruction & pattern distortion | **MEDIUM/HIGH** $\to$ `WARNING`/`CRITICAL` |
| **Isolation Forest** | Multi-variate point & cross-channel correlation outliers | **MEDIUM** $\to$ `WARNING` |
| **Statistical Rules** | Transient rate-of-change spikes & flatlines | **LOW** $\to$ `INFO`/`WARNING` |

## 3. Flagged Anomaly Incidents & Recommended Actions

| timestamp           | severity   |   confidence | anomaly_type                                            | affected_sensors      | recommended_action                                                                                        |
|:--------------------|:-----------|-------------:|:--------------------------------------------------------|:----------------------|:----------------------------------------------------------------------------------------------------------|
| 2026-09-01 09:30:00 | WARNING    |        0.95  | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-01 10:55:00 | WARNING    |        0.861 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-01 13:05:00 | WARNING    |        0.756 | TEMPORAL_MULTIVARIATE_ANOMALY                           | wind_speed            | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in wind_speed           |
| 2026-09-01 13:25:00 | CRITICAL   |        0.99  | THERMODYNAMIC_VIOLATION (THERMO_VAPOR_EXCEEDS_PRESSURE) | pressure, humidity    | Isolate barometer and RH sensor channels; check for sensor chamber flooding                               |
| 2026-09-01 13:30:00 | CRITICAL   |        0.99  | THERMODYNAMIC_VIOLATION (THERMO_VAPOR_EXCEEDS_PRESSURE) | pressure, humidity    | Isolate barometer and RH sensor channels; check for sensor chamber flooding                               |
| 2026-09-01 13:35:00 | CRITICAL   |        0.99  | THERMODYNAMIC_VIOLATION (THERMO_VAPOR_EXCEEDS_PRESSURE) | pressure, humidity    | Isolate barometer and RH sensor channels; check for sensor chamber flooding                               |
| 2026-09-01 13:40:00 | WARNING    |        0.982 | TEMPORAL_SEQUENCE_DISTORTION                            | temperature           | Inspect time-series trajectory of temperature; check for sudden rate-of-change distortion or sensor drift |
| 2026-09-01 13:45:00 | WARNING    |        1     | TEMPORAL_SEQUENCE_DISTORTION                            | temperature           | Inspect time-series trajectory of temperature; check for sudden rate-of-change distortion or sensor drift |
| 2026-09-01 13:50:00 | WARNING    |        1     | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-01 13:55:00 | WARNING    |        0.765 | TEMPORAL_MULTIVARIATE_ANOMALY                           | temperature           | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in temperature          |
| 2026-09-01 14:00:00 | WARNING    |        1     | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-01 14:05:00 | WARNING    |        0.982 | TEMPORAL_SEQUENCE_DISTORTION                            | temperature           | Inspect time-series trajectory of temperature; check for sudden rate-of-change distortion or sensor drift |
| 2026-09-01 14:10:00 | WARNING    |        0.966 | TEMPORAL_SEQUENCE_DISTORTION                            | temperature           | Inspect time-series trajectory of temperature; check for sudden rate-of-change distortion or sensor drift |
| 2026-09-01 14:25:00 | WARNING    |        0.993 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-02 01:10:00 | CRITICAL   |        0.98  | THERMODYNAMIC_VIOLATION (THERMO_TD_EXCEEDS_T)           | temperature, humidity | Emergency calibration required: Inspect hygrometer sensing element and aspirator fan                      |
| 2026-09-02 04:25:00 | WARNING    |        0.889 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-02 09:30:00 | CRITICAL   |        0.858 | TEMPORAL_MULTIVARIATE_ANOMALY                           | wind_speed            | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in wind_speed           |
| 2026-09-02 10:25:00 | WARNING    |        0.942 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-02 13:05:00 | WARNING    |        0.888 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-02 14:40:00 | WARNING    |        0.754 | TEMPORAL_MULTIVARIATE_ANOMALY                           | wind_speed            | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in wind_speed           |
| 2026-09-02 16:05:00 | WARNING    |        0.754 | TEMPORAL_MULTIVARIATE_ANOMALY                           | wind_speed            | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in wind_speed           |
| 2026-09-02 16:45:00 | WARNING    |        0.686 | TEMPORAL_MULTIVARIATE_ANOMALY                           | wind_speed            | Deep inspection required: Temporal pattern and multi-sensor correlation deviation in wind_speed           |
| 2026-09-02 17:20:00 | WARNING    |        0.994 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-03 10:00:00 | WARNING    |        0.952 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |
| 2026-09-03 10:15:00 | WARNING    |        0.958 | TEMPORAL_SEQUENCE_DISTORTION                            | wind_speed            | Inspect time-series trajectory of wind_speed; check for sudden rate-of-change distortion or sensor drift  |

*(Showing first 25 of 57 flagged alerts)*
