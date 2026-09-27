# 🏥 AWS Sensor Health & Maintenance Dashboard (AWS-STATION-NORTH-01)

## 1. Station Sensor Health Overview

| Sensor | Instrument Model | Health Score | Grade | Completeness | Bias Drift Rate | Noise Level | RUL (Days) | Maintenance Status |
|---|---|---|---|---|---|---|---|---|
| `Temperature` | Pt100 4-Wire RTD Class 1/10 DIN | `[█████░░░░░]` **55.8%** | 🟠 **DEGRADED** | 100.0% | `+0.201`/day | σ=0.13 | **0 d** | `URGENT_ACTION_REQUIRED` |
| `Pressure` | Vaisala PTB330 Digital Barometer | `[█████░░░░░]` **56.8%** | 🟠 **DEGRADED** | 100.0% | `-0.307`/day | σ=0.09 | **0 d** | `URGENT_ACTION_REQUIRED` |
| `Humidity` | Rotronic HygroMet4 Thin-Film Capacitive | `[████████░░]` **83.2%** | 🟢 **GOOD** | 100.0% | `-0.174`/day | σ=0.72 | **13 d** | `MAINTENANCE_DUE` |
| `Wind_speed` | Gill WindObserver II Ultrasonic Anemometer | `[███████░░░]` **75.9%** | 🟢 **GOOD** | 96.03% | `+0.054`/day | σ=0.36 | **10 d** | `MAINTENANCE_DUE` |

## 2. Sensor-by-Sensor Detailed Diagnostics

### 📡 Temperature (Pt100 4-Wire RTD Class 1/10 DIN)
- **Overall Health Score**: `55.8%` (DEGRADED)
- **Degradation Status**: ⚠️ **DEGRADED SENSOR FLAGGED**
- **Data Completeness**: `100.0%` (2016/2016 readings received)
- **Current Estimated Bias**: `+1.417` (Drift Rate: `+0.2010`/day)
- **Residual Noise Std**: `0.130` (Variance: `0.0169`)
- **Calibration Status**: `OUT_OF_SPEC` (207 days since last calibration)
- **Predicted Remaining Useful Life (RUL)**: **0.0 days**
- **Action Required**: URGENT: Immediate field inspection & sensor replacement / full recalibration required for temperature.

### 📡 Pressure (Vaisala PTB330 Digital Barometer)
- **Overall Health Score**: `56.8%` (DEGRADED)
- **Degradation Status**: ⚠️ **DEGRADED SENSOR FLAGGED**
- **Data Completeness**: `100.0%` (2016/2016 readings received)
- **Current Estimated Bias**: `-1.705` (Drift Rate: `-0.3073`/day)
- **Residual Noise Std**: `0.094` (Variance: `0.0089`)
- **Calibration Status**: `OUT_OF_SPEC` (252 days since last calibration)
- **Predicted Remaining Useful Life (RUL)**: **0.0 days**
- **Action Required**: URGENT: Immediate field inspection & sensor replacement / full recalibration required for pressure.

### 📡 Humidity (Rotronic HygroMet4 Thin-Film Capacitive)
- **Overall Health Score**: `83.2%` (GOOD)
- **Degradation Status**: ✅ **NORMAL OPERATION**
- **Data Completeness**: `100.0%` (2016/2016 readings received)
- **Current Estimated Bias**: `-0.745` (Drift Rate: `-0.1744`/day)
- **Residual Noise Std**: `0.718` (Variance: `0.5161`)
- **Calibration Status**: `DUE_SOON` (167 days since last calibration)
- **Predicted Remaining Useful Life (RUL)**: **13.0 days**
- **Action Required**: Schedule maintenance within 13 days: Recalibrate zero/span and clean sensor aperture.

### 📡 Wind_speed (Gill WindObserver II Ultrasonic Anemometer)
- **Overall Health Score**: `75.9%` (GOOD)
- **Degradation Status**: ✅ **NORMAL OPERATION**
- **Data Completeness**: `96.03%` (1936/2016 readings received)
- **Current Estimated Bias**: `+0.254` (Drift Rate: `+0.0541`/day)
- **Residual Noise Std**: `0.360` (Variance: `0.1297`)
- **Calibration Status**: `DUE_SOON` (327 days since last calibration)
- **Predicted Remaining Useful Life (RUL)**: **10.1 days**
- **Action Required**: Schedule maintenance within 10 days: Recalibrate zero/span and clean sensor aperture.

## 3. Prioritized Maintenance Alerts Queue
| Priority | Sensor | Trigger Diagnostic | Remaining Life | Technician Action |
|---|---|---|---|---|
| 🚨 **URGENT** | `Temperature` | Health Score Critical (55.8%) | Bias (+1.42) exceeds limits | RUL: 0.0 days | **0 days** | URGENT: Immediate field inspection & sensor replacement / full recalibration required for temperature. |
| 🚨 **URGENT** | `Pressure` | Health Score Critical (56.8%) | Bias (-1.71) exceeds limits | RUL: 0.0 days | **0 days** | URGENT: Immediate field inspection & sensor replacement / full recalibration required for pressure. |
| ⚠️ **HIGH** | `Humidity` | Sensor Degraded (83.2%) | RUL: 13.0 days | Drift: -0.174/day | **13 days** | Schedule maintenance within 13 days: Recalibrate zero/span and clean sensor aperture. |
| ⚠️ **HIGH** | `Wind_speed` | Sensor Degraded (75.9%) | RUL: 10.1 days | Drift: +0.054/day | **10 days** | Schedule maintenance within 10 days: Recalibrate zero/span and clean sensor aperture. |
