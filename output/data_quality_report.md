# 🌤️ AWS Sensor Data Preprocessing & Quality Report

## 1. Executive Summary
- **Total Records Analyzed**: `2,016` readings
- **Time Coverage**: `2026-09-01 00:00:00` to `2026-09-07 23:55:00`
- **Overall Clean Data Yield**: **93.65%**
- **Total Missing Rate**: **0.40%**
- **Total Anomaly Rate**: **6.10%** (Physics + Statistical)

## 2. Sensor Channel Breakdown
| Sensor      | Valid %   |   Missing |   Physics Violations |   IQR Outliers |   Z-Score Outliers |   Spikes |   Stuck / Frozen |
|:------------|:----------|----------:|---------------------:|---------------:|-------------------:|---------:|-----------------:|
| Temperature | 96.9%     |         3 |                    4 |              1 |                 40 |        4 |               17 |
| Pressure    | 98.8%     |         5 |                    1 |              1 |                 18 |        4 |                0 |
| Humidity    | 98.6%     |         0 |                    2 |              0 |                 28 |        3 |                0 |
| Wind_speed  | 99.1%     |         0 |                    2 |             10 |                 11 |        3 |                0 |

## 3. Physics-Based Consistency Violations
| Physics Rule Checked | Violations Count | Description |
|---|---|---|
| `T-P-RH Thermodynamic Violations` | **7** | Thermodynamic / physical limit breach |
| `Dew Point > Dry Bulb Temperature` | **1** | Thermodynamic / physical limit breach |
| `Relative Humidity Out of Bounds` | **2** | Thermodynamic / physical limit breach |
| `Vapor Pressure > Atmospheric` | **3** | Thermodynamic / physical limit breach |
| `Negative Wind Speed` | **1** | Thermodynamic / physical limit breach |
| `Wind Speed Exceeds Typical Range` | **1** | Thermodynamic / physical limit breach |
| `Temperature Climatological Breach` | **4** | Thermodynamic / physical limit breach |
| `Pressure Climatological Breach` | **1** | Thermodynamic / physical limit breach |

## 4. Statistical Anomaly & Outlier Breakdown
| Anomaly Type | Occurrences | Characteristics |
|---|---|---|
| `Rate-of-Change Spikes` | **14** | Detected via IQR / Z-Score / Step / Flatline |
| `Rolling Z-Score Outliers` | **97** | Detected via IQR / Z-Score / Step / Flatline |
| `IQR Boxplot Outliers` | **12** | Detected via IQR / Z-Score / Step / Flatline |
| `Stuck / Frozen Sensor Readings` | **17** | Detected via IQR / Z-Score / Step / Flatline |
| `Missing / Sentinel Readings` | **8** | Detected via IQR / Z-Score / Step / Flatline |

## 5. Sample Detected Anomaly Locations (Top 25)
| timestamp           | sensor               | observed_value                    | anomaly_category   | rule_violated                    | description                                                                                      |
|:--------------------|:---------------------|:----------------------------------|:-------------------|:---------------------------------|:-------------------------------------------------------------------------------------------------|
| 2026-09-01 13:25:00 | temperature          | -999.0                            | MISSING            | temperature_missing_or_sentinel  | Missing reading or sentinel value detected in temperature                                        |
| 2026-09-01 13:25:00 | pressure+humidity    | 1021.8256432354663                | PHYSICS_VIOLATION  | vapor_pressure_le_ambient        | Water vapor pressure exceeds total atmospheric ambient pressure                                  |
| 2026-09-01 13:30:00 | temperature          | -999.0                            | MISSING            | temperature_missing_or_sentinel  | Missing reading or sentinel value detected in temperature                                        |
| 2026-09-01 13:30:00 | pressure+humidity    | 1022.014788480272                 | PHYSICS_VIOLATION  | vapor_pressure_le_ambient        | Water vapor pressure exceeds total atmospheric ambient pressure                                  |
| 2026-09-01 13:35:00 | temperature          | -999.0                            | MISSING            | temperature_missing_or_sentinel  | Missing reading or sentinel value detected in temperature                                        |
| 2026-09-01 13:35:00 | pressure+humidity    | 1021.6237412402363                | PHYSICS_VIOLATION  | vapor_pressure_le_ambient        | Water vapor pressure exceeds total atmospheric ambient pressure                                  |
| 2026-09-02 01:10:00 | temperature+humidity | T=17.344021504673805°C, RH=118.5% | PHYSICS_VIOLATION  | dew_point_le_temp                | Thermodynamic violation: Dew Point (20.06°C) exceeds Dry Bulb Temperature (17.344021504673805°C) |
| 2026-09-02 01:10:00 | humidity             | 118.5                             | PHYSICS_VIOLATION  | rh_physical_bounds               | Relative Humidity (118.5%) out of physical bounds [0%, 102%]                                     |
| 2026-09-02 01:10:00 | humidity             | 118.5                             | SPIKE              | humidity_rate_of_change_spike    | Unphysical abrupt step change / transient spike in humidity (118.5)                              |
| 2026-09-02 18:00:00 | pressure             | nan                               | MISSING            | pressure_missing_or_sentinel     | Missing reading or sentinel value detected in pressure                                           |
| 2026-09-02 18:05:00 | pressure             | nan                               | MISSING            | pressure_missing_or_sentinel     | Missing reading or sentinel value detected in pressure                                           |
| 2026-09-02 18:10:00 | pressure             | nan                               | MISSING            | pressure_missing_or_sentinel     | Missing reading or sentinel value detected in pressure                                           |
| 2026-09-02 18:15:00 | pressure             | nan                               | MISSING            | pressure_missing_or_sentinel     | Missing reading or sentinel value detected in pressure                                           |
| 2026-09-02 18:20:00 | pressure             | nan                               | MISSING            | pressure_missing_or_sentinel     | Missing reading or sentinel value detected in pressure                                           |
| 2026-09-03 10:45:00 | wind_speed           | -4.5                              | PHYSICS_VIOLATION  | wind_speed_non_negative          | Physically impossible negative wind speed (-4.5 m/s)                                             |
| 2026-09-03 12:55:00 | wind_speed           | 21.565495298981688                | SPIKE              | wind_speed_rate_of_change_spike  | Unphysical abrupt step change / transient spike in wind_speed (21.565495298981688)               |
| 2026-09-04 03:35:00 | humidity             | -12.0                             | PHYSICS_VIOLATION  | rh_physical_bounds               | Relative Humidity (-12.0%) out of physical bounds [0%, 102%]                                     |
| 2026-09-04 03:35:00 | humidity             | -12.0                             | SPIKE              | humidity_rate_of_change_spike    | Unphysical abrupt step change / transient spike in humidity (-12.0)                              |
| 2026-09-04 03:40:00 | humidity             | 90.66904664838034                 | SPIKE              | humidity_rate_of_change_spike    | Unphysical abrupt step change / transient spike in humidity (90.66904664838034)                  |
| 2026-09-04 15:20:00 | temperature          | 44.84842929924851                 | SPIKE              | temperature_rate_of_change_spike | Unphysical abrupt step change / transient spike in temperature (44.84842929924851)               |
| 2026-09-04 15:25:00 | temperature          | 30.76647889319906                 | SPIKE              | temperature_rate_of_change_spike | Unphysical abrupt step change / transient spike in temperature (30.76647889319906)               |
| 2026-09-05 04:45:00 | pressure             | 1033.0745471230825                | SPIKE              | pressure_rate_of_change_spike    | Unphysical abrupt step change / transient spike in pressure (1033.0745471230825)                 |
| 2026-09-05 04:50:00 | pressure             | 1023.5445508977492                | SPIKE              | pressure_rate_of_change_spike    | Unphysical abrupt step change / transient spike in pressure (1023.5445508977492)                 |
| 2026-09-05 13:10:00 | wind_speed           | 108.0                             | PHYSICS_VIOLATION  | wind_speed_max_bounds            | Wind speed (108.0 m/s) exceeds typical 50 m/s bound                                              |
| 2026-09-05 13:10:00 | wind_speed           | 108.0                             | SPIKE              | wind_speed_rate_of_change_spike  | Unphysical abrupt step change / transient spike in wind_speed (108.0)                            |

*(Showing first 25 of 47 total anomaly instances)*
