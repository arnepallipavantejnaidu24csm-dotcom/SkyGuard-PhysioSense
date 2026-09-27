# 🛰️ Atmospheric State-Space Kalman Filter & Sensor Bias Tracking Report

## 1. Formulation & Architecture
- **State Vector**: Partitioned into True Atmospheric State $[x_w, \dot{x}_w]$ and Sensor Biases $[b_s]$.
- **State Dynamics**: Continuous white-noise acceleration kinematic model for atmospheric variables and random-walk diffusion for instrument drift.
- **Measurement Model**: $\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k$, where $z_i = x_{true, i} + b_i + v_i$.
- **Residual Tracking**: Computes pre-fit innovations $\mathbf{y}_k = \mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1}$, post-fit residuals $\mathbf{e}_k = \mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k}$, and Normalized Innovation Squared (NIS).

## 2. Benchmark Scenario Performance

### Linear Ramp Sensor Drift

| Sensor Channel   |   Raw Obs RMSE |   KF True State RMSE | RMSE Reduction   |   True Final Bias |   Estimated Bias |   Bias Tracking RMSE |   Residual Mean |   Residual Std |
|:-----------------|---------------:|---------------------:|:-----------------|------------------:|-----------------:|---------------------:|----------------:|---------------:|
| Temperature      |          0.755 |                0.534 | 29.4%            |              1.25 |            0.647 |                0.486 |           0.001 |          0.129 |
| Pressure         |          1.165 |                0.49  | 57.9%            |             -2    |           -1.25  |                0.471 |          -0.002 |          0.084 |
| Humidity         |          2.41  |                1.253 | 48.0%            |              4    |            2.717 |                0.89  |           0.003 |          0.639 |
| Wind_speed       |          0.525 |                0.772 | -47.0%           |              0.75 |           -0.281 |                0.651 |           0     |          0.314 |

### Sudden Calibration Step Offset

| Sensor Channel   |   Raw Obs RMSE |   KF True State RMSE | RMSE Reduction   |   True Final Bias |   Estimated Bias |   Bias Tracking RMSE |   Residual Mean |   Residual Std |
|:-----------------|---------------:|---------------------:|:-----------------|------------------:|-----------------:|---------------------:|----------------:|---------------:|
| Temperature      |          1.692 |                1.025 | 39.4%            |               2.2 |            1.665 |                1.004 |           0.002 |          0.135 |
| Pressure         |          2.29  |                1.208 | 47.2%            |              -3   |           -2.4   |                1.205 |          -0.003 |          0.107 |
| Humidity         |          4.986 |                2.86  | 42.6%            |               6.5 |            5.022 |                2.736 |           0.009 |          0.66  |
| Wind_speed       |          0.296 |                0.638 | -115.8%          |               0   |           -0.714 |                0.486 |          -0     |          0.314 |

### Stochastic Brownian Walk Drift

| Sensor Channel   |   Raw Obs RMSE |   KF True State RMSE | RMSE Reduction   |   True Final Bias |   Estimated Bias |   Bias Tracking RMSE |   Residual Mean |   Residual Std |
|:-----------------|---------------:|---------------------:|:-----------------|------------------:|-----------------:|---------------------:|----------------:|---------------:|
| Temperature      |          0.237 |                0.301 | -26.6%           |             0.241 |           -0.018 |                0.205 |           0     |          0.129 |
| Pressure         |          0.263 |                0.165 | 37.6%            |             0.11  |            0.28  |                0.096 |           0     |          0.082 |
| Humidity         |          1.124 |                1.232 | -9.5%            |            -1.528 |           -0.16  |                0.859 |          -0.002 |          0.656 |
| Wind_speed       |          0.315 |                0.662 | -110.5%          |             0.06  |           -0.659 |                0.506 |          -0     |          0.311 |

