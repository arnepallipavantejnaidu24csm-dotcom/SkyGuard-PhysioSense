"""
State-Space Kalman Filter for True Atmospheric State and Sensor Bias Tracking.

Implements a continuous-discrete Kalman Filter that:
1. Separates true atmospheric states (value + rate of change) from instrument bias
2. Continuously estimates sensor drift/bias as new readings arrive
3. Handles missing observations and variable sampling intervals (dt)
4. Computes pre-fit innovations (y_k), post-fit residuals (e_k), and Normalized Innovation Squared (NIS)
5. Computes dynamic uncertainty bounds (P covariance diagonals, 2-sigma / 95% confidence intervals)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field


@dataclass
class KalmanFilterConfig:
    """Configuration and tuning parameters for the Atmospheric State Kalman Filter."""
    # Variable channels to track
    channels: List[str] = field(default_factory=lambda: ["temperature", "pressure", "humidity", "wind_speed"])
    
    # Process noise standard deviations (true weather dynamics variability per sqrt(time))
    process_noise_weather: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.05,   # °C / min^(1.5)
        "pressure": 0.03,      # hPa / min^(1.5)
        "humidity": 0.15,      # % / min^(1.5)
        "wind_speed": 0.20,    # m/s / min^(1.5)
    })
    
    # Process noise standard deviations for sensor drift (slow Brownian diffusion)
    process_noise_bias: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.002,  # °C / sqrt(min) -> slow instrument drift
        "pressure": 0.001,     # hPa / sqrt(min)
        "humidity": 0.005,     # % / sqrt(min)
        "wind_speed": 0.003,   # m/s / sqrt(min)
    })
    
    # Measurement noise standard deviations (sensor precision / white noise)
    measurement_noise: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.3,    # °C sensor precision
        "pressure": 0.2,       # hPa sensor precision
        "humidity": 1.5,       # % sensor precision
        "wind_speed": 0.5,     # m/s sensor precision
    })
    
    # Initial state standard deviations (prior uncertainty)
    initial_std_value: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 5.0,
        "pressure": 10.0,
        "humidity": 15.0,
        "wind_speed": 5.0,
    })
    initial_std_rate: float = 0.1
    initial_std_bias: float = 1.0


@dataclass
class KalmanOutput:
    """Output results from the Kalman Filter estimation."""
    state_estimates_df: pd.DataFrame
    bias_estimates_df: pd.DataFrame
    residuals_df: pd.DataFrame
    uncertainty_df: pd.DataFrame
    full_output_df: pd.DataFrame
    summary_metrics: Dict[str, Any] = field(default_factory=dict)


class AtmosphericStateKalmanFilter:
    r"""
    State-Space Kalman Filter partitioning the state into:
    1. True physical weather states [x_1, \dot{x}_1, x_2, \dot{x}_2, ...]
    2. Sensor instrument biases [b_1, b_2, ...]
    
    State Vector Structure (for N channels):
    x = [
      x_true_1, \dot{x}_true_1,   # e.g., Temperature value and rate of change
      x_true_2, \dot{x}_true_2,   # e.g., Pressure value and rate of change
      ...
      b_1,                        # Sensor 1 bias
      b_2,                        # Sensor 2 bias
      ...
    ]^T
    
    Observation Model:
    z_i = x_true_i + b_i + v_i,   v_i ~ N(0, R_i)
    """

    def __init__(self, config: Optional[KalmanFilterConfig] = None):
        self.config = config or KalmanFilterConfig()
        self.channels = self.config.channels
        self.num_channels = len(self.channels)

        # State dimensions:
        # Each channel has 2 weather states (value, rate) + 1 bias state
        self.weather_dim = 2 * self.num_channels
        self.bias_dim = self.num_channels
        self.dim_x = self.weather_dim + self.bias_dim
        self.dim_z = self.num_channels

        # Initialize state mean x and covariance P
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64)
        
        # Track initialization status
        self.is_initialized = False

    def _get_state_indices(self, channel_idx: int) -> Tuple[int, int, int]:
        """Return indices for (true_value, rate_of_change, bias) of a channel."""
        val_idx = 2 * channel_idx
        rate_idx = 2 * channel_idx + 1
        bias_idx = self.weather_dim + channel_idx
        return val_idx, rate_idx, bias_idx

    def initialize_state(
        self,
        initial_observations: Dict[str, float],
        initial_biases: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize state vector and covariance matrix from initial sensor observations.
        """
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64)

        if initial_biases is None:
            initial_biases = {ch: 0.0 for ch in self.channels}

        for i, ch in enumerate(self.channels):
            val_idx, rate_idx, bias_idx = self._get_state_indices(i)
            
            # Initial bias estimate
            b_init = initial_biases.get(ch, 0.0)
            self.x[bias_idx, 0] = b_init
            
            # Initial true value = observed - initial_bias
            obs_val = initial_observations.get(ch, np.nan)
            if not np.isnan(obs_val):
                self.x[val_idx, 0] = obs_val - b_init
            else:
                self.x[val_idx, 0] = 0.0
            
            # Initial rate = 0
            self.x[rate_idx, 0] = 0.0

            # Initial covariance
            std_val = self.config.initial_std_value.get(ch, 5.0)
            self.P[val_idx, val_idx] = std_val ** 2
            self.P[rate_idx, rate_idx] = self.config.initial_std_rate ** 2
            self.P[bias_idx, bias_idx] = self.config.initial_std_bias ** 2

        self.is_initialized = True

    def build_transition_matrix(self, dt: float) -> np.ndarray:
        r"""
        Construct state transition matrix F for time step dt.
        
        Weather state kinematics:
          x_{k} = x_{k-1} + \dot{x}_{k-1} * dt
          \dot{x}_{k} = \dot{x}_{k-1}
        
        Bias state random walk:
          b_{k} = b_{k-1}
        """
        F = np.eye(self.dim_x, dtype=np.float64)
        for i in range(self.num_channels):
            val_idx, rate_idx, _ = self._get_state_indices(i)
            F[val_idx, rate_idx] = dt
        return F

    def build_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Construct discrete process noise covariance matrix Q for time step dt.
        
        Uses Continuous White Noise Acceleration (CWNA) model for weather dynamics:
          Q_weather = [ [dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt] ] * q_w
        
        And discrete random walk diffusion for sensor bias:
          Q_bias = dt * q_b
        """
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt

        for i, ch in enumerate(self.channels):
            val_idx, rate_idx, bias_idx = self._get_state_indices(i)
            
            # Weather process noise intensity
            q_w = self.config.process_noise_weather.get(ch, 0.1) ** 2
            Q[val_idx, val_idx] = (dt3 / 3.0) * q_w
            Q[val_idx, rate_idx] = (dt2 / 2.0) * q_w
            Q[rate_idx, val_idx] = (dt2 / 2.0) * q_w
            Q[rate_idx, rate_idx] = dt * q_w

            # Sensor bias process noise intensity (slow drift)
            q_b = self.config.process_noise_bias.get(ch, 0.001) ** 2
            Q[bias_idx, bias_idx] = dt * q_b

        return Q

    def build_observation_matrix(self) -> np.ndarray:
        """
        Construct observation matrix H mapping state to measurements.
        z_i = x_true_i + b_i
        """
        H = np.zeros((self.num_channels, self.dim_x), dtype=np.float64)
        for i in range(self.num_channels):
            val_idx, _, bias_idx = self._get_state_indices(i)
            H[i, val_idx] = 1.0
            H[i, bias_idx] = 1.0
        return H

    def build_measurement_noise_matrix(self) -> np.ndarray:
        """
        Construct measurement noise covariance matrix R.
        """
        R = np.zeros((self.num_channels, self.num_channels), dtype=np.float64)
        for i, ch in enumerate(self.channels):
            std_r = self.config.measurement_noise.get(ch, 0.5)
            R[i, i] = std_r ** 2
        return R

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman Filter Prediction Step (Time Update).
        
        x_{k|k-1} = F * x_{k-1|k-1}
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        F = self.build_transition_matrix(dt)
        Q = self.build_process_noise_matrix(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def update(
        self,
        z_dict: Dict[str, float],
        reference_constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Kalman Filter Measurement Update Step (Correction).
        
        Handles:
        1. Regular sensor observations (z = x_true + b + v)
        2. Missing values / NaNs (partial observation update)
        3. Optional reference ground truth calibration checkpoints (anchoring bias separation)
        
        Returns:
            Dictionary containing innovations (pre-fit residuals), post-fit residuals, NIS, and Kalman gains.
        """
        # Build full observation vector
        z_full = np.zeros((self.num_channels, 1), dtype=np.float64)
        valid_indices = []

        for i, ch in enumerate(self.channels):
            val = z_dict.get(ch, np.nan)
            if not np.isnan(val) and not np.isinf(val):
                z_full[i, 0] = val
                valid_indices.append(i)

        if not valid_indices and (reference_constraints is None or not reference_constraints):
            # No valid measurements available this step: skip measurement update
            return {
                "innovations": {ch: np.nan for ch in self.channels},
                "post_fit_residuals": {ch: np.nan for ch in self.channels},
                "nis": np.nan,
                "measurement_updated": False,
            }

        H_full = self.build_observation_matrix()
        R_full = self.build_measurement_noise_matrix()

        # Extract active sub-matrices for available observations
        H_sub = H_full[valid_indices, :]
        R_sub = R_full[np.ix_(valid_indices, valid_indices)]
        z_sub = z_full[valid_indices, :]

        # Optional reference calibration constraints (e.g. Ground Truth Reference Sensor where b=0)
        if reference_constraints:
            for ch, ref_val in reference_constraints.items():
                if ch in self.channels and not np.isnan(ref_val):
                    ch_idx = self.channels.index(ch)
                    val_idx, _, _ = self._get_state_indices(ch_idx)
                    
                    # Row measuring ONLY true value (without instrument bias)
                    H_ref_row = np.zeros((1, self.dim_x), dtype=np.float64)
                    H_ref_row[0, val_idx] = 1.0
                    
                    R_ref_row = np.array([[ (0.05) ** 2 ]])  # Highly accurate reference
                    z_ref_row = np.array([[ ref_val ]])

                    if len(valid_indices) == 0:
                        H_sub = H_ref_row
                        R_sub = R_ref_row
                        z_sub = z_ref_row
                    else:
                        H_sub = np.vstack([H_sub, H_ref_row])
                        R_sub = scipy_block_diag(R_sub, R_ref_row)
                        z_sub = np.vstack([z_sub, z_ref_row])

        # 1. Innovation (Pre-fit residual): y = z - H * x_{k|k-1}
        y = z_sub - H_sub @ self.x

        # 2. Innovation Covariance: S = H * P * H^T + R
        S = H_sub @ self.P @ H_sub.T + R_sub

        # 3. Kalman Gain: K = P * H^T * S^-1
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H_sub.T @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = self.P @ H_sub.T @ S_inv

        # 4. State Update: x_{k|k} = x_{k|k-1} + K * y
        self.x = self.x + K @ y

        # 5. Covariance Update (Joseph stabilized form for numerical symmetry and positive definiteness):
        # P_{k|k} = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(self.dim_x) - K @ H_sub
        self.P = I_KH @ self.P @ I_KH.T + K @ R_sub @ K.T
        # Force numerical symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # 6. Post-fit Residuals: e = z - H * x_{k|k}
        e = z_sub - H_sub @ self.x

        # 7. Normalized Innovation Squared (NIS): y^T * S^-1 * y
        nis = float((y.T @ S_inv @ y).item())

        # Map innovations & post-fit residuals back to named channels
        innovations_dict = {ch: np.nan for ch in self.channels}
        post_fit_dict = {ch: np.nan for ch in self.channels}

        for k, idx in enumerate(valid_indices):
            ch_name = self.channels[idx]
            innovations_dict[ch_name] = float(y[k, 0])
            post_fit_dict[ch_name] = float(e[k, 0])

        return {
            "innovations": innovations_dict,
            "post_fit_residuals": post_fit_dict,
            "nis": nis,
            "measurement_updated": True,
        }

    def get_current_estimates(self) -> Dict[str, Any]:
        """
        Extract current estimated states, rates, biases, and standard deviations (uncertainties).
        """
        result = {}
        for i, ch in enumerate(self.channels):
            val_idx, rate_idx, bias_idx = self._get_state_indices(i)
            
            val = float(self.x[val_idx, 0])
            rate = float(self.x[rate_idx, 0])
            bias = float(self.x[bias_idx, 0])
            
            std_val = float(np.sqrt(max(0.0, self.P[val_idx, val_idx])))
            std_rate = float(np.sqrt(max(0.0, self.P[rate_idx, rate_idx])))
            std_bias = float(np.sqrt(max(0.0, self.P[bias_idx, bias_idx])))

            result[f"{ch}_true"] = val
            result[f"{ch}_rate"] = rate
            result[f"{ch}_bias"] = bias
            result[f"{ch}_true_std"] = std_val
            result[f"{ch}_bias_std"] = std_bias
            result[f"{ch}_true_ci95_lower"] = val - 1.96 * std_val
            result[f"{ch}_true_ci95_upper"] = val + 1.96 * std_val
            result[f"{ch}_bias_ci95_lower"] = bias - 1.96 * std_bias
            result[f"{ch}_bias_ci95_upper"] = bias + 1.96 * std_bias

        return result

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        dt_default_minutes: float = 5.0,
        reference_df: Optional[pd.DataFrame] = None
    ) -> KalmanOutput:
        """
        Run recursive Kalman Filter across an entire time-series DataFrame.

        Args:
            df: DataFrame containing sensor observations.
            timestamp_col: Timestamp column name.
            dt_default_minutes: Fallback delta time if timestamps are not provided.
            reference_df: Optional DataFrame with ground-truth reference calibration readings.

        Returns:
            KalmanOutput with state estimates, biases, residuals, and uncertainty bounds.
        """
        n_rows = len(df)
        records = []

        # Extract timestamps
        has_timestamps = timestamp_col in df.columns
        if has_timestamps:
            ts_series = pd.to_datetime(df[timestamp_col])
        else:
            ts_series = pd.date_range(start="2026-01-01", periods=n_rows, freq=f"{int(dt_default_minutes)}min")

        # Initialize filter with first row
        first_obs = {ch: df[ch].iloc[0] if ch in df.columns else np.nan for ch in self.channels}
        self.initialize_state(first_obs)

        prev_time = ts_series.iloc[0] if n_rows > 0 else None

        for k in range(n_rows):
            curr_time = ts_series.iloc[k]
            
            # Compute dt (in minutes)
            if k == 0:
                dt_minutes = dt_default_minutes
            else:
                dt_seconds = (curr_time - prev_time).total_seconds()
                dt_minutes = max(0.1, dt_seconds / 60.0) if dt_seconds > 0 else dt_default_minutes
            
            prev_time = curr_time

            # 1. Prediction step
            self.predict(dt=dt_minutes)

            # 2. Extract current row observation
            z_dict = {ch: df[ch].iloc[k] if ch in df.columns else np.nan for ch in self.channels}

            # Optional reference constraints
            ref_constraints = None
            if reference_df is not None and k < len(reference_df):
                ref_constraints = {
                    ch: reference_df[ch].iloc[k] for ch in self.channels if ch in reference_df.columns
                }

            # 3. Measurement Update step
            update_res = self.update(z_dict=z_dict, reference_constraints=ref_constraints)

            # 4. Collect estimates and metrics
            estimates = self.get_current_estimates()
            
            row_record = {
                "timestamp": curr_time,
                **{f"{ch}_observed": z_dict[ch] for ch in self.channels},
                **estimates,
                **{f"{ch}_innovation": update_res["innovations"][ch] for ch in self.channels},
                **{f"{ch}_post_fit_residual": update_res["post_fit_residuals"][ch] for ch in self.channels},
                "nis": update_res["nis"],
            }
            records.append(row_record)

        full_df = pd.DataFrame(records)

        # Build clean partitioned sub-dataframes
        state_cols = ["timestamp"] + [f"{ch}_true" for ch in self.channels] + [f"{ch}_rate" for ch in self.channels]
        bias_cols = ["timestamp"] + [f"{ch}_bias" for ch in self.channels] + [f"{ch}_bias_std" for ch in self.channels]
        res_cols = ["timestamp"] + [f"{ch}_innovation" for ch in self.channels] + [f"{ch}_post_fit_residual" for ch in self.channels] + ["nis"]
        unc_cols = ["timestamp"] + [f"{ch}_true_std" for ch in self.channels] + [f"{ch}_bias_std" for ch in self.channels]

        state_estimates_df = full_df[[c for c in state_cols if c in full_df.columns]].copy()
        bias_estimates_df = full_df[[c for c in bias_cols if c in full_df.columns]].copy()
        residuals_df = full_df[[c for c in res_cols if c in full_df.columns]].copy()
        uncertainty_df = full_df[[c for c in unc_cols if c in full_df.columns]].copy()

        # Compute summary metrics
        summary = {}
        for ch in self.channels:
            if f"{ch}_post_fit_residual" in full_df.columns:
                res_series = full_df[f"{ch}_post_fit_residual"].dropna()
                summary[f"{ch}_residual_mean"] = float(res_series.mean()) if not res_series.empty else 0.0
                summary[f"{ch}_residual_std"] = float(res_series.std()) if not res_series.empty else 0.0
                summary[f"{ch}_residual_rmse"] = float(np.sqrt((res_series ** 2).mean())) if not res_series.empty else 0.0
            if f"{ch}_bias" in full_df.columns:
                final_bias = float(full_df[f"{ch}_bias"].iloc[-1])
                summary[f"{ch}_final_bias_est"] = final_bias

        return KalmanOutput(
            state_estimates_df=state_estimates_df,
            bias_estimates_df=bias_estimates_df,
            residuals_df=residuals_df,
            uncertainty_df=uncertainty_df,
            full_output_df=full_df,
            summary_metrics=summary,
        )


def scipy_block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Helper to construct block diagonal matrix from two 2D numpy arrays."""
    rA, cA = A.shape
    rB, cB = B.shape
    out = np.zeros((rA + rB, cA + cB), dtype=np.float64)
    out[:rA, :cA] = A
    out[rA:, cA:] = B
    return out
