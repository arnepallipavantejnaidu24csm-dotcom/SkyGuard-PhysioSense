"""
PyTorch LSTM Autoencoder for Time-Series Sequence Pattern Recognition and Reconstruction Anomaly Detection.

Implements:
1. Multi-layer LSTM Encoder-Decoder architecture for temporal sequence encoding
2. Sliding window sequence transformation
3. Reconstruction error (MSE / MAE) calculation across time windows
4. Dynamic error thresholding and calibrated anomaly confidence scoring
5. Channel-specific reconstruction attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


@dataclass
class LSTMAutoencoderResult:
    """Output from LSTM Autoencoder evaluation."""
    is_anomaly: pd.Series
    reconstruction_error: pd.Series
    anomaly_confidence: pd.Series
    channel_errors_df: pd.DataFrame
    threshold: float


class LSTMEncoder(nn.Module):
    """LSTM Encoder sub-network."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state: shape (batch_size, hidden_dim)
        last_hidden = h_n[-1]
        latent = self.fc_latent(last_hidden)  # (batch_size, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """LSTM Decoder sub-network."""
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent shape: (batch_size, latent_dim)
        hidden = self.fc_expand(latent)  # (batch_size, hidden_dim)
        # Repeat vector over sequence length
        repeated = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(repeated)
        recon = self.fc_out(lstm_out)  # (batch_size, seq_len, output_dim)
        return recon


class LSTMAutoencoderNet(nn.Module):
    """Full LSTM Autoencoder Network."""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 32, latent_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class AWSLSTMAutoencoderDetector:
    """
    High-level LSTM Autoencoder detector for AWS weather time-series.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        seq_len: int = 12,           # e.g., 12 steps (1 hour at 5-min intervals)
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.003,
        threshold_std_multiplier: float = 3.0,
        device: Optional[str] = None,
    ):
        self.channels = channels or ["temperature", "pressure", "humidity", "wind_speed"]
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.threshold_multiplier = threshold_std_multiplier

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scaler = StandardScaler()
        self.model: Optional[LSTMAutoencoderNet] = None
        self.threshold: float = 0.0
        self.baseline_mean_err: float = 0.0
        self.baseline_std_err: float = 1.0
        self.is_fitted = False

    def _create_sequences(self, data_arr: np.ndarray) -> np.ndarray:
        """Create sliding window sequences of shape (num_windows, seq_len, num_channels)."""
        num_samples = len(data_arr)
        if num_samples < self.seq_len:
            raise ValueError(f"Data length ({num_samples}) is shorter than sequence window ({self.seq_len})")
        
        windows = []
        for i in range(num_samples - self.seq_len + 1):
            windows.append(data_arr[i : i + self.seq_len])
        return np.array(windows, dtype=np.float32)

    def fit(self, normal_df: pd.DataFrame) -> "AWSLSTMAutoencoderDetector":
        """
        Train LSTM Autoencoder on normal baseline operating patterns.
        """
        valid_cols = [c for c in self.channels if c in normal_df.columns]
        data_clean = normal_df[valid_cols].ffill().bfill().values
        
        # Fit scaler
        scaled_data = self.scaler.fit_transform(data_clean)
        
        # Create sequences
        seqs = self._create_sequences(scaled_data)
        
        input_dim = len(valid_cols)
        self.model = LSTMAutoencoderNet(
            input_dim=input_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(seqs, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

        # Compute baseline reconstruction error statistics on training data
        self.model.eval()
        with torch.no_grad():
            tensor_seqs = torch.tensor(seqs, dtype=torch.float32).to(self.device)
            recons = self.model(tensor_seqs).cpu().numpy()
            
            # Pointwise MSE across the last time step of each window
            train_errors = np.mean((seqs[:, -1, :] - recons[:, -1, :]) ** 2, axis=1)
            self.baseline_mean_err = float(np.mean(train_errors))
            self.baseline_std_err = float(np.std(train_errors))
            if self.baseline_std_err < 1e-6:
                self.baseline_std_err = 1e-6
            self.threshold = self.baseline_mean_err + self.threshold_multiplier * self.baseline_std_err

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> LSTMAutoencoderResult:
        """
        Evaluate temporal reconstruction errors and calculate anomaly confidence.
        """
        if not self.is_fitted:
            self.fit(df)

        valid_cols = [c for c in self.channels if c in df.columns]
        n_rows = len(df)
        
        data_clean = df[valid_cols].ffill().bfill().values
        scaled_data = self.scaler.transform(data_clean)

        seqs = self._create_sequences(scaled_data)
        
        self.model.eval()
        with torch.no_grad():
            tensor_seqs = torch.tensor(seqs, dtype=torch.float32).to(self.device)
            recons = self.model(tensor_seqs).cpu().numpy()
            
            # Window-last-step squared error per channel
            ch_errors_seq = (seqs[:, -1, :] - recons[:, -1, :]) ** 2  # (num_seq, num_channels)
            total_mse_seq = np.mean(ch_errors_seq, axis=1)

        # Pad initial seq_len - 1 positions with baseline error
        pad_len = self.seq_len - 1
        padded_total_err = np.zeros(n_rows, dtype=np.float64)
        padded_total_err[:pad_len] = self.baseline_mean_err
        padded_total_err[pad_len:] = total_mse_seq

        padded_ch_errors = np.zeros((n_rows, len(valid_cols)), dtype=np.float64)
        padded_ch_errors[:pad_len, :] = self.baseline_mean_err / len(valid_cols)
        padded_ch_errors[pad_len:, :] = ch_errors_seq

        # Normalized Z-score of reconstruction error relative to baseline
        z_scores = (padded_total_err - self.baseline_mean_err) / self.baseline_std_err
        
        # Calibrate confidence using sigmoid of (z_score - threshold_multiplier)
        confidence = 1.0 / (1.0 + np.exp(-(z_scores - (self.threshold_multiplier - 1.0)) * 1.5))
        confidence = np.clip(confidence, 0.0, 1.0)

        is_anomaly = padded_total_err > self.threshold

        ch_err_df = pd.DataFrame(
            padded_ch_errors,
            columns=[f"lstm_err_{col}" for col in valid_cols],
            index=df.index
        )

        return LSTMAutoencoderResult(
            is_anomaly=pd.Series(is_anomaly, index=df.index, name="lstm_anomaly"),
            reconstruction_error=pd.Series(padded_total_err, index=df.index, name="lstm_recon_error"),
            anomaly_confidence=pd.Series(confidence, index=df.index, name="lstm_confidence"),
            channel_errors_df=ch_err_df,
            threshold=self.threshold,
        )
