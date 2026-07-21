"""
LSTM Autoencoder - GarageMind M2 Anomaly Detection
Sequence autoencoder for CAN traffic windows, trained on normal traffic only.

Detection principle:
    The encoder compresses a window (window_size, n_features) into a latent
    vector; the decoder reconstructs the window from it. Trained exclusively
    on normal traffic, the per-window reconstruction error (MSE) is used as
    the anomaly score.

Why detection is BILATERAL (failure analysis, v1 -> v2):
    The naive rule "attack = high reconstruction error" scored ROC-AUC 0.056
    on the HCRL DoS dataset: near-perfectly inverted ranking. Root cause: a
    DoS flood hammers one CAN ID with constant payloads at near-zero
    inter-arrival, producing sequences MORE regular than normal traffic and
    therefore EASIER to reconstruct than the varied normal bus traffic.
    An anomaly is any window whose error leaves the normal distribution in
    EITHER direction: too high (unseen patterns) or too low (unnaturally
    regular traffic such as floods). Both bounds are calibrated as
    percentiles of the normal training scores.

Threshold calibration:
    low = low_percentile of normal scores (default 1st)
    high = high_percentile of normal scores (default 99th)
    Total false-alarm budget on normal traffic: ~2%. Calibration remains
    label-free: only normal training windows are used.

Design decisions:
    - The decoder receives the latent vector repeated at each timestep
      (standard seq2seq AE without teacher forcing), so reconstruction
      relies entirely on the latent representation.
    - Errors are averaged per window (mean over timesteps and features),
      giving one scalar score per window.
    - Seeds are fixed for reproducibility of the benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMAutoencoder(nn.Module):
    """
    Seq2seq LSTM autoencoder for fixed-length CAN windows.

    Args:
        n_features: features per timestep (11 from preprocessing)
        hidden_size: LSTM hidden dimension
        latent_size: bottleneck dimension
        num_layers: stacked LSTM layers in encoder and decoder
    """

    def __init__(
        self,
        n_features: int = 11,
        hidden_size: int = 64,
        latent_size: int = 16,
        num_layers: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_size, n_features) -> reconstruction, same shape.
        """
        window_size = x.shape[1]
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])
        dec_input = self.from_latent(latent).unsqueeze(1).repeat(1, window_size, 1)
        dec_out, _ = self.decoder(dec_input)
        return self.output_layer(dec_out)


def reconstruction_errors(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Compute the per-window mean squared reconstruction error.

    Returns: (n_windows,) float32 array of anomaly scores.
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False
    )
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def train_autoencoder(
    model: LSTMAutoencoder,
    X_train_normal: np.ndarray,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """
    Train the autoencoder on normal windows only.

    Returns the list of mean training losses per epoch.
    """
    model.to(device)
    model.train()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
        epoch_loss = total / count
        history.append(epoch_loss)
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} - loss {epoch_loss:.6f}")
    return history


def calibrate_threshold(scores_normal: np.ndarray, percentile: float = 99.0) -> float:
    """Single upper threshold = given percentile of normal errors (v1 rule)."""
    return float(np.percentile(scores_normal, percentile))


def calibrate_thresholds(
    scores_normal: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    """
    Bilateral thresholds from normal reconstruction errors (v2 rule).

    Returns (low, high). A window is anomalous if its score falls outside
    [low, high]. See module docstring for the failure analysis motivating
    the lower bound.
    """
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError(
            f"need 0 <= low < high <= 100, got ({low_percentile}, {high_percentile})"
        )
    low = float(np.percentile(scores_normal, low_percentile))
    high = float(np.percentile(scores_normal, high_percentile))
    return low, high


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions, upper threshold only (v1): 1 = attack."""
    return (scores > threshold).astype(np.int64)


def predict_bilateral(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Binary predictions with bilateral thresholds (v2): 1 = attack.

    A window is flagged if its reconstruction error is below `low`
    (unnaturally regular traffic, e.g. DoS flood) or above `high`
    (unseen/irregular patterns).
    """
    return ((scores < low) | (scores > high)).astype(np.int64)