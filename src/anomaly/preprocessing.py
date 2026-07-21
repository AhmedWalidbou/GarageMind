"""
Preprocessing - GarageMind M2 Anomaly Detection
Transforms parsed CAN frames into fixed-size sliding windows usable by
sequence models (LSTM autoencoder, transformer-based detectors).

Pipeline:
    frames (from data_loader) -> per-frame features -> temporal split -> windows

Design decisions (documented for reproducibility):
    - inter_arrival uses log1p: raw gaps span several orders of magnitude
      (sub-millisecond under DoS flood, seconds during idle). Log compresses
      this range so it does not dominate other features.
    - Train/test split is TEMPORAL and done on frames BEFORE windowing.
      Overlapping windows (stride < window_size) would leak identical
      frames into both sets under a random split.
    - LSTM autoencoders train on normal traffic only; extract_normal_windows
      provides that subset, while the test set keeps attacks for evaluation.
    - A window is labeled attack if more than attack_threshold of its frames
      are attacks (default 10%): a DoS flood saturates windows far above
      this, while isolated frames do not flip a window.
"""

import numpy as np
import pandas as pd


def add_frame_features(df: pd.DataFrame, id_freq_window: int = 1000) -> pd.DataFrame:
    """
    Add per-frame numerical features to a parsed CAN DataFrame.

    Input columns required: timestamp, can_id, dlc, data, label
    Added columns:
        inter_arrival: log1p of the time gap with the previous frame
        id_freq: rolling frequency of this CAN ID over the last
                 id_freq_window frames (spoofed or flooded IDs spike)
    """
    out = df.copy()
    raw_gap = out["timestamp"].diff().fillna(0.0).clip(lower=0.0)
    out["inter_arrival"] = np.log1p(raw_gap)

    id_codes = out["can_id"].astype("category").cat.codes
    freq = np.zeros(len(out), dtype=np.float64)
    counts: dict[int, int] = {}
    history: list[int] = []
    for i, code in enumerate(id_codes):
        history.append(code)
        counts[code] = counts.get(code, 0) + 1
        if len(history) > id_freq_window:
            old = history.pop(0)
            counts[old] -= 1
        freq[i] = counts[code] / len(history)
    out["id_freq"] = freq
    return out


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split frames chronologically: first train_ratio for train, rest for test.

    Done BEFORE windowing so no frame appears in both sets (overlapping
    windows would otherwise leak data across the split).
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    cut = int(len(df) * train_ratio)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build the per-frame feature matrix (n_frames, 11).

    Features: inter_arrival, id_freq, dlc/8, 8 data bytes /255.
    """
    inter = df["inter_arrival"].to_numpy(dtype=np.float32).reshape(-1, 1)
    idf = df["id_freq"].to_numpy(dtype=np.float32).reshape(-1, 1)
    dlc = (df["dlc"].to_numpy(dtype=np.float32) / 8.0).reshape(-1, 1)
    data = np.stack(df["data"].to_numpy()).astype(np.float32) / 255.0
    return np.hstack([inter, idf, dlc, data])


def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
    attack_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice the feature matrix into overlapping windows.

    Returns:
        X: (n_windows, window_size, n_features) float32
        y: (n_windows,) int64, 1 if attack ratio in window > attack_threshold
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    n = len(features)
    if n < window_size:
        return (
            np.empty((0, window_size, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    starts = range(0, n - window_size + 1, stride)
    X = np.stack([features[s:s + window_size] for s in starts]).astype(np.float32)
    y = np.array(
        [1 if labels[s:s + window_size].mean() > attack_threshold else 0 for s in starts],
        dtype=np.int64,
    )
    return X, y


def extract_normal_windows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Keep only normal windows (label 0), for autoencoder training.

    An autoencoder learns to reconstruct normal traffic; attacks are then
    detected at inference by their high reconstruction error.
    """
    return X[y == 0]


def prepare_datasets(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    window_size: int = 64,
    stride: int = 32,
    attack_threshold: float = 0.1,
) -> dict:
    """
    Full pipeline: parsed frames -> temporal split -> windowed datasets.

    Returns a dict with:
        X_train, y_train: all training windows (with labels)
        X_train_normal: normal-only training windows (autoencoder input)
        X_test, y_test: test windows for evaluation
    """
    enriched = add_frame_features(df)
    train_df, test_df = temporal_split(enriched, train_ratio)

    X_train, y_train = make_windows(
        build_feature_matrix(train_df),
        train_df["label"].to_numpy(dtype=np.int64),
        window_size, stride, attack_threshold,
    )
    X_test, y_test = make_windows(
        build_feature_matrix(test_df),
        test_df["label"].to_numpy(dtype=np.int64),
        window_size, stride, attack_threshold,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_train_normal": extract_normal_windows(X_train, y_train),
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    from src.anomaly.data_loader import load_can_dataset

    PATH = "data/raw/DoS_dataset.csv"
    print("Loading 200000 frames...")
    df = load_can_dataset(PATH, nrows=200000)

    print("Building datasets (train 70% / test 30%, window=64, stride=32)...")
    ds = prepare_datasets(df)

    print(f"\nX_train        : {ds['X_train'].shape}")
    print(f"X_train_normal : {ds['X_train_normal'].shape}")
    print(f"X_test         : {ds['X_test'].shape}")
    print(f"Attack windows train: {int(ds['y_train'].sum())} / {len(ds['y_train'])} "
          f"({100 * ds['y_train'].mean():.2f}%)")
    print(f"Attack windows test : {int(ds['y_test'].sum())} / {len(ds['y_test'])} "
          f"({100 * ds['y_test'].mean():.2f}%)")
    print(f"Feature range  : min={ds['X_train'].min():.4f} max={ds['X_train'].max():.4f}")