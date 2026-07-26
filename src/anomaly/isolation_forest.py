"""
Isolation Forest baseline - GarageMind M2 Anomaly Detection
Classical (non-deep) baseline for the anomaly benchmark, using the SAME
windows, split and labels as the LSTM autoencoder so results are directly
comparable. Only the model family changes.

Why a classical baseline belongs in the benchmark:
    Deep models must justify their complexity. Isolation Forest gives the
    reference point: any sequence model that cannot beat a forest on
    aggregated statistics is not exploiting temporal structure.

How windows become tabular:
    Isolation Forest handles vectors, not sequences. Each window
    (window_size, n_features) is aggregated into per-feature statistics
    (mean, std, min, max) -> a (4 * n_features,) vector. Temporal ORDER
    inside the window is deliberately lost: that is the point of the
    baseline, it measures how much of the signal is captured by
    distributional statistics alone.

Protocol (identical to the AE):
    - fit on NORMAL training windows only (contamination is irrelevant to
      the fit set, but the decision threshold uses score_samples so the
      contamination parameter is bypassed entirely)
    - anomaly score = -score_samples (higher = more anomalous)
    - threshold = percentile of normal training scores (same label-free
      calibration as the AE's percentile rule, upper side only: isolation
      scores are monotone in abnormality, unlike reconstruction errors)
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def aggregate_windows(X: np.ndarray) -> np.ndarray:
    """
    Aggregate sequence windows into tabular statistics.

    Input:  X (n_windows, window_size, n_features)
    Output: (n_windows, 4 * n_features) float32
            [means | stds | mins | maxs] per feature.
    """
    if X.ndim != 3:
        raise ValueError(f"expected 3D windows array, got shape {X.shape}")
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    mins = X.min(axis=1)
    maxs = X.max(axis=1)
    return np.hstack([means, stds, mins, maxs]).astype(np.float32)


def fit_isolation_forest(
    X_train_normal: np.ndarray,
    n_estimators: int = 200,
    seed: int = 42,
) -> IsolationForest:
    """
    Fit an Isolation Forest on aggregated NORMAL windows only.

    contamination is set to "auto" but plays no role in our pipeline:
    decisions use score_samples with our own percentile threshold.
    """
    features = aggregate_windows(X_train_normal)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features)
    return model


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Anomaly scores for windows: higher = more anomalous.

    sklearn's score_samples returns higher = more normal, so we negate it.
    """
    features = aggregate_windows(X)
    return (-model.score_samples(features)).astype(np.float32)


def calibrate_threshold(scores_normal: np.ndarray, percentile: float = 98.0) -> float:
    """
    Upper threshold = percentile of normal training scores.

    Default 98.0 keeps the same ~2% false-alarm budget on normal traffic
    as the AE's bilateral P1/P99 band, making error budgets comparable.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    return float(np.percentile(scores_normal, percentile))


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions: 1 = attack (score above threshold)."""
    return (scores > threshold).astype(np.int64)