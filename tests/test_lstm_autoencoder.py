"""
Tests for src/anomaly/lstm_autoencoder.py
Covers: architecture shapes, training loss decrease, scoring,
unilateral (v1) and bilateral (v2) threshold calibration and prediction.
The bilateral rule is the fix for the DoS failure analysis: floods are
MORE regular than normal traffic, hence low reconstruction errors.
Uses a tiny model and synthetic data so the suite stays fast.
"""

import numpy as np
import pytest
import torch

from src.anomaly.lstm_autoencoder import (
    LSTMAutoencoder,
    set_seed,
    train_autoencoder,
    reconstruction_errors,
    calibrate_threshold,
    calibrate_thresholds,
    predict,
    predict_bilateral,
)

WINDOW = 16
FEATURES = 11


def tiny_model() -> LSTMAutoencoder:
    return LSTMAutoencoder(n_features=FEATURES, hidden_size=8, latent_size=4)


def normal_windows(n: int) -> np.ndarray:
    """Smooth low-amplitude signals imitating normal traffic."""
    rng = np.random.default_rng(0)
    base = np.sin(np.linspace(0, 3, WINDOW)).reshape(1, WINDOW, 1)
    X = np.tile(base, (n, 1, FEATURES)) * 0.1
    return (X + rng.normal(0, 0.01, X.shape)).astype(np.float32)


def attack_windows(n: int) -> np.ndarray:
    """High-amplitude noise imitating irregular attack traffic."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.8, 1.0, (n, WINDOW, FEATURES)).astype(np.float32)


def flood_windows(n: int) -> np.ndarray:
    """Perfectly constant traffic imitating a DoS flood (trivial to reconstruct)."""
    return np.zeros((n, WINDOW, FEATURES), dtype=np.float32)


class TestArchitecture:
    def test_output_shape_matches_input(self):
        set_seed()
        model = tiny_model()
        x = torch.zeros(4, WINDOW, FEATURES)
        assert model(x).shape == (4, WINDOW, FEATURES)

    def test_works_with_different_window_sizes(self):
        set_seed()
        model = tiny_model()
        for w in [8, 32]:
            x = torch.zeros(2, w, FEATURES)
            assert model(x).shape == (2, w, FEATURES)


class TestTraining:
    def test_loss_decreases(self):
        set_seed()
        model = tiny_model()
        X = normal_windows(64)
        history = train_autoencoder(model, X, epochs=8, batch_size=16, verbose=False)
        assert history[-1] < history[0]

    def test_history_length_equals_epochs(self):
        set_seed()
        model = tiny_model()
        X = normal_windows(32)
        history = train_autoencoder(model, X, epochs=3, batch_size=16, verbose=False)
        assert len(history) == 3


class TestScoring:
    def test_scores_shape(self):
        set_seed()
        model = tiny_model()
        X = normal_windows(20)
        scores = reconstruction_errors(model, X)
        assert scores.shape == (20,)

    def test_attacks_score_higher_after_training(self):
        set_seed()
        model = tiny_model()
        X_normal = normal_windows(128)
        train_autoencoder(model, X_normal, epochs=15, batch_size=16, verbose=False)
        s_normal = reconstruction_errors(model, normal_windows(32))
        s_attack = reconstruction_errors(model, attack_windows(32))
        assert s_attack.mean() > s_normal.mean()


class TestUnilateralThreshold:
    def test_threshold_is_percentile(self):
        scores = np.arange(100, dtype=np.float32)
        assert calibrate_threshold(scores, percentile=99.0) == pytest.approx(98.01)

    def test_predict_binary_output(self):
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        preds = predict(scores, threshold=0.5)
        assert preds.tolist() == [0, 0, 1]


class TestBilateralThresholds:
    def test_returns_ordered_bounds(self):
        scores = np.arange(1000, dtype=np.float32)
        low, high = calibrate_thresholds(scores, low_percentile=1.0, high_percentile=99.0)
        assert low < high
        assert low == pytest.approx(np.percentile(scores, 1.0))
        assert high == pytest.approx(np.percentile(scores, 99.0))

    def test_invalid_percentiles_raise(self):
        scores = np.arange(100, dtype=np.float32)
        with pytest.raises(ValueError):
            calibrate_thresholds(scores, low_percentile=99.0, high_percentile=1.0)
        with pytest.raises(ValueError):
            calibrate_thresholds(scores, low_percentile=-1.0, high_percentile=99.0)

    def test_predict_flags_both_sides(self):
        scores = np.array([0.05, 0.5, 0.95], dtype=np.float32)
        preds = predict_bilateral(scores, low=0.1, high=0.9)
        assert preds.tolist() == [1, 0, 1]

    def test_inside_band_is_normal(self):
        scores = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        preds = predict_bilateral(scores, low=0.1, high=0.9)
        assert preds.tolist() == [0, 0, 0]


class TestEndToEndDetection:
    def test_detects_irregular_and_flood_attacks(self):
        set_seed()
        model = tiny_model()
        X_normal = normal_windows(128)
        train_autoencoder(model, X_normal, epochs=15, batch_size=16, verbose=False)
        s_train = reconstruction_errors(model, X_normal)
        low, high = calibrate_thresholds(s_train)

        preds_attack = predict_bilateral(
            reconstruction_errors(model, attack_windows(32)), low, high
        )
        preds_flood = predict_bilateral(
            reconstruction_errors(model, flood_windows(32)), low, high
        )
        preds_normal = predict_bilateral(
            reconstruction_errors(model, normal_windows(64)), low, high
        )

        assert preds_attack.mean() > 0.9
        assert preds_flood.mean() > 0.9
        assert preds_normal.mean() < 0.2