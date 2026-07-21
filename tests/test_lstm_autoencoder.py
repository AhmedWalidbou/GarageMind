"""
Tests for src/anomaly/lstm_autoencoder.py
Covers: architecture shapes, training loss decrease, scoring,
threshold calibration and prediction logic. Uses a tiny model and
synthetic data so the suite stays fast.
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
    predict,
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
    """High-amplitude noise imitating flood traffic."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.8, 1.0, (n, WINDOW, FEATURES)).astype(np.float32)


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


class TestThresholdAndPrediction:
    def test_threshold_is_percentile(self):
        scores = np.arange(100, dtype=np.float32)
        assert calibrate_threshold(scores, percentile=99.0) == pytest.approx(98.01)

    def test_predict_binary_output(self):
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        preds = predict(scores, threshold=0.5)
        assert preds.tolist() == [0, 0, 1]

    def test_end_to_end_detection(self):
        set_seed()
        model = tiny_model()
        X_normal = normal_windows(128)
        train_autoencoder(model, X_normal, epochs=15, batch_size=16, verbose=False)
        threshold = calibrate_threshold(reconstruction_errors(model, X_normal))
        preds_attack = predict(reconstruction_errors(model, attack_windows(32)), threshold)
        assert preds_attack.mean() > 0.9