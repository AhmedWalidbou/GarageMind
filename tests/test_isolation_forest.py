"""
Tests for src/anomaly/isolation_forest.py
Covers: window aggregation correctness, fitting, scoring direction,
threshold calibration and the end-to-end detection protocol — mirroring
the LSTM AE test structure so both benchmark models meet the same bar.
Uses small synthetic data so the suite stays fast.
"""

import numpy as np
import pytest

from src.anomaly.isolation_forest import (
    aggregate_windows,
    fit_isolation_forest,
    anomaly_scores,
    calibrate_threshold,
    predict,
)

WINDOW = 16
FEATURES = 11


def normal_windows(n: int, seed: int = 0) -> np.ndarray:
    """Low-amplitude noisy signals imitating normal traffic."""
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 3, WINDOW)).reshape(1, WINDOW, 1)
    X = np.tile(base, (n, 1, FEATURES)) * 0.1
    return (X + rng.normal(0, 0.01, X.shape)).astype(np.float32)


def attack_windows(n: int) -> np.ndarray:
    """High-amplitude noise imitating irregular attack traffic."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.8, 1.0, (n, WINDOW, FEATURES)).astype(np.float32)


class TestAggregation:
    def test_output_shape(self):
        X = normal_windows(10)
        agg = aggregate_windows(X)
        assert agg.shape == (10, 4 * FEATURES)

    def test_statistics_are_correct(self):
        X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        agg = aggregate_windows(X)
        # window 0, feature 0 holds values [0, 2, 4]
        assert agg[0, 0] == pytest.approx(2.0)            # mean
        assert agg[0, 2] == pytest.approx(np.std([0, 2, 4]))  # std
        assert agg[0, 4] == pytest.approx(0.0)            # min
        assert agg[0, 6] == pytest.approx(4.0)            # max

    def test_rejects_non_3d_input(self):
        with pytest.raises(ValueError):
            aggregate_windows(np.zeros((5, 4), dtype=np.float32))

    def test_dtype_is_float32(self):
        agg = aggregate_windows(normal_windows(5))
        assert agg.dtype == np.float32


class TestFitAndScore:
    def test_fit_returns_model(self):
        model = fit_isolation_forest(normal_windows(64), n_estimators=50)
        assert hasattr(model, "score_samples")

    def test_scores_shape(self):
        model = fit_isolation_forest(normal_windows(64), n_estimators=50)
        scores = anomaly_scores(model, normal_windows(20, seed=2))
        assert scores.shape == (20,)

    def test_attacks_score_higher(self):
        model = fit_isolation_forest(normal_windows(128), n_estimators=100)
        s_normal = anomaly_scores(model, normal_windows(32, seed=3))
        s_attack = anomaly_scores(model, attack_windows(32))
        assert s_attack.mean() > s_normal.mean()

    def test_deterministic_with_seed(self):
        X = normal_windows(64)
        s1 = anomaly_scores(fit_isolation_forest(X, n_estimators=50, seed=7), X)
        s2 = anomaly_scores(fit_isolation_forest(X, n_estimators=50, seed=7), X)
        assert np.allclose(s1, s2)


class TestThresholdAndPrediction:
    def test_threshold_is_percentile(self):
        scores = np.arange(100, dtype=np.float32)
        assert calibrate_threshold(scores, percentile=98.0) == pytest.approx(97.02)

    def test_invalid_percentile_raises(self):
        scores = np.arange(100, dtype=np.float32)
        with pytest.raises(ValueError):
            calibrate_threshold(scores, percentile=0.0)
        with pytest.raises(ValueError):
            calibrate_threshold(scores, percentile=100.0)

    def test_predict_binary_output(self):
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        preds = predict(scores, threshold=0.5)
        assert preds.tolist() == [0, 0, 1]


class TestEndToEndDetection:
    def test_detects_attacks_with_low_false_alarms(self):
        model = fit_isolation_forest(normal_windows(128), n_estimators=100)
        s_train = anomaly_scores(model, normal_windows(128))
        threshold = calibrate_threshold(s_train, percentile=98.0)

        preds_attack = predict(anomaly_scores(model, attack_windows(32)), threshold)
        preds_normal = predict(anomaly_scores(model, normal_windows(64, seed=5)), threshold)

        assert preds_attack.mean() > 0.9
        assert preds_normal.mean() < 0.2