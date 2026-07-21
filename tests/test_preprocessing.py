"""
Tests for src/anomaly/preprocessing.py
Covers: frame features, temporal split integrity (no leakage),
windowing shapes and labeling, normal-window extraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly.preprocessing import (
    add_frame_features,
    temporal_split,
    build_feature_matrix,
    make_windows,
    extract_normal_windows,
    prepare_datasets,
)


def make_frames(n: int, attack_from: int | None = None) -> pd.DataFrame:
    """Build a synthetic parsed CAN DataFrame like data_loader output."""
    rows = []
    for i in range(n):
        label = 1 if attack_from is not None and i >= attack_from else 0
        rows.append((float(i) * 0.001, "0316", 8, [i % 256] * 8, label))
    return pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", "data", "label"])


class TestFrameFeatures:
    def test_adds_expected_columns(self):
        df = add_frame_features(make_frames(10))
        assert "inter_arrival" in df.columns
        assert "id_freq" in df.columns

    def test_inter_arrival_first_frame_is_zero(self):
        df = add_frame_features(make_frames(10))
        assert df["inter_arrival"].iloc[0] == 0.0

    def test_inter_arrival_is_log1p_of_gap(self):
        df = add_frame_features(make_frames(10))
        expected = np.log1p(0.001)
        assert df["inter_arrival"].iloc[1] == pytest.approx(expected)

    def test_inter_arrival_never_negative(self):
        frames = make_frames(10)
        frames.loc[5, "timestamp"] = 0.0
        df = add_frame_features(frames)
        assert (df["inter_arrival"] >= 0).all()

    def test_single_id_freq_is_one(self):
        df = add_frame_features(make_frames(50))
        assert df["id_freq"].iloc[-1] == pytest.approx(1.0)

    def test_id_freq_reflects_id_share(self):
        rows = []
        for i in range(100):
            can_id = "0316" if i % 2 == 0 else "0260"
            rows.append((float(i) * 0.001, can_id, 8, [0] * 8, 0))
        frames = pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", "data", "label"])
        df = add_frame_features(frames)
        assert df["id_freq"].iloc[-1] == pytest.approx(0.5, abs=0.01)


class TestTemporalSplit:
    def test_split_sizes(self):
        train, test = temporal_split(make_frames(100), train_ratio=0.7)
        assert len(train) == 70
        assert len(test) == 30

    def test_no_frame_overlap(self):
        frames = make_frames(100)
        train, test = temporal_split(frames, train_ratio=0.7)
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_chronological_order_preserved(self):
        train, test = temporal_split(make_frames(100))
        assert train["timestamp"].is_monotonic_increasing
        assert test["timestamp"].is_monotonic_increasing

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            temporal_split(make_frames(10), train_ratio=1.5)
        with pytest.raises(ValueError):
            temporal_split(make_frames(10), train_ratio=0.0)


class TestFeatureMatrix:
    def test_shape_is_n_by_11(self):
        df = add_frame_features(make_frames(20))
        features = build_feature_matrix(df)
        assert features.shape == (20, 11)

    def test_data_bytes_normalized(self):
        frames = make_frames(300)
        df = add_frame_features(frames)
        features = build_feature_matrix(df)
        assert features[:, 3:].max() <= 1.0
        assert features[:, 3:].min() >= 0.0

    def test_dtype_is_float32(self):
        df = add_frame_features(make_frames(10))
        assert build_feature_matrix(df).dtype == np.float32


class TestWindows:
    def test_window_shapes(self):
        features = np.zeros((100, 11), dtype=np.float32)
        labels = np.zeros(100, dtype=np.int64)
        X, y = make_windows(features, labels, window_size=10, stride=5)
        assert X.shape == (19, 10, 11)
        assert y.shape == (19,)

    def test_too_few_frames_returns_empty(self):
        features = np.zeros((5, 11), dtype=np.float32)
        labels = np.zeros(5, dtype=np.int64)
        X, y = make_windows(features, labels, window_size=10, stride=5)
        assert X.shape == (0, 10, 11)
        assert len(y) == 0

    def test_all_normal_labels_zero(self):
        features = np.zeros((100, 11), dtype=np.float32)
        labels = np.zeros(100, dtype=np.int64)
        _, y = make_windows(features, labels, window_size=10, stride=5)
        assert (y == 0).all()

    def test_attack_ratio_above_threshold_flips_label(self):
        features = np.zeros((10, 11), dtype=np.float32)
        labels = np.array([0] * 8 + [1] * 2, dtype=np.int64)
        _, y = make_windows(features, labels, window_size=10, stride=10, attack_threshold=0.1)
        assert y[0] == 1

    def test_attack_ratio_below_threshold_stays_normal(self):
        features = np.zeros((10, 11), dtype=np.float32)
        labels = np.array([0] * 9 + [1], dtype=np.int64)
        _, y = make_windows(features, labels, window_size=10, stride=10, attack_threshold=0.1)
        assert y[0] == 0

    def test_invalid_params_raise(self):
        features = np.zeros((10, 11), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError):
            make_windows(features, labels, window_size=0)
        with pytest.raises(ValueError):
            make_windows(features, labels, stride=0)


class TestNormalExtraction:
    def test_keeps_only_label_zero(self):
        X = np.arange(4 * 2 * 11, dtype=np.float32).reshape(4, 2, 11)
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        normal = extract_normal_windows(X, y)
        assert normal.shape[0] == 2
        assert np.array_equal(normal[0], X[0])
        assert np.array_equal(normal[1], X[2])


class TestFullPipeline:
    def test_prepare_datasets_keys_and_shapes(self):
        df = make_frames(500, attack_from=350)
        ds = prepare_datasets(df, train_ratio=0.7, window_size=32, stride=16)
        for key in ["X_train", "y_train", "X_train_normal", "X_test", "y_test"]:
            assert key in ds
        assert ds["X_train"].shape[1:] == (32, 11)
        assert ds["X_test"].shape[1:] == (32, 11)

    def test_train_is_fully_normal_test_has_attacks(self):
        df = make_frames(500, attack_from=350)
        ds = prepare_datasets(df, train_ratio=0.7, window_size=32, stride=16)
        assert ds["y_train"].sum() == 0
        assert ds["y_test"].sum() > 0

    def test_normal_subset_never_larger_than_train(self):
        df = make_frames(500, attack_from=100)
        ds = prepare_datasets(df, train_ratio=0.7, window_size=32, stride=16)
        assert len(ds["X_train_normal"]) <= len(ds["X_train"])