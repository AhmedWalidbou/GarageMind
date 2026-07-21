"""
Training script - LSTM Autoencoder on the HCRL DoS dataset
GarageMind M2 Anomaly Detection

Pipeline:
    load frames -> preprocessing (temporal split, windows) ->
    train AE on normal windows -> calibrate BILATERAL thresholds on normal
    train scores -> evaluate on test windows -> save model + metrics JSON

Detection rule (v2, see lstm_autoencoder.py for the failure analysis):
    a window is anomalous if its reconstruction error falls outside the
    [low, high] percentile band of normal training errors. The lower bound
    catches DoS floods, whose constant traffic is EASIER to reconstruct
    than normal traffic (v1 upper-threshold-only rule scored ROC-AUC 0.056).

Outputs:
    models/lstm_ae.pt                 trained weights + config + thresholds
    results/lstm_ae_metrics.json      dataset stats, config and test metrics

Usage (from repo root):
    python -m src.anomaly.train_lstm_ae
    python -m src.anomaly.train_lstm_ae --nrows 1000000 --epochs 15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.anomaly.data_loader import load_can_dataset
from src.anomaly.preprocessing import prepare_datasets
from src.anomaly.lstm_autoencoder import (
    LSTMAutoencoder,
    set_seed,
    train_autoencoder,
    reconstruction_errors,
    calibrate_thresholds,
    predict_bilateral,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM AE on CAN DoS dataset")
    parser.add_argument("--data", default="data/raw/DoS_dataset.csv")
    parser.add_argument("--nrows", type=int, default=500000,
                        help="number of frames to load")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"[1/5] Loading {args.nrows} frames from {args.data} ...")
    t0 = time.time()
    df = load_can_dataset(args.data, nrows=args.nrows)
    print(f"      {len(df)} frames loaded in {time.time() - t0:.1f}s "
          f"({100 * df['label'].mean():.2f}% attack frames)")

    print(f"[2/5] Preprocessing (window={args.window_size}, stride={args.stride}) ...")
    ds = prepare_datasets(
        df,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"      X_train={ds['X_train'].shape} "
          f"X_train_normal={ds['X_train_normal'].shape} "
          f"X_test={ds['X_test'].shape}")

    print(f"[3/5] Training LSTM AE ({args.epochs} epochs, "
          f"hidden={args.hidden_size}, latent={args.latent_size}) ...")
    model = LSTMAutoencoder(
        n_features=ds["X_train"].shape[2],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
    )
    t0 = time.time()
    history = train_autoencoder(
        model,
        ds["X_train_normal"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train_time = time.time() - t0
    print(f"      done in {train_time:.1f}s (final loss {history[-1]:.6f})")

    print(f"[4/5] Calibrating bilateral thresholds "
          f"(P{args.low_percentile} / P{args.high_percentile} of normal scores) ...")
    scores_train_normal = reconstruction_errors(model, ds["X_train_normal"])
    low, high = calibrate_thresholds(
        scores_train_normal,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    print(f"      low = {low:.6f}  high = {high:.6f}")

    print("[5/5] Evaluating on test windows ...")
    scores_test = reconstruction_errors(model, ds["X_test"])
    y_pred = predict_bilateral(scores_test, low, high)
    y_true = ds["y_test"]

    # ROC-AUC needs a score where higher = more anomalous. With a bilateral
    # rule the natural score is the distance to the normal band.
    band_distance = np.maximum(low - scores_test, scores_test - high)

    metrics = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc_band_distance": round(float(roc_auc_score(y_true, band_distance)), 4),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n=== Test metrics (bilateral rule) ===")
    for key, value in metrics.items():
        print(f"  {key:22s}: {value}")
    print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_features": ds["X_train"].shape[2],
                "hidden_size": args.hidden_size,
                "latent_size": args.latent_size,
                "window_size": args.window_size,
                "stride": args.stride,
            },
            "threshold_low": low,
            "threshold_high": high,
        },
        models_dir / "lstm_ae.pt",
    )

    report = {
        "model": "lstm_autoencoder_bilateral",
        "dataset": {
            "path": args.data,
            "frames_loaded": len(df),
            "attack_frame_ratio_pct": round(100 * float(df["label"].mean()), 2),
            "train_windows": int(len(ds["X_train"])),
            "train_normal_windows": int(len(ds["X_train_normal"])),
            "test_windows": int(len(ds["X_test"])),
            "test_attack_ratio_pct": round(100 * float(y_true.mean()), 2),
        },
        "config": {
            "window_size": args.window_size,
            "stride": args.stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "threshold_low": round(low, 6),
            "threshold_high": round(high, 6),
            "seed": args.seed,
        },
        "training": {
            "final_loss": round(history[-1], 6),
            "train_time_sec": round(train_time, 1),
        },
        "test_metrics": {
            **metrics,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }
    out_path = results_dir / "lstm_ae_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nModel saved to {models_dir / 'lstm_ae.pt'}")
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()