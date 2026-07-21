"""
Multi-attack evaluation - GarageMind M2 Anomaly Detection
Runs the full LSTM AE pipeline on each HCRL Car-Hacking attack dataset
(DoS, Fuzzy, Spoofing gear, Spoofing RPM) with identical config and seed,
so results are directly comparable.

Attack profiles (why generalization is not guaranteed):
    DoS      : one ID flooded with constant payloads -> unnaturally regular
               traffic, caught by the LOW threshold.
    Fuzzy    : random IDs and payloads -> chaotic traffic, expected to
               trigger the HIGH threshold.
    Gear/RPM : spoofed values injected on legitimate IDs -> traffic closely
               imitates normal patterns, hardest case for reconstruction-
               based detection.

Outputs:
    results/benchmark_attacks.json    per-attack metrics and config
    stdout                            markdown table ready for the README

Usage (from repo root):
    python -m src.anomaly.evaluate_attacks
    python -m src.anomaly.evaluate_attacks --nrows 200000 --epochs 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
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

DATASETS = {
    "DoS": "data/raw/DoS_dataset.csv",
    "Fuzzy": "data/raw/Fuzzy_dataset.csv",
    "Gear": "data/raw/gear_dataset.csv",
    "RPM": "data/raw/RPM_dataset.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM AE on all HCRL attacks")
    parser.add_argument("--nrows", type=int, default=500000)
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


def evaluate_dataset(name: str, path: str, args: argparse.Namespace) -> dict:
    """Full pipeline on one dataset; returns the metrics record."""
    print(f"\n===== {name} =====")
    set_seed(args.seed)

    t0 = time.time()
    df = load_can_dataset(path, nrows=args.nrows)
    print(f"[1/4] {len(df)} frames loaded "
          f"({100 * df['label'].mean():.2f}% attack frames)")

    ds = prepare_datasets(df, window_size=args.window_size, stride=args.stride)
    print(f"[2/4] windows: train={len(ds['X_train'])} "
          f"normal={len(ds['X_train_normal'])} test={len(ds['X_test'])}")

    model = LSTMAutoencoder(
        n_features=ds["X_train"].shape[2],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
    )
    history = train_autoencoder(
        model,
        ds["X_train_normal"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=False,
    )
    print(f"[3/4] trained ({args.epochs} epochs, final loss {history[-1]:.6f})")

    scores_train_normal = reconstruction_errors(model, ds["X_train_normal"])
    low, high = calibrate_thresholds(
        scores_train_normal,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    scores_test = reconstruction_errors(model, ds["X_test"])
    y_pred = predict_bilateral(scores_test, low, high)
    y_true = ds["y_test"]
    band_distance = np.maximum(low - scores_test, scores_test - high)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    record = {
        "dataset": name,
        "frames": len(df),
        "attack_frame_ratio_pct": round(100 * float(df["label"].mean()), 2),
        "test_windows": int(len(y_true)),
        "test_attack_ratio_pct": round(100 * float(y_true.mean()), 2),
        "threshold_low": round(low, 6),
        "threshold_high": round(high, 6),
        "final_loss": round(history[-1], 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc_band_distance": round(float(roc_auc_score(y_true, band_distance)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "runtime_sec": round(time.time() - t0, 1),
    }
    print(f"[4/4] precision={record['precision']} recall={record['recall']} "
          f"f1={record['f1']} auc={record['roc_auc_band_distance']}")
    return record


def markdown_table(records: list[dict]) -> str:
    """Benchmark table ready to paste into the README."""
    lines = [
        "| Attack | Precision | Recall | F1 | ROC-AUC |",
        "|--------|-----------|--------|-----|---------|",
    ]
    for r in records:
        lines.append(
            f"| {r['dataset']} | {r['precision']} | {r['recall']} "
            f"| {r['f1']} | {r['roc_auc_band_distance']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = []
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"\n===== {name} ===== SKIPPED (missing {path})")
            continue
        records.append(evaluate_dataset(name, path, args))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report = {
        "model": "lstm_autoencoder_bilateral",
        "config": {
            "nrows": args.nrows,
            "window_size": args.window_size,
            "stride": args.stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "seed": args.seed,
        },
        "results": records,
    }
    out_path = results_dir / "benchmark_attacks.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Benchmark (bilateral LSTM AE) ===\n")
    print(markdown_table(records))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()