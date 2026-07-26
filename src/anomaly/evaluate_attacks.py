"""
Multi-attack, multi-model benchmark - GarageMind M2 Anomaly Detection
Evaluates every benchmark model on each HCRL Car-Hacking attack dataset
(DoS, Fuzzy, Spoofing gear, Spoofing RPM) with identical windows, split,
seed and false-alarm budget, so results are directly comparable.

Models:
    lstm_ae  : LSTM autoencoder, bilateral percentile thresholds (P1/P99)
    iforest  : Isolation Forest on aggregated window statistics (P98)
    Both are calibrated label-free on normal training windows with the
    same ~2% false-alarm budget on normal traffic.

Attack profiles (why generalization is not guaranteed):
    DoS      : one ID flooded with constant payloads -> unnaturally regular
               traffic, caught by the AE's LOW threshold.
    Fuzzy    : random IDs and payloads -> chaotic traffic, high scores.
    Gear/RPM : spoofed values injected on legitimate IDs -> traffic closely
               imitates normal patterns; Gear is the current weak spot
               (AE recall 0.36).

Outputs:
    results/benchmark_attacks.json    per-model, per-attack metrics
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
from src.anomaly import isolation_forest as iforest

DATASETS = {
    "DoS": "data/raw/DoS_dataset.csv",
    "Fuzzy": "data/raw/Fuzzy_dataset.csv",
    "Gear": "data/raw/gear_dataset.csv",
    "RPM": "data/raw/RPM_dataset.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anomaly models on HCRL attacks")
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
    parser.add_argument("--iforest-estimators", type=int, default=200)
    parser.add_argument("--iforest-percentile", type=float, default=98.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric_record(model_name: str, dataset_name: str, y_true, y_pred, auc_scores) -> dict:
    """Shared metric computation so every model is measured identically."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": model_name,
        "dataset": dataset_name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, auc_scores)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_lstm_ae(ds: dict, args: argparse.Namespace, dataset_name: str) -> dict:
    """Train + evaluate the bilateral LSTM AE on prepared windows."""
    set_seed(args.seed)
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
    scores_train = reconstruction_errors(model, ds["X_train_normal"])
    low, high = calibrate_thresholds(
        scores_train,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    scores_test = reconstruction_errors(model, ds["X_test"])
    y_pred = predict_bilateral(scores_test, low, high)
    band_distance = np.maximum(low - scores_test, scores_test - high)
    record = metric_record("lstm_ae", dataset_name, ds["y_test"], y_pred, band_distance)
    record["final_loss"] = round(history[-1], 6)
    record["threshold_low"] = round(low, 6)
    record["threshold_high"] = round(high, 6)
    return record


def run_iforest(ds: dict, args: argparse.Namespace, dataset_name: str) -> dict:
    """Fit + evaluate the Isolation Forest baseline on the same windows."""
    model = iforest.fit_isolation_forest(
        ds["X_train_normal"],
        n_estimators=args.iforest_estimators,
        seed=args.seed,
    )
    scores_train = iforest.anomaly_scores(model, ds["X_train_normal"])
    threshold = iforest.calibrate_threshold(scores_train, percentile=args.iforest_percentile)
    scores_test = iforest.anomaly_scores(model, ds["X_test"])
    y_pred = iforest.predict(scores_test, threshold)
    record = metric_record("iforest", dataset_name, ds["y_test"], y_pred, scores_test)
    record["threshold"] = round(threshold, 6)
    return record


def evaluate_dataset(name: str, path: str, args: argparse.Namespace) -> list[dict]:
    """Prepare windows ONCE, evaluate every model on the same windows."""
    print(f"\n===== {name} =====")
    set_seed(args.seed)

    t0 = time.time()
    df = load_can_dataset(path, nrows=args.nrows)
    print(f"[1/4] {len(df)} frames loaded "
          f"({100 * df['label'].mean():.2f}% attack frames)")

    ds = prepare_datasets(df, window_size=args.window_size, stride=args.stride)
    print(f"[2/4] windows: train={len(ds['X_train'])} "
          f"normal={len(ds['X_train_normal'])} test={len(ds['X_test'])}")

    records = []

    rec_ae = run_lstm_ae(ds, args, name)
    print(f"[3/4] lstm_ae : precision={rec_ae['precision']} recall={rec_ae['recall']} "
          f"f1={rec_ae['f1']} auc={rec_ae['roc_auc']}")
    records.append(rec_ae)

    rec_if = run_iforest(ds, args, name)
    print(f"[4/4] iforest : precision={rec_if['precision']} recall={rec_if['recall']} "
          f"f1={rec_if['f1']} auc={rec_if['roc_auc']}")
    records.append(rec_if)

    for r in records:
        r["runtime_context_sec"] = round(time.time() - t0, 1)
    return records


def markdown_table(records: list[dict]) -> str:
    """Benchmark table (one row per model x attack) for the README."""
    lines = [
        "| Model | Attack | Precision | Recall | F1 | ROC-AUC |",
        "|-------|--------|-----------|--------|-----|---------|",
    ]
    for r in records:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['precision']} "
            f"| {r['recall']} | {r['f1']} | {r['roc_auc']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    all_records = []
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"\n===== {name} ===== SKIPPED (missing {path})")
            continue
        all_records.extend(evaluate_dataset(name, path, args))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report = {
        "models": ["lstm_ae", "iforest"],
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
            "iforest_estimators": args.iforest_estimators,
            "iforest_percentile": args.iforest_percentile,
            "seed": args.seed,
        },
        "results": all_records,
    }
    out_path = results_dir / "benchmark_attacks.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Benchmark (all models) ===\n")
    print(markdown_table(all_records))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()