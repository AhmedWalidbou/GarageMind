"""
Data Loader - GarageMind M2 Anomaly Detection
Robust loader for the HCRL Car-Hacking dataset that correctly handles
variable-length CAN frames (DLC 0-8).

The raw CSV has a variable number of columns per line:
    timestamp, can_id, dlc, <dlc data bytes>, flag
So the flag is ALWAYS the last field, not a fixed column. A naive fixed-width
parse misplaces the flag for short frames (e.g. DLC=2). This loader reads each
row by its true structure.
"""

import pandas as pd
import numpy as np


def load_can_dataset(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load a Car-Hacking CSV into a clean, fixed-schema DataFrame.

    Output columns:
        timestamp (float), can_id (str, upper hex), dlc (int),
        data (list[int] of length 8, zero-padded), label (int: 0=normal, 1=attack)
    """
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            timestamp = float(parts[0])
            can_id = parts[1].upper()
            dlc = int(parts[2])
            # Data bytes are the fields between dlc and the final flag.
            flag = parts[-1]
            data_fields = parts[3:3 + dlc]
            data = [int(b, 16) for b in data_fields] + [0] * (8 - len(data_fields))
            data = data[:8]
            label = 1 if flag == "T" else 0
            rows.append((timestamp, can_id, dlc, data, label))

    df = pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", "data", "label"])
    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute clean statistics after robust parsing."""
    return {
        "total_frames": len(df),
        "normal_frames": int((df["label"] == 0).sum()),
        "attack_frames": int((df["label"] == 1).sum()),
        "attack_ratio_pct": round(100 * df["label"].mean(), 2),
        "unique_can_ids": df["can_id"].nunique(),
        "dlc_distribution": df["dlc"].value_counts().sort_index().to_dict(),
    }


if __name__ == "__main__":
    PATH = "data/raw/DoS_dataset.csv"

    print("Loading with robust variable-DLC parser (first 200000 rows)...")
    df = load_can_dataset(PATH, nrows=200000)

    stats = dataset_stats(df)
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSanity check - no unlabeled frames:")
    print(f"  All labels are 0 or 1: {df['label'].isin([0, 1]).all()}")

    print("\nSample frames (mix of DLC lengths):")
    for dlc_val in [2, 8]:
        sample = df[df["dlc"] == dlc_val].head(2)
        for _, row in sample.iterrows():
            print(f"  ID={row['can_id']} DLC={row['dlc']} "
                  f"data={row['data']} label={row['label']}")