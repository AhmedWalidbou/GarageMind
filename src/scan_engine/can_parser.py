"""
CAN Parser - GarageMind M1 Scan Engine
Parses raw CAN bus logs (HCRL Car-Hacking format) into structured frames.
Format per line: timestamp, can_id, dlc, d0..d7, flag(R/T)
"""

import pandas as pd

COLUMNS = ["timestamp", "can_id", "dlc",
           "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "flag"]

DATA_COLS = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]


def load_can_log(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load a raw CAN log into a DataFrame.
    nrows: limit number of rows (useful for quick tests on huge files).
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
    )
    df["timestamp"] = df["timestamp"].astype(float)
    df["dlc"] = df["dlc"].astype(int)
    df["can_id"] = df["can_id"].str.upper()
    return df


def filter_normal(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only genuine (non-injected) messages, flagged 'R'."""
    return df[df["flag"] == "R"].reset_index(drop=True)


def filter_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only injected/attack messages, flagged 'T'."""
    return df[df["flag"] == "T"].reset_index(drop=True)


def payload_bytes(row: pd.Series) -> list[int]:
    """Return the data payload of one frame as a list of integers."""
    dlc = row["dlc"]
    result = []
    for col in DATA_COLS[:dlc]:
        value = row[col]
        result.append(int(value, 16) if value else 0)
    return result


def bus_summary(df: pd.DataFrame) -> dict:
    """Compute high-level statistics about the CAN bus traffic."""
    duration = df["timestamp"].max() - df["timestamp"].min()
    unique_ids = df["can_id"].nunique()
    total = len(df)
    counts = df["flag"].value_counts().to_dict()
    top_ids = df["can_id"].value_counts().head(10).to_dict()
    return {
        "total_frames": total,
        "duration_seconds": round(duration, 2),
        "frames_per_second": round(total / duration, 1) if duration > 0 else 0,
        "unique_can_ids": unique_ids,
        "normal_frames": counts.get("R", 0),
        "attack_frames": counts.get("T", 0),
        "top_10_can_ids": top_ids,
    }


if __name__ == "__main__":
    LOG_PATH = "data/raw/DoS_dataset.csv"

    print("Loading CAN log (first 100000 frames for quick test)...")
    df = load_can_log(LOG_PATH, nrows=100000)
    print(f"Loaded {len(df)} frames\n")

    print("Bus summary:")
    for key, value in bus_summary(df).items():
        if key == "top_10_can_ids":
            print(f"  {key}:")
            for cid, cnt in value.items():
                print(f"    {cid}: {cnt} frames")
        else:
            print(f"  {key}: {value}")

    print("\nFirst normal frame decoded:")
    normal = filter_normal(df)
    first = normal.iloc[0]
    print(f"  CAN ID: {first['can_id']}")
    print(f"  Timestamp: {first['timestamp']}")
    print(f"  Payload (int): {payload_bytes(first)}")