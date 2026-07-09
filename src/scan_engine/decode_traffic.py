"""
Decode Traffic - GarageMind M1 Scan Engine
Full signal-extraction engine: connects the CAN parser and DBC decoder,
extracts complete time-series for every decoded signal, validates values
against physical plausibility ranges, auto-selects the best-matching DBC,
and exports structured data for downstream modules (M2 anomaly detection).
"""

import os
import glob
import pandas as pd

from src.scan_engine.can_parser import (
    load_can_log, filter_normal, payload_bytes, bus_summary
)
from src.scan_engine.dbc_decoder import DbcDecoder


# Physical plausibility ranges for common powertrain signals.
# Used to flag decoding errors or genuinely abnormal readings.
PHYSICAL_RANGES = {
    "TEMP_ENG": (-40, 150, "degC"),        # engine coolant temperature
    "N": (0, 8000, "rpm"),                 # engine speed
    "N_ENG": (0, 8000, "rpm"),
    "TQI": (0, 100, "%"),                  # indicated torque
    "TQI_TARGET": (0, 100, "%"),
    "TQI_MIN": (0, 100, "%"),
    "VS": (0, 300, "km/h"),                # vehicle speed
    "SLD_VS": (0, 300, "km/h"),
    "TPS": (0, 100, "%"),                  # throttle position
    "PV_AV_CAN": (0, 100, "%"),            # accelerator pedal
    "TEMP_FUEL": (-40, 120, "degC"),
    "BAT_VOLT": (8, 16, "V"),              # battery voltage
}


class SignalExtractor:
    """Extracts, validates and exports named signal time-series from CAN logs."""

    def __init__(self, dbc_path: str):
        self.decoder = DbcDecoder(dbc_path)
        self.dbc_path = dbc_path

    def extract(self, df_normal: pd.DataFrame) -> dict:
        """
        Decode every frame and build a long-format time-series table:
        one row per (timestamp, message, signal, value).
        """
        records = []
        decoded_count = 0
        seen_messages = set()

        for _, row in df_normal.iterrows():
            result = self.decoder.decode_frame(row["can_id"], payload_bytes(row))
            if not result:
                continue
            decoded_count += 1
            seen_messages.add(result["message_name"])
            for sig_name, sig_val in result["signals"].items():
                # Keep only numeric signals for time-series analysis
                if isinstance(sig_val, (int, float)):
                    records.append({
                        "timestamp": row["timestamp"],
                        "message": result["message_name"],
                        "can_id": result["can_id_hex"],
                        "signal": sig_name,
                        "value": float(sig_val),
                    })

        ts = pd.DataFrame.from_records(records)
        return {
            "timeseries": ts,
            "frames_decoded": decoded_count,
            "unique_messages": len(seen_messages),
        }

    @staticmethod
    def validate_physical(ts: pd.DataFrame) -> pd.DataFrame:
        """
        For each signal with a known physical range, compute how many
        readings fall inside the plausible band. Flags decoding issues.
        """
        rows = []
        for sig_name, (lo, hi, unit) in PHYSICAL_RANGES.items():
            sig_data = ts[ts["signal"] == sig_name]["value"]
            if len(sig_data) == 0:
                continue
            in_range = sig_data.between(lo, hi).sum()
            total = len(sig_data)
            rows.append({
                "signal": sig_name,
                "unit": unit,
                "expected_range": f"[{lo}, {hi}]",
                "observed_min": round(sig_data.min(), 2),
                "observed_max": round(sig_data.max(), 2),
                "observed_mean": round(sig_data.mean(), 2),
                "samples": total,
                "in_range_pct": round(100 * in_range / total, 1),
                "status": "OK" if in_range == total else "CHECK",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def signal_summary(ts: pd.DataFrame) -> pd.DataFrame:
        """Statistical summary of every extracted numeric signal."""
        if ts.empty:
            return pd.DataFrame()
        summary = ts.groupby("signal")["value"].agg(
            ["count", "min", "max", "mean", "std"]
        ).round(3).reset_index()
        return summary.sort_values("count", ascending=False)


def find_hyundai_dbcs(dbc_dir: str) -> list[str]:
    """Return all Hyundai DBC files available for auto-matching."""
    pattern = os.path.join(dbc_dir, "hyundai*.dbc")
    return sorted(glob.glob(pattern))


def auto_select_dbc(df_normal: pd.DataFrame, dbc_dir: str) -> dict:
    """
    Try every Hyundai DBC and rank them by coverage on this vehicle's traffic.
    Returns the ranking and the best-matching DBC path.
    """
    observed_ids = df_normal["can_id"].unique().tolist()
    ranking = []
    for dbc_path in find_hyundai_dbcs(dbc_dir):
        decoder = DbcDecoder(dbc_path)
        if not decoder.loaded:
            continue
        cov = decoder.coverage(observed_ids)
        ranking.append({
            "dbc": os.path.basename(dbc_path),
            "path": dbc_path,
            "matched_ids": cov["matched_ids"],
            "coverage_percent": cov["coverage_percent"],
        })
    ranking.sort(key=lambda r: r["coverage_percent"], reverse=True)
    return {
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
    }


def run_full_extraction(log_path: str, dbc_dir: str, nrows: int = 100000,
                        export_dir: str = "data/processed") -> dict:
    """End-to-end: load, auto-select DBC, extract, validate, export."""
    df = load_can_log(log_path, nrows=nrows)
    normal = filter_normal(df)

    # 1. Auto-select the best DBC
    selection = auto_select_dbc(normal, dbc_dir)
    best = selection["best"]
    if not best:
        raise RuntimeError("No usable Hyundai DBC found")

    # 2. Extract time-series with the best DBC
    extractor = SignalExtractor(best["path"])
    extraction = extractor.extract(normal)
    ts = extraction["timeseries"]

    # 3. Validate physically
    validation = extractor.validate_physical(ts)
    summary = extractor.signal_summary(ts)

    # 4. Export for downstream modules
    os.makedirs(export_dir, exist_ok=True)
    ts_path = os.path.join(export_dir, "signals_timeseries.parquet")
    summary_path = os.path.join(export_dir, "signals_summary.csv")
    ts.to_parquet(ts_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "bus": bus_summary(df),
        "dbc_selection": selection,
        "best_dbc": best,
        "frames_decoded": extraction["frames_decoded"],
        "unique_messages": extraction["unique_messages"],
        "signals_extracted": ts["signal"].nunique() if not ts.empty else 0,
        "total_datapoints": len(ts),
        "validation": validation,
        "summary": summary,
        "exports": {"timeseries": ts_path, "summary": summary_path},
    }


if __name__ == "__main__":
    LOG_PATH = "data/raw/DoS_dataset.csv"
    DBC_DIR = "data/dbc/opendbc-master/opendbc/dbc"

    print("=" * 64)
    print("GarageMind - Full signal extraction pipeline")
    print("=" * 64)

    report = run_full_extraction(LOG_PATH, DBC_DIR, nrows=100000)

    print("\n--- DBC auto-selection (ranked by coverage) ---")
    for r in report["dbc_selection"]["ranking"]:
        mark = " <-- selected" if r["dbc"] == report["best_dbc"]["dbc"] else ""
        print(f"  {r['dbc']:28} {r['coverage_percent']:5}% "
              f"({r['matched_ids']} ids){mark}")

    print(f"\nFrames decoded      : {report['frames_decoded']}")
    print(f"Unique messages     : {report['unique_messages']}")
    print(f"Signals extracted   : {report['signals_extracted']}")
    print(f"Total datapoints    : {report['total_datapoints']}")

    print("\n--- Physical validation of key signals ---")
    val = report["validation"]
    if not val.empty:
        for _, r in val.iterrows():
            print(f"  [{r['status']:5}] {r['signal']:12} "
                  f"{r['observed_min']:>8} .. {r['observed_max']:<8} {r['unit']:5} "
                  f"(mean {r['observed_mean']}, {r['in_range_pct']}% in range)")
    else:
        print("  (no known-range signals found in this DBC)")

    print("\n--- Top 10 extracted signals ---")
    print(report["summary"].head(10).to_string(index=False))

    print("\n--- Exports ---")
    print(f"  Time-series : {report['exports']['timeseries']}")
    print(f"  Summary     : {report['exports']['summary']}")