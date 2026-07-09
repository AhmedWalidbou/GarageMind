"""
Signal Analysis - GarageMind M1 Scan Engine
Deep analysis layer over extracted CAN signals:
  1. Forensic investigation of signals flagged as physically implausible
     (inspects DBC scaling: factor, offset, length, byte order).
  2. Time-series visualization of key powertrain signals.

Turns 800k+ raw datapoints into human-understandable engineering insight.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt

from src.scan_engine.dbc_decoder import DbcDecoder


def investigate_signal(dbc_path: str, signal_name: str) -> dict:
    """
    Forensic look at how a signal is defined in the DBC.
    Reveals why a decoded value might be physically implausible:
    scaling factor, offset, bit length, byte order, declared range.
    """
    decoder = DbcDecoder(dbc_path)
    if not decoder.loaded:
        return {"found": False, "error": "DBC not loaded"}

    for msg in decoder.db.messages:
        for sig in msg.signals:
            if sig.name == signal_name:
                return {
                    "found": True,
                    "signal": sig.name,
                    "message": msg.name,
                    "can_id_hex": f"{msg.frame_id:X}",
                    "start_bit": sig.start,
                    "length_bits": sig.length,
                    "byte_order": sig.byte_order,
                    "is_signed": sig.is_signed,
                    "factor": sig.scale,
                    "offset": sig.offset,
                    "unit": sig.unit or "",
                    "declared_min": sig.minimum,
                    "declared_max": sig.maximum,
                }
    return {"found": False, "error": f"Signal '{signal_name}' not in DBC"}


def diagnose_implausible(dbc_path: str, ts: pd.DataFrame, signal_name: str) -> dict:
    """
    Combine the DBC definition with observed data to explain an anomaly.
    Checks whether the raw (pre-scaling) values suggest a scaling mismatch.
    """
    definition = investigate_signal(dbc_path, signal_name)
    if not definition.get("found"):
        return definition

    observed = ts[ts["signal"] == signal_name]["value"]
    if len(observed) == 0:
        definition["observed"] = "no data"
        return definition

    factor = definition["factor"] or 1
    offset = definition["offset"] or 0
    # Reconstruct the raw integer values from the scaled ones
    raw_estimate = (observed - offset) / factor

    definition.update({
        "observed_min": round(observed.min(), 2),
        "observed_max": round(observed.max(), 2),
        "raw_min": round(raw_estimate.min(), 1),
        "raw_max": round(raw_estimate.max(), 1),
        "hypothesis": _scaling_hypothesis(definition, observed),
    })
    return definition


def _scaling_hypothesis(defn: dict, observed: pd.Series) -> str:
    """Simple heuristic to explain an implausible reading."""
    val = observed.mean()
    if defn["offset"] and abs(defn["offset"]) > 40:
        return (f"Large offset ({defn['offset']}) dominates. If this DBC's offset "
                f"does not match the actual ECU, the signal is mis-scaled. "
                f"Likely a different vehicle variant than the log source.")
    if val < defn.get("declared_min", val):
        return ("Value below declared minimum -> probable scaling/variant mismatch, "
                "not a real physical reading.")
    return "Value within declared bounds; may be a genuine reading."


def plot_signal(ts: pd.DataFrame, signal_name: str, out_dir: str) -> str | None:
    """Plot one signal over time and save it as PNG."""
    data = ts[ts["signal"] == signal_name].sort_values("timestamp")
    if data.empty:
        return None
    t0 = data["timestamp"].min()
    t = data["timestamp"] - t0

    plt.figure(figsize=(10, 3.5))
    plt.plot(t, data["value"], linewidth=0.8)
    plt.title(f"GarageMind - {signal_name} over time")
    plt.xlabel("Time (s)")
    plt.ylabel(signal_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"signal_{signal_name}.png")
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_dashboard(ts: pd.DataFrame, signals: list[str], out_dir: str) -> str | None:
    """Multi-signal dashboard: several key signals stacked vertically."""
    available = [s for s in signals if s in ts["signal"].unique()]
    if not available:
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.2 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, sig in zip(axes, available):
        data = ts[ts["signal"] == sig].sort_values("timestamp")
        t0 = data["timestamp"].min()
        ax.plot(data["timestamp"] - t0, data["value"], linewidth=0.8)
        ax.set_ylabel(sig)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("GarageMind - Powertrain signals dashboard")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dashboard_powertrain.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    TS_PATH = "data/processed/signals_timeseries.parquet"
    DBC_PATH = "data/dbc/opendbc-master/opendbc/dbc/hyundai_i30_2014.dbc"
    PLOT_DIR = "data/processed/plots"

    print("=" * 64)
    print("GarageMind - Deep signal analysis")
    print("=" * 64)

    ts = pd.read_parquet(TS_PATH)
    print(f"\nLoaded {len(ts)} datapoints, {ts['signal'].nunique()} unique signals")

    print("\n--- Forensic investigation: TEMP_FUEL (-48 degC anomaly) ---")
    diag = diagnose_implausible(DBC_PATH, ts, "TEMP_FUEL")
    if diag.get("found"):
        print(f"  Message      : {diag['message']} (ID {diag['can_id_hex']})")
        print(f"  Bit layout   : start={diag['start_bit']}, len={diag['length_bits']}, "
              f"order={diag['byte_order']}, signed={diag['is_signed']}")
        print(f"  Scaling      : factor={diag['factor']}, offset={diag['offset']} "
              f"{diag['unit']}")
        print(f"  Declared range: [{diag['declared_min']}, {diag['declared_max']}]")
        print(f"  Observed     : {diag['observed_min']} .. {diag['observed_max']} "
              f"(raw ~{diag['raw_min']} .. {diag['raw_max']})")
        print(f"  Hypothesis   : {diag['hypothesis']}")
    else:
        print(f"  {diag.get('error')}")

    print("\n--- Generating time-series plots ---")
    key_signals = ["N", "TQI", "IntAirTemp", "CR_Fatc_OutTempSns"]
    for sig in key_signals:
        path = plot_signal(ts, sig, PLOT_DIR)
        print(f"  {sig:10} -> {path if path else 'no data'}")

    dash = plot_dashboard(ts, key_signals, PLOT_DIR)
    print(f"\n  Dashboard  -> {dash}")