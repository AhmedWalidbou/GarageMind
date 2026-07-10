"""
Report Generator - GarageMind M1 Scan Engine
Unifies every M1 component into a single, rigorous diagnostic report.

Sources combined:
  - Vehicle identity          (VIN decode)
  - UDS diagnostic scan       (DTCs + status + freeze frames)
  - Decoded CAN live signals  (from the extraction pipeline, parquet)
  - REAL correlation          (compare each freeze-frame value to the live
                               signal distribution: is it normal or an outlier?)
  - Health assessment         (0-100 score + verdict)
  - Prioritized repair plan   (severity-ordered; effort from a documented table)

Outputs: canonical JSON (machine) + Markdown (human).
Everything downstream (M2, M4) consumes this report.
"""

import os
import json
from datetime import datetime, timezone

from src.scan_engine.vin_decoder import decode_vin
from src.uds.uds_client import UdsClient
from src.uds.uds_ecu_sim import build_demo_ecu


SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1, "unknown": 1, None: 0}

# Repair-effort reference table.
# NOTE: these are INDICATIVE ranges compiled from public flat-rate manual
# conventions (ordres de grandeur atelier), NOT vehicle-specific quotes.
# Flagged as 'indicative' in the output so nothing is presented as exact.
EFFORT_TABLE = {
    "cylindre":     {"effort": "moyen",    "hours_range": (1.0, 2.0)},
    "egr":          {"effort": "moyen",    "hours_range": (1.5, 3.0)},
    "fap":          {"effort": "eleve",    "hours_range": (2.5, 4.0)},
    "catalyseur":   {"effort": "eleve",    "hours_range": (3.0, 5.0)},
    "communication":{"effort": "variable", "hours_range": (1.0, 4.0)},
    "thermostat":   {"effort": "faible",   "hours_range": (0.5, 1.5)},
    "melange":      {"effort": "moyen",    "hours_range": (1.0, 3.0)},
}

# Maps a freeze-frame condition key to the live CAN signal(s) that measure
# the same physical quantity, so we can cross-check them.
FREEZE_TO_SIGNAL = {
    "engine_rpm":     ["N", "N_ENG"],
    "vehicle_speed":  ["VS", "SLD_VS"],
    "coolant_temp":   ["TEMP_ENG"],
}


def _effort_for(fault: dict) -> dict:
    text = (fault.get("description_fr", "") + " " +
            " ".join(fault.get("likely_causes_fr", []))).lower()
    for keyword, hint in EFFORT_TABLE.items():
        if keyword in text:
            lo, hi = hint["hours_range"]
            return {
                "effort": hint["effort"],
                "hours_min": lo,
                "hours_max": hi,
                "basis": "indicative (flat-rate order of magnitude)",
            }
    return {"effort": "a evaluer", "hours_min": None,
            "hours_max": None, "basis": "unknown subsystem"}


def assess_health(interpretation: dict) -> dict:
    results = interpretation.get("results", [])
    if not results:
        return {"score": 100, "verdict_en": "No faults detected",
                "verdict_fr": "Aucun defaut detecte", "critical_count": 0,
                "medium_count": 0, "action_required": False}

    penalty = sum(SEVERITY_WEIGHT.get(r.get("severity_key"), 1) * 12 for r in results)
    score = max(0, 100 - penalty)
    critical = sum(1 for r in results if r.get("severity_key") == "high")
    medium = sum(1 for r in results if r.get("severity_key") == "medium")

    if critical > 0:
        verdict_fr, verdict_en = "Intervention immediate requise", "Immediate action required"
    elif score < 70:
        verdict_fr, verdict_en = "Reparations recommandees", "Repairs recommended"
    else:
        verdict_fr, verdict_en = "Defauts mineurs a surveiller", "Minor faults to monitor"

    return {"score": score, "verdict_en": verdict_en, "verdict_fr": verdict_fr,
            "critical_count": critical, "medium_count": medium,
            "action_required": critical > 0 or score < 70}


def load_signals(parquet_path: str | None) -> dict:
    """Load decoded CAN signals; compute per-signal distribution stats."""
    if not parquet_path or not os.path.exists(parquet_path):
        return {"available": False}
    try:
        import pandas as pd
        ts = pd.read_parquet(parquet_path)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    stats = {}
    for sig in ts["signal"].unique():
        s = ts[ts["signal"] == sig]["value"]
        if len(s) == 0:
            continue
        stats[sig] = {
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "latest": float(s.iloc[-1]),
            "count": int(len(s)),
        }
    return {"available": True, "signal_count": ts["signal"].nunique(),
            "datapoint_count": int(len(ts)), "stats": stats}


def correlate_freeze_frames(faults: list, signal_stats: dict) -> list:
    """
    REAL correlation: for each fault's freeze-frame value, find the matching
    live CAN signal and quantify how the snapshot compares to the observed
    distribution (z-score + inside/outside the observed range).
    """
    correlations = []
    for fault in faults:
        ff = fault.get("freeze_frame", {})
        checks = []
        context_notes = []

        for cond_key, cond_data in ff.items():
            value = cond_data.get("value")
            unit = cond_data.get("unit", "")

            # Qualitative context (always available from the freeze frame)
            if cond_key == "engine_rpm" and value is not None:
                if value < 900:
                    context_notes.append("defaut au ralenti")
                elif value > 2000:
                    context_notes.append("defaut en charge / haut regime")
                else:
                    context_notes.append("defaut en regime intermediaire")
            elif cond_key == "vehicle_speed" and value is not None:
                context_notes.append("vehicule a l'arret" if value == 0
                                     else f"vehicule en mouvement ({value} {unit})")
            elif cond_key == "coolant_temp" and value is not None:
                if value < 60:
                    context_notes.append("moteur froid")
                elif value > 90:
                    context_notes.append("moteur chaud")

            # Quantitative cross-check against live signal distribution
            candidate_signals = FREEZE_TO_SIGNAL.get(cond_key, [])
            matched = next((s for s in candidate_signals if s in signal_stats), None)
            if matched and value is not None:
                st = signal_stats[matched]
                inside = st["min"] <= value <= st["max"]
                # z-score only meaningful when the signal actually varies
                if st["std"] > 1e-6:
                    z = round((value - st["mean"]) / st["std"], 2)
                    z_note = None
                else:
                    z = None
                    z_note = "signal constant sur le log (variance nulle)"
                checks.append({
                    "freeze_condition": cond_key,
                    "freeze_value": value,
                    "matched_signal": matched,
                    "live_mean": round(st["mean"], 2),
                    "live_range": [round(st["min"], 2), round(st["max"], 2)],
                    "z_score": z,
                    "z_note": z_note,
                    "within_live_range": bool(inside),
                    "interpretation": (
                        "coherent avec le comportement observe" if inside
                        else "hors de la plage live observee -> condition atypique"
                    ),
                })

        correlations.append({
            "code": fault["code"],
            "context_notes_fr": context_notes,
            "quantitative_checks": checks,
        })
    return correlations


def build_repair_plan(faults: list) -> list:
    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    plan = []
    for fault in sorted(faults, key=lambda f: order.get(f.get("severity_key"), 3)):
        effort = _effort_for(fault)
        plan.append({
            "priority": len(plan) + 1,
            "code": fault["code"],
            "action_fr": fault["description_fr"],
            "severity_fr": fault.get("severity_fr", ""),
            "first_checks_fr": fault.get("likely_causes_fr", [])[:3],
            "effort": effort["effort"],
            "estimated_hours_range": [effort["hours_min"], effort["hours_max"]],
            "estimate_basis": effort["basis"],
        })
    return plan


def generate_report(vin: str | None = None, ecu=None,
                    signals_parquet_path: str | None = None,
                    lang: str = "fr") -> dict:
    if ecu is None:
        ecu = build_demo_ecu()
    client = UdsClient(ecu)
    scan = client.full_scan(lang=lang)

    vin_to_decode = vin or scan.get("vin")
    identity = decode_vin(vin_to_decode, use_api=False) if vin_to_decode else {}

    signals = load_signals(signals_parquet_path)
    faults = scan["interpretation"].get("results", [])
    health = assess_health(scan["interpretation"])
    correlations = correlate_freeze_frames(faults, signals.get("stats", {}))
    repair_plan = build_repair_plan(faults)

    return {
        "report_metadata": {
            "tool": "GarageMind", "module": "M1 Scan Engine", "version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "language": lang,
            "disclaimer": ("Effort/hours are indicative order-of-magnitude "
                           "values, not vehicle-specific quotes."),
        },
        "vehicle_identity": identity,
        "health_assessment": health,
        "diagnostic_scan": {
            "vin": scan.get("vin"),
            "dtc_count": scan.get("dtc_count", 0),
            "highest_severity": scan["interpretation"].get("highest_severity"),
            "faults": faults,
        },
        "context_correlation": correlations,
        "repair_plan": repair_plan,
        "signals_overview": {k: v for k, v in signals.items() if k != "stats"},
        "transport_stats": scan.get("transport_stats", {}),
    }


def save_json(report: dict, out_dir: str = "data/processed") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"diagnostic_report_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def save_markdown(report: dict, out_dir: str = "data/processed") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"diagnostic_report_{stamp}.md")
    identity = report["vehicle_identity"]
    health = report["health_assessment"]
    scan = report["diagnostic_scan"]
    corr_by_code = {c["code"]: c for c in report["context_correlation"]}

    lines = [
        "# GarageMind - Rapport de diagnostic", "",
        f"**Vehicule** : {identity.get('manufacturer', 'Inconnu')} "
        f"({identity.get('country', '?')})  ",
        f"**VIN** : {scan.get('vin', 'N/A')}  ",
        f"**Genere le** : {report['report_metadata']['generated_at']}", "",
        f"## Sante du vehicule : {health['score']}/100",
        f"**Verdict** : {health['verdict_fr']}  ",
        f"Critiques : {health['critical_count']} | moyens : {health['medium_count']}", "",
        "## Plan de reparation priorise", "",
        "| # | Code | Action | Gravite | Effort | Heures (indic.) |",
        "|---|------|--------|---------|--------|-----------------|",
    ]
    for item in report["repair_plan"]:
        rng = item["estimated_hours_range"]
        hours = f"{rng[0]}-{rng[1]}h" if rng[0] is not None else "-"
        lines.append(f"| {item['priority']} | {item['code']} | {item['action_fr']} | "
                     f"{item['severity_fr']} | {item['effort']} | {hours} |")

    lines += ["", "## Correlation freeze-frame vs signaux live", ""]
    for fault in scan["faults"]:
        corr = corr_by_code.get(fault["code"], {})
        notes = ", ".join(corr.get("context_notes_fr", [])) or "pas de contexte"
        lines.append(f"### {fault['code']} - {fault['description_fr']}")
        lines.append(f"- Contexte : {notes}")
        for chk in corr.get("quantitative_checks", []):
            zpart = (f"z={chk['z_score']}" if chk["z_score"] is not None
                     else chk["z_note"])
            lines.append(
                f"- {chk['matched_signal']} = {chk['freeze_value']} "
                f"(live moy {chk['live_mean']}, plage {chk['live_range']}, "
                f"{zpart}) -> {chk['interpretation']}")
        lines.append("")

    lines.append(f"> {report['report_metadata']['disclaimer']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def print_summary(report: dict) -> None:
    identity = report["vehicle_identity"]
    health = report["health_assessment"]
    scan = report["diagnostic_scan"]
    corr_by_code = {c["code"]: c for c in report["context_correlation"]}

    print("=" * 64)
    print("  GARAGEMIND - RAPPORT DE DIAGNOSTIC")
    print("=" * 64)
    print(f"\n  Vehicule : {identity.get('manufacturer', 'Inconnu')} "
          f"({identity.get('country', '?')})")
    print(f"  VIN      : {scan.get('vin', 'N/A')}")
    if identity.get("year"):
        print(f"  Annee    : {identity['year']}")
    print(f"\n  Sante    : {health['score']}/100 - {health['verdict_fr']}")
    print(f"  Defauts  : {scan['dtc_count']} critiques={health['critical_count']} "
          f"moyens={health['medium_count']}")
    if health["action_required"]:
        print(f"  /!\\ ACTION REQUISE")

    sig = report["signals_overview"]
    if sig.get("available"):
        print(f"\n  Signaux live : {sig['signal_count']} signaux, "
              f"{sig['datapoint_count']} points")

    print(f"\n  --- Plan de reparation priorise ---")
    for item in report["repair_plan"]:
        rng = item["estimated_hours_range"]
        hours = f"{rng[0]}-{rng[1]}h" if rng[0] is not None else "?"
        print(f"    {item['priority']}. [{item['code']}] {item['action_fr']}")
        print(f"       Gravite: {item['severity_fr']} | Effort: {item['effort']} (~{hours})")
        corr = corr_by_code.get(item["code"], {})
        notes = corr.get("context_notes_fr", [])
        if notes:
            print(f"       Contexte: {', '.join(notes)}")
        for chk in corr.get("quantitative_checks", []):
            flag = "OK" if chk["within_live_range"] else "ATYPIQUE"
            if chk["z_score"] is not None:
                detail = f"z={chk['z_score']}"
            else:
                detail = chk["z_note"]
            print(f"       [{flag}] {chk['matched_signal']}={chk['freeze_value']} "
                  f"vs live {chk['live_range']} ({detail})")

    stats = report["transport_stats"]
    print(f"\n  Transport UDS: {stats.get('frames_sent', 0)} envoyees, "
          f"{stats.get('frames_received', 0)} recues")
    print(f"\n  Note: {report['report_metadata']['disclaimer']}")


if __name__ == "__main__":
    report = generate_report(
        signals_parquet_path="data/processed/signals_timeseries.parquet",
        lang="fr",
    )
    print_summary(report)
    json_path = save_json(report)
    md_path = save_markdown(report)
    print(f"\n  Rapport JSON     : {json_path}")
    print(f"  Rapport Markdown : {md_path}")