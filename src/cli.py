"""
GarageMind CLI - the diagnostic scan tool entry point.

Usage examples:
    garagemind scan                                   # full diagnostic, demo vehicle
    garagemind scan --lang en                          # English report
    garagemind scan --json out.json                    # save report to a path
    garagemind decode-vin WAUZZZ8V1JA123456 --no-api
    garagemind analyze-can log.csv --brand hyundai     # force a brand's DBCs
    garagemind version
"""

import os
import sys
import glob
import json
import argparse

from src.scan_engine.vin_decoder import decode_vin, validate_vin
from src.scan_engine.report_generator import (
    generate_report, print_summary, save_json, save_markdown,
)

VERSION = "1.0.0"

DBC_DIR = "data/dbc/opendbc-master/opendbc/dbc"
DEFAULT_SIGNALS = "data/processed/signals_timeseries.parquet"

# Known brand -> DBC filename glob, so --brand actually drives decoding.
BRAND_DBC_GLOBS = {
    "hyundai": "hyundai*.dbc",
    "toyota": "toyota*pt.dbc",
    "vw": "vw_*.dbc",
    "volkswagen": "vw_*.dbc",
    "ford": "ford*pt.dbc",
    "tesla": "tesla*.dbc",
    "psa": "psa_*.dbc",
    "peugeot": "psa_*.dbc",
    "citroen": "psa_*.dbc",
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_scan(args) -> int:
    """Run a full diagnostic scan and produce a report."""
    signals = args.signals
    if signals and not os.path.exists(signals):
        _eprint(f"Avertissement: fichier signaux introuvable ({signals}), "
                f"le rapport sera genere sans signaux live.")
        signals = None

    try:
        report = generate_report(signals_parquet_path=signals, lang=args.lang)
    except Exception as exc:
        _eprint(f"Erreur lors de la generation du rapport: {exc}")
        return 1

    print_summary(report)

    try:
        if args.json:
            out_dir = os.path.dirname(args.json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n  Rapport JSON sauvegarde: {args.json}")
        elif not args.no_save:
            json_path = save_json(report)
            md_path = save_markdown(report)
            print(f"\n  Rapport JSON     : {json_path}")
            print(f"  Rapport Markdown : {md_path}")
    except OSError as exc:
        _eprint(f"Erreur d'ecriture du rapport: {exc}")
        return 1
    return 0


def cmd_decode_vin(args) -> int:
    """Decode a VIN into vehicle identity."""
    vin = args.vin.strip().upper()
    if not validate_vin(vin):
        _eprint(f"Erreur: VIN invalide '{args.vin}' "
                f"(17 caracteres, sans I/O/Q attendus).")
        return 2
    result = decode_vin(vin, use_api=not args.no_api)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid") else 1


def _resolve_brand_dbcs(brand: str) -> list[str]:
    """Return the DBC files matching a brand, or [] if unknown/none found."""
    pattern = BRAND_DBC_GLOBS.get(brand.lower())
    if not pattern:
        return []
    return sorted(glob.glob(os.path.join(DBC_DIR, pattern)))


def cmd_analyze_can(args) -> int:
    """Decode a raw CAN log and print signal extraction stats."""
    if not os.path.exists(args.log):
        _eprint(f"Erreur: fichier log introuvable: {args.log}")
        return 2

    from src.scan_engine.can_parser import load_can_log, filter_normal
    from src.scan_engine.decode_traffic import (
        run_full_extraction, auto_select_dbc, SignalExtractor,
    )

    # If --brand is given, restrict decoding to that brand's DBCs.
    if args.brand:
        dbc_files = _resolve_brand_dbcs(args.brand)
        if not dbc_files:
            _eprint(f"Erreur: aucune DBC connue pour la marque '{args.brand}'. "
                    f"Marques supportees: {', '.join(sorted(set(BRAND_DBC_GLOBS)))}")
            return 2
        try:
            df = load_can_log(args.log, nrows=args.rows)
            normal = filter_normal(df)
            observed = normal["can_id"].unique().tolist()
            # Rank the brand's DBCs by coverage and keep the best.
            best = None
            for dbc in dbc_files:
                extractor = SignalExtractor(dbc)
                cov = extractor.decoder.coverage(observed)
                if best is None or cov["coverage_percent"] > best[1]:
                    best = (dbc, cov["coverage_percent"], extractor)
            dbc_path, coverage, extractor = best
            extraction = extractor.extract(normal)
            print(f"Marque forcee       : {args.brand}")
            print(f"DBC selectionne     : {os.path.basename(dbc_path)} "
                  f"({coverage}% couverture)")
            print(f"Trames decodees     : {extraction['frames_decoded']}")
            print(f"Messages uniques    : {extraction['unique_messages']}")
            print(f"Points de donnees   : {len(extraction['timeseries'])}")
            return 0
        except Exception as exc:
            _eprint(f"Erreur lors de l'analyse: {exc}")
            return 1

    # No brand: full auto-selection pipeline with export.
    try:
        report = run_full_extraction(args.log, DBC_DIR, nrows=args.rows)
    except Exception as exc:
        _eprint(f"Erreur lors de l'analyse: {exc}")
        return 1

    best = report["best_dbc"]
    print(f"DBC selectionne     : {best['dbc']} ({best['coverage_percent']}% couverture)")
    print(f"Trames decodees     : {report['frames_decoded']}")
    print(f"Signaux extraits    : {report['signals_extracted']}")
    print(f"Points de donnees   : {report['total_datapoints']}")
    print(f"\nExports:")
    print(f"  {report['exports']['timeseries']}")
    print(f"  {report['exports']['summary']}")
    return 0


def cmd_version(args) -> int:
    print(f"GarageMind {VERSION}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garagemind",
        description="GarageMind - AI-powered vehicle diagnostic copilot (M1 scan engine)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Run a full diagnostic scan")
    p_scan.add_argument("--lang", default="fr", choices=["fr", "en"])
    p_scan.add_argument("--signals", default=DEFAULT_SIGNALS)
    p_scan.add_argument("--json", help="Save the report to this JSON path")
    p_scan.add_argument("--no-save", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_vin = sub.add_parser("decode-vin", help="Decode a VIN")
    p_vin.add_argument("vin", help="17-character VIN")
    p_vin.add_argument("--no-api", action="store_true")
    p_vin.set_defaults(func=cmd_decode_vin)

    p_can = sub.add_parser("analyze-can", help="Decode a raw CAN log")
    p_can.add_argument("log", help="Path to the CAN log CSV")
    p_can.add_argument("--brand", help="Force a brand's DBCs "
                       f"({', '.join(sorted(set(BRAND_DBC_GLOBS)))})")
    p_can.add_argument("--rows", type=int, default=100000)
    p_can.set_defaults(func=cmd_analyze_can)

    p_ver = sub.add_parser("version", help="Show version")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())