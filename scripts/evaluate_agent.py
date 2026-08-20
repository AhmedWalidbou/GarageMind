"""
Agent evaluation - GarageMind M4

Runs the 15 scenarios from knowledge/agent_scenarios.json against the
real model and reports what the agent actually did, not what it said it
did.

Scoring decisions, stated so they can be argued with:

    - Tools are scored as sets, not sequences. A scenario declares which
      tools the question requires, not the order in which a model should
      chain them; penalising order would score style, not capability.
    - Citation grounding is checked against tool results, never against
      the answer text alone. A case id the tools never returned is a
      hallucinated citation, whatever it looks like.
    - Declining is measured twice: a lexical signal (does the answer read
      like a refusal) and a structural one (no tool called, no case
      cited). The lexical detector is the weak part of this harness and
      both numbers are reported separately rather than merged into a
      flattering single score.
    - The first scenario pays for loading the embedding model, so it is
      excluded from the latency average and reported on its own.

Usage:
    python scripts/evaluate_agent.py
    python scripts/evaluate_agent.py --limit 3        # cheap smoke run
    python scripts/evaluate_agent.py --model mistral-medium-latest
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.backend import get_backend
from src.agent.graph import MAX_TURNS, run_agent

SCENARIOS_PATH = Path("knowledge/agent_scenarios.json")
RESULTS_PATH = Path("results/agent_eval.json")

CASE_PATTERN = re.compile(r"case-\d+", re.IGNORECASE)

# Words a refusal or a request for more information tends to use. This is
# a lexical heuristic, not ground truth - see the module docstring.
DECLINE_MARKERS = [
    "hors de mon domaine", "en dehors de mon domaine", "pas mon domaine",
    "je ne peux pas répondre", "je ne peux pas vous aider",
    "outside my scope", "outside of my scope", "cannot answer",
    "je n'ai pas de définition", "je ne dispose pas", "je ne connais pas",
    "no definition", "i do not have a definition",
    "pouvez-vous préciser", "pourriez-vous préciser", "quel code",
    "avez-vous", "could you specify", "which fault code",
]


# The backend degrades API failures into a final answer carrying this
# prefix. Such a run measures the network, not the agent, and must never
# be averaged in with real behaviour.
DEGRADED_PREFIX = "Error: the language model is unavailable"


def is_degraded(answer: str) -> bool:
    return (answer or "").startswith(DEGRADED_PREFIX)


def load_scenarios(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["scenarios"] if isinstance(data, dict) else data


def cases_in(text: str) -> set[str]:
    """Every case id mentioned in a piece of text, normalised."""
    return {match.lower() for match in CASE_PATTERN.findall(text or "")}


def looks_like_a_decline(answer: str) -> bool:
    lowered = " ".join((answer or "").lower().split())
    return any(marker in lowered for marker in DECLINE_MARKERS)


def score_scenario(scenario: dict, result) -> dict:
    """Turn one run into a row of evidence."""
    expected_tools = set(scenario["expected_tools"])
    used_tools = set(result.tools_used)

    # What the tools actually returned, versus what the answer claims.
    grounded_cases = {case.lower() for case in result.cases_cited}
    claimed_cases = cases_in(result.answer)
    hallucinated = sorted(claimed_cases - grounded_cases)

    expected_cases = {case.lower() for case in scenario["expected_cases"]}
    found_cases = sorted(expected_cases & grounded_cases)
    missed_cases = sorted(expected_cases - grounded_cases)

    lowered_answer = " ".join((result.answer or "").lower().split())
    keywords = [k.lower() for k in scenario["expected_keywords"]]
    hit_keywords = [k for k in keywords if k in lowered_answer]

    must_decline = scenario["must_decline"]
    declined_lexically = looks_like_a_decline(result.answer)
    declined_structurally = not used_tools and not claimed_cases

    return {
        "id": scenario["id"],
        "type": scenario["type"],
        "input": scenario["input"],
        "answer": result.answer,
        "turns": result.turns,
        "latency_ms": round(result.latency_ms, 1),
        "stop_reason": result.stop_reason,
        "tools_used": result.tools_used,
        "tools_expected": sorted(expected_tools),
        "tools_exact": used_tools == expected_tools,
        "tools_covered": expected_tools.issubset(used_tools),
        "extra_tools": sorted(used_tools - expected_tools),
        "cases_grounded": sorted(grounded_cases),
        "cases_claimed": sorted(claimed_cases),
        "cases_hallucinated": hallucinated,
        "cases_expected_found": found_cases,
        "cases_expected_missed": missed_cases,
        "keywords_expected": keywords,
        "keywords_hit": hit_keywords,
        "must_decline": must_decline,
        "declined_lexically": declined_lexically,
        "declined_structurally": declined_structurally,
        # Structural is primary: "called no tool and cited no case" is a
        # property of what the agent did, not of the words it chose. The
        # lexical signal is kept alongside because a silent non-answer and
        # an explicit refusal are not the same quality of behaviour, but it
        # cannot be trusted to enumerate every phrasing of a refusal.
        "decline_ok": declined_structurally if must_decline else None,
        "degraded": is_degraded(result.answer),
    }


def rate(numerator: int, denominator: int):
    """
    None, not 0.0, when there is nothing to score.

    A zero denominator means the question was never asked; reporting it
    as 0.0 reads as total failure and would misrepresent any partial run.
    """
    return round(numerator / denominator, 4) if denominator else None


def summarise(all_rows: list[dict]) -> dict:
    """
    Score only the runs that actually reached the model.

    A degraded run reflects an API failure; counting it as a wrong answer
    would understate the agent, and counting it as a right one would
    flatter it. It is excluded and reported on its own line.
    """
    degraded = [row for row in all_rows if row["degraded"]]
    rows = [row for row in all_rows if not row["degraded"]]
    traps = [row for row in rows if row["must_decline"]]
    normal = [row for row in rows if not row["must_decline"]]
    with_cases = [row for row in rows if row["keywords_expected"] or row["cases_expected_found"]
                  or row["cases_expected_missed"]]

    # The first row carries the embedding model load; keep it out of the mean.
    cold_start_ms = rows[0]["latency_ms"] if rows else 0.0
    warm = [row["latency_ms"] for row in rows[1:]]

    keyword_hits = sum(len(row["keywords_hit"]) for row in rows)
    keyword_total = sum(len(row["keywords_expected"]) for row in rows)

    expected_case_hits = sum(len(row["cases_expected_found"]) for row in rows)
    expected_case_total = expected_case_hits + sum(len(row["cases_expected_missed"]) for row in rows)

    return {
        "scenarios_run": len(all_rows),
        "scenarios_scored": len(rows),
        "degraded_runs": len(degraded),
        "degraded_ids": [row["id"] for row in degraded],
        "tool_exact_match": rate(sum(row["tools_exact"] for row in rows), len(rows)),
        "tool_coverage": rate(sum(row["tools_covered"] for row in rows), len(rows)),
        "citation_grounding": rate(
            sum(not row["cases_hallucinated"] for row in rows), len(rows)
        ),
        "hallucinated_citations": sum(len(row["cases_hallucinated"]) for row in rows),
        "expected_case_recall": rate(expected_case_hits, expected_case_total),
        "keyword_coverage": rate(keyword_hits, keyword_total),
        "traps": len(traps),
        "traps_declined": sum(bool(row["decline_ok"]) for row in traps),
        "traps_declined_rate": rate(sum(bool(row["decline_ok"]) for row in traps), len(traps)),
        "traps_declined_lexically": sum(row["declined_lexically"] for row in traps),
        "traps_lexical_only": sum(
            row["declined_lexically"] and not row["declined_structurally"] for row in traps
        ),
        "traps_structural_only": sum(
            row["declined_structurally"] and not row["declined_lexically"] for row in traps
        ),
        "normal_scenarios": len(normal),
        "hit_turn_limit": sum(row["stop_reason"] == "turn_limit" for row in rows),
        "repeated_calls": 0,  # filled by the caller from the traces
        "mean_turns": round(statistics.mean([row["turns"] for row in rows]), 2) if rows else 0,
        "cold_start_ms": cold_start_ms,
        "mean_latency_ms_warm": round(statistics.mean(warm), 1) if warm else 0.0,
        "median_latency_ms_warm": round(statistics.median(warm), 1) if warm else 0.0,
        "max_latency_ms_warm": round(max(warm), 1) if warm else 0.0,
        "unused_scenarios": len(with_cases) and 0,
    }


def print_row(row: dict) -> None:
    tools = ", ".join(row["tools_used"]) or "-"
    flags = []
    if row["must_decline"]:
        flags.append("DECLINE OK" if row["decline_ok"] else "DECLINE FAILED")
    else:
        flags.append("tools ok" if row["tools_exact"] else "tools differ")
    if row["cases_hallucinated"]:
        flags.append(f"HALLUCINATED {','.join(row['cases_hallucinated'])}")
    if row["stop_reason"] == "turn_limit":
        flags.append("TURN LIMIT")
    if row["degraded"]:
        flags = ["DEGRADED (api failure, excluded)"]
    print(f"  {row['id']} [{row['type']:<22}] {row['turns']} turns "
          f"{row['latency_ms']:>7.0f} ms  tools=[{tools}]  {' | '.join(flags)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ReAct agent on the scenario set")
    parser.add_argument("--model", default="mistral-small-latest")
    parser.add_argument("--limit", type=int, help="Run only the first N scenarios")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    args = parser.parse_args()

    scenarios = load_scenarios(SCENARIOS_PATH)
    if args.limit:
        scenarios = scenarios[:args.limit]

    backend = get_backend("mistral", model=args.model)
    if not backend.api_key:
        print("Error: MISTRAL_API_KEY is not set.", file=sys.stderr)
        return 2

    print(f"Evaluating {len(scenarios)} scenarios on {args.model} "
          f"(max {args.max_turns} turns)\n")

    rows = []
    repeated_calls = 0
    started = time.perf_counter()

    for scenario in scenarios:
        result = run_agent(scenario["input"], backend, max_turns=args.max_turns)
        # One retry: a transient API error should not silently void a
        # scenario. If it fails twice the row is kept and marked degraded.
        if is_degraded(result.answer):
            time.sleep(2)
            result = run_agent(scenario["input"], backend, max_turns=args.max_turns)
        repeated_calls += sum(entry["repeated"] for entry in result.trace)
        row = score_scenario(scenario, result)
        rows.append(row)
        print_row(row)

    total_seconds = time.perf_counter() - started
    summary = summarise(rows)
    summary["repeated_calls"] = repeated_calls
    summary.pop("unused_scenarios", None)
    summary["model"] = args.model
    summary["max_turns"] = args.max_turns
    summary["total_seconds"] = round(total_seconds, 1)

    print("\n--- summary ---")
    for key in ("tool_exact_match", "tool_coverage", "citation_grounding",
                "expected_case_recall", "keyword_coverage", "traps_declined_rate"):
        value = summary[key]
        print(f"  {key:<24} {'n/a (nothing to score)' if value is None else value}")
    print(f"  {'hallucinated citations':<24} {summary['hallucinated_citations']}")
    print(f"  {'repeated tool calls':<24} {summary['repeated_calls']}")
    print(f"  {'hit turn limit':<24} {summary['hit_turn_limit']}")
    print(f"  {'degraded runs (excluded)':<24} {summary['degraded_runs']} "
          f"{summary['degraded_ids'] or ''}")
    print(f"  {'traps declined (lexical)':<24} {summary['traps_declined_lexically']}"
          f"/{summary['traps']}")
    print(f"  {'mean turns':<24} {summary['mean_turns']}")
    print(f"  {'cold start':<24} {summary['cold_start_ms']:.0f} ms")
    print(f"  {'mean latency (warm)':<24} {summary['mean_latency_ms_warm']:.0f} ms")
    print(f"  {'median latency (warm)':<24} {summary['median_latency_ms_warm']:.0f} ms")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "scenarios": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())