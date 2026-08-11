"""
Retrieval evaluation - GarageMind M3 EBR-RAG
Runs the 25-query evaluation set against the persistent index and
reports Hit@k, MRR, per-language breakdown and query latency.

Design decisions:
    - Hit@k ("success at k"): a query is solved at k if at least one
      acceptable case id appears in the top-k unique cases. The query
      set defines relevant_cases as "every acceptable case id", so any
      of them serves the mechanic equally well; requiring all of them
      would punish queries that are ambiguous by design.
    - MRR uses the rank of the first relevant case: it rewards putting
      an acceptable answer as high as possible.
    - Per-language aggregates (fr/en) expose cross-lingual asymmetry
      that global averages would hide (18 fr / 7 en: the en slice is
      indicative only).
    - Fail fast on a missing or wrong-cardinality index: metrics
      computed on a stale index are worse than no metrics.
    - The model is warmed up before timing so the first query's latency
      does not include model loading.
    - Per-query rows are persisted to results/ for failure analysis;
      aggregates feed the README table.
"""

import json
import time
from pathlib import Path

from src.ebr.embedder import Embedder
from src.ebr.retriever import Retriever
from src.ebr.vectorstore import VectorStore

QUERIES_PATH = Path("knowledge/eval_queries.json")
INDEX_PATH = Path("data/qdrant")
RESULTS_PATH = Path("results/retrieval_eval.json")
EXPECTED_POINTS = 40
TOP_K = 5
K_LEVELS = (1, 3, 5)


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"no queries found in {path}")
    return queries


def evaluate_query(retriever: Retriever, query: dict) -> dict:
    t0 = time.perf_counter()
    results = retriever.retrieve(query["query"], top_k=TOP_K)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ranked = [r.case_id for r in results]
    relevant = set(query["relevant_cases"])
    first_rank = next(
        (i for i, cid in enumerate(ranked, start=1) if cid in relevant), None
    )
    return {
        "id": query["id"],
        "lang": query["lang"],
        "relevant_cases": query["relevant_cases"],
        "retrieved": [
            {"case_id": r.case_id, "lang": r.lang, "score": round(r.score, 4)}
            for r in results
        ],
        "first_relevant_rank": first_rank,
        "hits": {
            f"hit@{k}": int(first_rank is not None and first_rank <= k)
            for k in K_LEVELS
        },
        "reciprocal_rank": (
            0.0 if first_rank is None else round(1.0 / first_rank, 4)
        ),
        "latency_ms": round(latency_ms, 1),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict]) -> dict:
    return {
        "n_queries": len(rows),
        **{
            f"hit@{k}": round(mean([r["hits"][f"hit@{k}"] for r in rows]), 4)
            for k in K_LEVELS
        },
        "mrr": round(mean([r["reciprocal_rank"] for r in rows]), 4),
    }


def main() -> None:
    queries = load_queries(QUERIES_PATH)

    store = VectorStore(location=str(INDEX_PATH))
    try:
        count = store.count()
    except Exception:
        raise SystemExit(
            f"no index found at {INDEX_PATH} - run scripts/build_index.py first"
        )
    if count != EXPECTED_POINTS:
        raise SystemExit(
            f"index at {INDEX_PATH} has {count} points, expected "
            f"{EXPECTED_POINTS} - run scripts/build_index.py first"
        )

    embedder = Embedder()
    retriever = Retriever(embedder, store)

    print("Warming up the model ...")
    embedder.embed_query("warmup")

    print(f"Evaluating {len(queries)} queries (top_k={TOP_K}) ...\n")
    rows = [evaluate_query(retriever, q) for q in queries]

    overall = aggregate(rows)
    by_lang = {
        lang: aggregate([r for r in rows if r["lang"] == lang])
        for lang in sorted({r["lang"] for r in rows})
    }
    latencies = sorted(r["latency_ms"] for r in rows)
    p95 = latencies[max(0, round(0.95 * (len(latencies) - 1)))]

    print(f"{'':10}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for label, agg in [("overall", overall)] + list(by_lang.items()):
        print(
            f"{label:10}{agg['n_queries']:>4}"
            f"{agg['hit@1']:>8.2f}{agg['hit@3']:>8.2f}"
            f"{agg['hit@5']:>8.2f}{agg['mrr']:>8.4f}"
        )
    print(f"\nlatency: mean {mean(latencies):.0f} ms, p95 {p95:.0f} ms")

    failures = [r for r in rows if r["hits"]["hit@1"] == 0]
    if failures:
        print(f"\nqueries not solved at rank 1 ({len(failures)}):")
        for r in failures:
            got = ", ".join(
                f"{h['case_id']}({h['score']:.3f})" for h in r["retrieved"][:3]
            )
            print(
                f"  {r['id']} [{r['lang']}] expected "
                f"{'/'.join(r['relevant_cases'])} -> got {got}"
            )
    else:
        print("\nall queries solved at rank 1")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "model": embedder.model_name,
            "top_k": TOP_K,
            "k_levels": list(K_LEVELS),
            "index_points": count,
            "queries_file": str(QUERIES_PATH),
        },
        "overall": overall,
        "by_lang": by_lang,
        "latency_ms": {"mean": round(mean(latencies), 1), "p95": p95},
        "per_query": rows,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nresults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()