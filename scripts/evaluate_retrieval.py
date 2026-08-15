"""
Retrieval benchmark - GarageMind M3 EBR-RAG
Runs the 25-query evaluation set through two retrievers - dense
(e5-small + Qdrant) and lexical (BM25) - with a single shared harness,
and reports Hit@k, MRR, per-language breakdown, latency and per-query
divergences.

Design decisions:
    - One harness, two retrievers: both systems expose
      retrieve(query, top_k) -> list[RetrievedCase], so the exact same
      evaluation code path scores both. Protocol divergence is
      impossible by construction.
    - Hit@k ("success at k"): a query is solved at k if at least one
      acceptable case id appears in the top-k unique cases (the query
      set defines relevant_cases as "every acceptable case id").
    - MRR uses the rank of the first relevant case.
    - Raw scores are never compared across systems (cosine and BM25
      live on different scales): only ranks and rank-based metrics.
    - The divergence report (queries solved at rank 1 by exactly one
      system) is the benchmark's most informative output: it shows
      where semantics beat lexical matching and vice versa.
    - Fail fast on a missing or wrong-cardinality index: metrics
      computed on a stale index are worse than no metrics.
"""

import json
import time
from pathlib import Path

from src.ebr.corpus import load_documents
from src.ebr.embedder import Embedder
from src.ebr.lexical import LexicalRetriever
from src.ebr.retriever import Retriever
from src.ebr.vectorstore import VectorStore

KNOWLEDGE_PATH = Path("knowledge/repair_cases.json")
QUERIES_PATH = Path("knowledge/eval_queries.json")
INDEX_PATH = Path("data/qdrant")
RESULTS_PATH = Path("results/retrieval_benchmark.json")
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


def evaluate_query(retriever, query: dict) -> dict:
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
        "latency_ms": round(latency_ms, 2),
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


def latency_stats(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in rows)
    p95 = latencies[max(0, round(0.95 * (len(latencies) - 1)))]
    return {"mean": round(mean(latencies), 2), "p95": p95}


def print_table(all_rows: dict) -> None:
    print(f"\n{'system':10}{'slice':9}{'n':>4}"
          f"{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}{'lat ms':>10}")
    for name, rows in all_rows.items():
        lat = latency_stats(rows)
        slices = [("overall", rows)] + [
            (lang, [r for r in rows if r["lang"] == lang])
            for lang in sorted({r["lang"] for r in rows})
        ]
        for label, subset in slices:
            agg = aggregate(subset)
            lat_str = (
                f"{lat['mean']:.0f}/{lat['p95']:.0f}"
                if label == "overall" else ""
            )
            print(
                f"{name:10}{label:9}{agg['n_queries']:>4}"
                f"{agg['hit@1']:>8.2f}{agg['hit@3']:>8.2f}"
                f"{agg['hit@5']:>8.2f}{agg['mrr']:>8.4f}{lat_str:>10}"
            )


def print_divergences(queries: list[dict], all_rows: dict) -> None:
    print("\nqueries solved at rank 1 by exactly one system:")
    names = list(all_rows.keys())
    found = False
    for i, q in enumerate(queries):
        hits = {n: all_rows[n][i]["hits"]["hit@1"] for n in names}
        if len(set(hits.values())) > 1:
            found = True
            winner = max(hits, key=hits.get)
            ranks = ", ".join(
                f"{n} rank {all_rows[n][i]['first_relevant_rank']}"
                for n in names
            )
            print(f"  {q['id']} [{q['lang']}] -> {winner} wins ({ranks})")
    if not found:
        print("  none - identical hit@1 behavior")


def print_failures(all_rows: dict) -> None:
    for name, rows in all_rows.items():
        failures = [r for r in rows if r["hits"]["hit@1"] == 0]
        print(f"\n{name}: {len(failures)} queries not solved at rank 1")
        for r in failures:
            top = r["retrieved"][0] if r["retrieved"] else None
            got = f"{top['case_id']}" if top else "nothing"
            print(
                f"  {r['id']} [{r['lang']}] expected "
                f"{'/'.join(r['relevant_cases'])} -> top1 {got} "
                f"(first relevant at rank {r['first_relevant_rank']})"
            )


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

    documents = load_documents(KNOWLEDGE_PATH)
    if len(documents) != EXPECTED_POINTS:
        raise SystemExit(
            f"knowledge base yields {len(documents)} documents, expected "
            f"{EXPECTED_POINTS}"
        )

    embedder = Embedder()
    systems = {
        "dense": Retriever(embedder, store),
        "bm25": LexicalRetriever(documents),
    }

    print("Warming up the dense model ...")
    embedder.embed_query("warmup")

    all_rows: dict[str, list[dict]] = {}
    for name, retriever in systems.items():
        print(f"Evaluating {name} on {len(queries)} queries (top_k={TOP_K}) ...")
        all_rows[name] = [evaluate_query(retriever, q) for q in queries]

    print_table(all_rows)
    print_divergences(queries, all_rows)
    print_failures(all_rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "dense_model": embedder.model_name,
            "lexical": "BM25Okapi (rank-bm25), lowercase + NFKD accent "
                       "stripping + alphanumeric tokens",
            "top_k": TOP_K,
            "k_levels": list(K_LEVELS),
            "index_points": count,
            "queries_file": str(QUERIES_PATH),
        },
        "systems": {
            name: {
                "overall": aggregate(rows),
                "by_lang": {
                    lang: aggregate([r for r in rows if r["lang"] == lang])
                    for lang in sorted({r["lang"] for r in rows})
                },
                "latency_ms": latency_stats(rows),
                "per_query": rows,
            }
            for name, rows in all_rows.items()
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nresults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()