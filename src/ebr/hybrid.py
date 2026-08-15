"""
Hybrid retriever (RRF fusion) - GarageMind M3 EBR-RAG
Fuses the dense and lexical retrievers with Reciprocal Rank Fusion.

Design decisions:
    - RRF works on ranks only, never on raw scores: cosine similarities
      and BM25 scores live on incomparable scales (documented constraint
      of both retrievers). Each system contributes 1/(k + rank) per
      case; the sum ranks the fusion.
    - k=60 is the standard constant from the RRF literature and is
      deliberately NOT tuned: tuning it on the 25-query eval set would
      turn the evaluation into training data.
    - Motivated by measured complementarity, not hope: q-016 and q-022
      are solved only by BM25, q-021 only by dense - a case ranked
      first by one system and missed by the other outranks a case
      ranked mid-pack by both.
    - Same interface as both underlying retrievers, so the shared
      evaluation harness accepts it unchanged.
    - Each underlying system is queried with an over-fetch depth
      (top_k * 2 + 4 unique cases) so fusion sees enough of both
      rankings to disagree meaningfully.
"""

from src.ebr.retriever import RetrievedCase

RRF_K = 60


class HybridRetriever:
    """Rank-level fusion of the dense and lexical retrievers."""

    def __init__(self, dense, lexical, rrf_k: int = RRF_K):
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        """
        Return up to top_k unique cases, best RRF score first.

        Raises ValueError on empty query or top_k < 1 (propagated from
        the underlying retrievers, same contract).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        depth = top_k * 2 + 4
        rankings = [
            self.dense.retrieve(query, top_k=depth),
            self.lexical.retrieve(query, top_k=depth),
        ]

        rrf_scores: dict[str, float] = {}
        best_payload: dict[str, RetrievedCase] = {}
        for ranking in rankings:
            for rank, case in enumerate(ranking, start=1):
                rrf_scores[case.case_id] = (
                    rrf_scores.get(case.case_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )
                if case.case_id not in best_payload:
                    best_payload[case.case_id] = case

        ranked_ids = sorted(
            rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True
        )

        fused = []
        for cid in ranked_ids[:top_k]:
            base = best_payload[cid]
            fused.append(
                RetrievedCase(
                    case_id=base.case_id,
                    score=round(rrf_scores[cid], 6),
                    lang=base.lang,
                    text=base.text,
                    dtc_codes=list(base.dtc_codes),
                    brands=list(base.brands),
                    system=base.system,
                    engine_family=base.engine_family,
                )
            )
        return fused