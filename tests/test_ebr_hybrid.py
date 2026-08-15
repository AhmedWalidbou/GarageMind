"""
Hybrid retriever tests - GarageMind M3 EBR-RAG

Stub retrievers with preset rankings make every RRF score computable
by hand (k=60): each assertion pins an exact arithmetic outcome.

True RRF properties encoded here:
    - Consensus beats a single enthusiastic vote: rank 2 in both
      systems (2/62) outranks rank 1 in one system only (1/61).
    - The q-016 profile is rescued: a case ranked 5th by dense and
      1st by BM25 (1/65 + 1/61) beats a distractor ranked 1st by
      dense and 8th by BM25 (1/61 + 1/68).
"""

import pytest

from src.ebr.hybrid import RRF_K, HybridRetriever
from src.ebr.retriever import RetrievedCase


def make_case(case_id: str, lang: str = "fr") -> RetrievedCase:
    return RetrievedCase(
        case_id=case_id,
        score=1.0,
        lang=lang,
        text=f"text for {case_id}",
        dtc_codes=["P0301"],
        brands=["vw"],
        system="ignition",
        engine_family="1.4 TSI (CZCA)",
    )


class StubRetriever:
    """Returns a preset ranking; records the top_k it was asked for."""

    def __init__(self, ranking: list[RetrievedCase]):
        self.ranking = ranking
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        if not query or not query.strip():
            raise ValueError("query text is empty")
        self.last_top_k = top_k
        return self.ranking[:top_k]


def hybrid(dense_ids: list[str], lexical_ids: list[str]) -> HybridRetriever:
    return HybridRetriever(
        dense=StubRetriever([make_case(c) for c in dense_ids]),
        lexical=StubRetriever([make_case(c) for c in lexical_ids]),
    )


class TestRRFMath:
    def test_rank_one_in_both_scores_two_over_61(self):
        h = hybrid(["case-A", "case-B"], ["case-A", "case-C"])
        results = h.retrieve("q", top_k=3)
        assert results[0].case_id == "case-A"
        assert results[0].score == pytest.approx(2.0 / (RRF_K + 1), abs=1e-6)

    def test_consensus_beats_single_vote(self):
        # case-B: rank 2 in both -> 2/62; case-A: rank 1 dense only -> 1/61
        h = hybrid(["case-A", "case-B"], ["case-C", "case-B"])
        results = h.retrieve("q", top_k=3)
        assert results[0].case_id == "case-B"
        assert results[0].score == pytest.approx(2.0 / 62.0, abs=1e-6)

    def test_q016_profile_rescued(self):
        # Expected case: dense rank 5 + bm25 rank 1 = 1/65 + 1/61
        # Distractor:    dense rank 1 + bm25 rank 8 = 1/61 + 1/68
        dense_ids = ["case-D", "f1", "f2", "f3", "case-E"]
        lexical_ids = ["case-E", "f4", "f5", "f6", "f7", "f8", "f9", "case-D"]
        results = hybrid(dense_ids, lexical_ids).retrieve("q", top_k=2)
        assert results[0].case_id == "case-E"
        assert results[0].score == pytest.approx(
            1.0 / 65.0 + 1.0 / 61.0, abs=1e-6
        )
        assert results[1].case_id == "case-D"
        assert results[1].score == pytest.approx(
            1.0 / 61.0 + 1.0 / 68.0, abs=1e-6
        )


class TestFusionBehavior:
    def test_over_fetch_depth(self):
        h = hybrid(["case-A"], ["case-A"])
        h.retrieve("q", top_k=3)
        assert h.dense.last_top_k == 10
        assert h.lexical.last_top_k == 10

    def test_top_k_limits_results(self):
        h = hybrid(["a", "b", "c"], ["c", "d", "e"])
        assert len(h.retrieve("q", top_k=2)) == 2

    def test_results_sorted_best_first(self):
        h = hybrid(["a", "b", "c"], ["b", "c", "a"])
        results = h.retrieve("q", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_payload_fields_carried_over(self):
        h = hybrid(["case-A"], ["case-A"])
        r = h.retrieve("q", top_k=1)[0]
        assert r.text == "text for case-A"
        assert r.dtc_codes == ["P0301"]
        assert r.brands == ["vw"]
        assert r.system == "ignition"
        assert r.engine_family == "1.4 TSI (CZCA)"


class TestValidation:
    def test_top_k_below_one_rejected(self):
        h = hybrid(["case-A"], ["case-A"])
        with pytest.raises(ValueError, match="top_k"):
            h.retrieve("q", top_k=0)

    def test_empty_query_propagated(self):
        h = hybrid(["case-A"], ["case-A"])
        with pytest.raises(ValueError):
            h.retrieve("   ")