"""
Retriever tests - GarageMind M3 EBR-RAG

Setup: the real Embedder (its input validation is exercised for real)
with a fake model injected through the lazy-loading seam, returning a
preset query vector; a real in-memory Qdrant filled with handcrafted
vectors whose cosine similarity to that query is chosen exactly.
Every expected score is therefore predictable.

Key scenario: both language variants of a case land in the raw hits
-> exactly one case comes out, carrying the best-scoring variant.
"""

import numpy as np
import pytest

from src.ebr.corpus import RepairDocument
from src.ebr.embedder import EMBEDDING_DIM, Embedder
from src.ebr.retriever import Retriever
from src.ebr.vectorstore import VectorStore

QUERY_AXIS = 0


class FixedVectorModel:
    """Fake sentence-transformers model returning a preset unit vector."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector

    def encode(self, texts, **kwargs):
        return np.stack([self.vector] * len(texts))


def make_doc(doc_id: str, case_id: str, lang: str) -> RepairDocument:
    return RepairDocument(
        doc_id=doc_id,
        case_id=case_id,
        lang=lang,
        text=f"text for {doc_id}",
        dtc_codes=["P2002"],
        brands=["peugeot"],
        system="exhaust_aftertreatment",
        engine_family="1.6 BlueHDi (DV6FD)",
    )


def vec(cos_sim: float, other_axis: int) -> np.ndarray:
    """Unit vector whose cosine similarity with the query axis is cos_sim."""
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[QUERY_AXIS] = cos_sim
    v[other_axis] = np.sqrt(1.0 - cos_sim ** 2)
    return v


def index(store: VectorStore, entries: list[tuple]) -> None:
    """entries: (doc_id, case_id, lang, cos_sim, other_axis)"""
    docs = [make_doc(d, c, lg) for d, c, lg, _, _ in entries]
    vectors = np.stack([vec(cs, ax) for _, _, _, cs, ax in entries])
    store.upsert_documents(docs, vectors)


@pytest.fixture
def store():
    s = VectorStore(location=":memory:")
    s.ensure_collection()
    return s


@pytest.fixture
def retriever(store):
    emb = Embedder()
    query_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query_vector[QUERY_AXIS] = 1.0
    emb._model = FixedVectorModel(query_vector)
    return Retriever(emb, store)


class TestDedup:
    def test_keeps_best_scoring_variant(self, store, retriever):
        index(store, [
            ("case-001-fr", "case-001", "fr", 0.95, 1),
            ("case-001-en", "case-001", "en", 0.80, 2),
        ])
        results = retriever.retrieve("voyant moteur", top_k=3)
        assert len(results) == 1
        assert results[0].case_id == "case-001"
        assert results[0].lang == "fr"
        assert results[0].score == pytest.approx(0.95, abs=1e-3)

    def test_counts_unique_cases_not_documents(self, store, retriever):
        index(store, [
            ("case-001-fr", "case-001", "fr", 0.95, 1),
            ("case-001-en", "case-001", "en", 0.90, 2),
            ("case-002-fr", "case-002", "fr", 0.85, 3),
            ("case-002-en", "case-002", "en", 0.80, 4),
        ])
        results = retriever.retrieve("q", top_k=5)
        assert [r.case_id for r in results] == ["case-001", "case-002"]


class TestRanking:
    def test_ranked_best_first_across_languages(self, store, retriever):
        index(store, [
            ("case-001-fr", "case-001", "fr", 0.70, 1),
            ("case-002-en", "case-002", "en", 0.90, 2),
            ("case-003-fr", "case-003", "fr", 0.50, 3),
            ("case-002-fr", "case-002", "fr", 0.60, 4),
            ("case-001-en", "case-001", "en", 0.80, 5),
        ])
        results = retriever.retrieve("q", top_k=3)
        assert [r.case_id for r in results] == [
            "case-002", "case-001", "case-003"
        ]
        assert [r.lang for r in results] == ["en", "en", "fr"]
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, store, retriever):
        index(store, [
            ("case-001-fr", "case-001", "fr", 0.90, 1),
            ("case-002-fr", "case-002", "fr", 0.80, 2),
            ("case-003-fr", "case-003", "fr", 0.70, 3),
        ])
        assert len(retriever.retrieve("q", top_k=2)) == 2


class TestValidation:
    def test_top_k_below_one_rejected(self, retriever):
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve("q", top_k=0)

    def test_empty_query_rejected(self, retriever):
        with pytest.raises(ValueError):
            retriever.retrieve("   ")


class TestFieldMapping:
    def test_payload_fields_carried_over(self, store, retriever):
        index(store, [("case-001-fr", "case-001", "fr", 0.95, 1)])
        r = retriever.retrieve("q", top_k=1)[0]
        assert r.text == "text for case-001-fr"
        assert r.dtc_codes == ["P2002"]
        assert r.brands == ["peugeot"]
        assert r.system == "exhaust_aftertreatment"
        assert r.engine_family == "1.6 BlueHDi (DV6FD)"