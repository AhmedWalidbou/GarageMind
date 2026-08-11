"""
Vector store tests - GarageMind M3 EBR-RAG

All tests run against an in-memory Qdrant with handcrafted basis
vectors: search results are mathematically predictable and no model
is ever loaded, keeping the suite fast. The idempotent re-indexing
property (deterministic UUIDv5 point ids) is covered explicitly -
it is the design decision this layer exists for.
"""

import uuid

import numpy as np
import pytest

from src.ebr.corpus import RepairDocument
from src.ebr.embedder import EMBEDDING_DIM
from src.ebr.vectorstore import VectorStore, point_id_for


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


def basis_vector(axis: int) -> np.ndarray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


@pytest.fixture
def store():
    s = VectorStore(location=":memory:")
    s.ensure_collection()
    return s


class TestPointIds:
    def test_deterministic(self):
        assert point_id_for("case-001-fr") == point_id_for("case-001-fr")

    def test_distinct_for_distinct_doc_ids(self):
        assert point_id_for("case-001-fr") != point_id_for("case-001-en")

    def test_valid_uuid_string(self):
        uuid.UUID(point_id_for("case-001-fr"))


class TestCollection:
    def test_ensure_collection_idempotent(self, store):
        store.ensure_collection()
        assert store.count() == 0


class TestUpsert:
    def test_upsert_and_count(self, store):
        docs = [
            make_doc("case-001-fr", "case-001", "fr"),
            make_doc("case-001-en", "case-001", "en"),
        ]
        vectors = np.stack([basis_vector(0), basis_vector(1)])
        assert store.upsert_documents(docs, vectors) == 2
        assert store.count() == 2

    def test_reindexing_is_idempotent(self, store):
        docs = [make_doc("case-001-fr", "case-001", "fr")]
        vectors = basis_vector(0)[np.newaxis, :]
        store.upsert_documents(docs, vectors)
        store.upsert_documents(docs, vectors)
        assert store.count() == 1

    def test_length_mismatch_rejected(self, store):
        docs = [make_doc("case-001-fr", "case-001", "fr")]
        vectors = np.stack([basis_vector(0), basis_vector(1)])
        with pytest.raises(ValueError, match="1 documents but 2 vectors"):
            store.upsert_documents(docs, vectors)

    def test_wrong_dim_rejected(self, store):
        docs = [make_doc("case-001-fr", "case-001", "fr")]
        vectors = np.ones((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="expected dim"):
            store.upsert_documents(docs, vectors)

    def test_empty_upsert_returns_zero(self, store):
        assert store.upsert_documents([], np.empty((0, EMBEDDING_DIM))) == 0


class TestSearch:
    def test_returns_best_match_first(self, store):
        docs = [
            make_doc("case-001-fr", "case-001", "fr"),
            make_doc("case-002-fr", "case-002", "fr"),
        ]
        vectors = np.stack([basis_vector(0), basis_vector(1)])
        store.upsert_documents(docs, vectors)
        hits = store.search(basis_vector(0), top_k=2)
        assert len(hits) == 2
        assert hits[0].payload["doc_id"] == "case-001-fr"
        assert hits[0].score > hits[1].score

    def test_top_k_respected(self, store):
        docs = [
            make_doc(f"case-00{i}-fr", f"case-00{i}", "fr")
            for i in range(1, 5)
        ]
        vectors = np.stack([basis_vector(i) for i in range(4)])
        store.upsert_documents(docs, vectors)
        assert len(store.search(basis_vector(0), top_k=2)) == 2

    def test_payload_round_trip(self, store):
        doc = make_doc("case-001-fr", "case-001", "fr")
        store.upsert_documents([doc], basis_vector(0)[np.newaxis, :])
        payload = store.search(basis_vector(0), top_k=1)[0].payload
        assert payload["doc_id"] == "case-001-fr"
        assert payload["case_id"] == "case-001"
        assert payload["lang"] == "fr"
        assert payload["dtc_codes"] == ["P2002"]
        assert payload["system"] == "exhaust_aftertreatment"