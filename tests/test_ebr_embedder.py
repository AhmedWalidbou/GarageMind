"""
Embedder tests - GarageMind M3 EBR-RAG

Two layers:
    - Unit tests with a fake model injected through the lazy-loading seam:
      prefix logic, shapes, input validation. Fast, no weights involved.
    - Integration tests with the real e5 model (locally cached): dimensions,
      normalization, and the cross-lingual FR->EN ranking check promoted
      to a permanent regression test.
"""

import numpy as np
import pytest

from src.ebr.embedder import EMBEDDING_DIM, Embedder


class FakeModel:
    """Records encode() inputs and returns deterministic unit vectors."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        rng = np.random.default_rng(len(self.calls))
        out = rng.normal(size=(len(texts), EMBEDDING_DIM)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def fake_embedder():
    emb = Embedder()
    emb._model = FakeModel()
    return emb


class TestEmbedderUnit:
    def test_model_not_loaded_at_init(self):
        assert Embedder()._model is None

    def test_embed_passages_adds_prefix(self, fake_embedder):
        fake_embedder.embed_passages(["fap colmate", "egr valve stuck"])
        sent = fake_embedder._model.calls[0]
        assert sent == ["passage: fap colmate", "passage: egr valve stuck"]

    def test_embed_query_adds_prefix_and_strips(self, fake_embedder):
        fake_embedder.embed_query("  voyant moteur  ")
        assert fake_embedder._model.calls[0] == ["query: voyant moteur"]

    def test_embed_passages_shape(self, fake_embedder):
        out = fake_embedder.embed_passages(["a", "b", "c"])
        assert out.shape == (3, EMBEDDING_DIM)

    def test_embed_query_shape(self, fake_embedder):
        out = fake_embedder.embed_query("perte de puissance")
        assert out.shape == (EMBEDDING_DIM,)

    def test_empty_passage_list_skips_model(self):
        emb = Embedder()
        out = emb.embed_passages([])
        assert out.shape == (0, EMBEDDING_DIM)
        assert emb._model is None

    def test_blank_passage_rejected_with_index(self, fake_embedder):
        with pytest.raises(ValueError, match="passage 1"):
            fake_embedder.embed_passages(["ok", "   "])

    def test_empty_query_rejected(self, fake_embedder):
        with pytest.raises(ValueError):
            fake_embedder.embed_query("   ")


@pytest.fixture(scope="module")
def real_embedder():
    return Embedder()


class TestEmbedderIntegration:
    def test_dims_and_dtype(self, real_embedder):
        vec = real_embedder.embed_query("voyant moteur allume")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_normalization(self, real_embedder):
        vec = real_embedder.embed_query("perte de puissance a chaud")
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_passage_batch_shape(self, real_embedder):
        out = real_embedder.embed_passages(["turbo whistle", "fumee noire"])
        assert out.shape == (2, EMBEDDING_DIM)

    def test_cross_lingual_ranking(self, real_embedder):
        query = real_embedder.embed_query(
            "voyant moteur allume et perte de puissance"
        )
        passages = real_embedder.embed_passages([
            "Check engine light on with loss of power, limp mode.",
            "Le lecteur CD saute pendant la lecture.",
        ])
        sim_relevant = float(query @ passages[0])
        sim_offtopic = float(query @ passages[1])
        assert sim_relevant > sim_offtopic