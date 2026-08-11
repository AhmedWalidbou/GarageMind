"""
Embedder - GarageMind M3 EBR-RAG
Wraps the multilingual sentence-transformers model behind a minimal,
testable interface.

Design decisions:
    - e5 models require asymmetric prefixes: "query: " for user queries,
      "passage: " for indexed documents. Forgetting them silently degrades
      retrieval quality (a classic e5 pitfall), so prefixing lives here,
      in one place - never in the corpus, never in callers.
    - L2-normalized embeddings: cosine similarity becomes a plain dot
      product, matching the COSINE distance of the vector store and
      keeping scores comparable across queries.
    - Lazy model loading: importing the module stays cheap; the model
      (~470 MB of weights) loads on first use only. Tests that do not
      need the real model inject a fake through the same interface.
    - CPU device pinned explicitly: reproducible behavior on the dev
      machine, no silent device lookup. The M4 agent will embed one
      query per reasoning turn, so the small 384-dim model keeps that
      latency acceptable.
"""

from collections.abc import Sequence

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


class Embedder:
    """Encodes queries and passages with the e5 prefix convention."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazily load and cache the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed documents for indexing. Returns shape (n, EMBEDDING_DIM).

        Empty input returns an empty (0, EMBEDDING_DIM) array without
        touching the model.
        """
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"passage {i} is empty")
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one user query. Returns shape (EMBEDDING_DIM,)."""
        if not text or not text.strip():
            raise ValueError("query text is empty")
        return self._encode([f"query: {text.strip()}"])[0]