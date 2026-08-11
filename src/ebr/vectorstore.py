"""
Vector store - GarageMind M3 EBR-RAG
Thin wrapper around Qdrant (embedded mode) for storing and searching
repair-case documents.

Design decisions:
    - Embedded Qdrant, no Docker: a local path persists the index on
      disk, ":memory:" gives tests a throwaway instance with the exact
      same code path. One class, two lifecycles.
    - Qdrant only accepts UUIDs or unsigned integers as point ids, so
      the readable doc_id ("case-001-fr") is mapped to a deterministic
      UUIDv5. Same doc_id -> same point id -> re-indexing is idempotent
      (upsert overwrites instead of duplicating). The readable id stays
      in the payload.
    - This layer never embeds: it stores vectors it is given. That
      separation keeps it testable with handcrafted vectors, no model
      involved.
    - COSINE distance on L2-normalized vectors (the embedder guarantees
      normalization), so scores stay comparable across queries.
"""

import uuid
from dataclasses import asdict, dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ebr.corpus import RepairDocument
from src.ebr.embedder import EMBEDDING_DIM

COLLECTION_NAME = "repair_cases"

# Fixed namespace so doc_id -> point id stays stable across runs and machines.
_POINT_NAMESPACE = uuid.UUID("b7e6a1f0-4c2d-4d8e-9f3a-2b1c0d9e8f7a")


def point_id_for(doc_id: str) -> str:
    """Deterministic UUIDv5 point id derived from a readable doc_id."""
    return str(uuid.uuid5(_POINT_NAMESPACE, doc_id))


@dataclass
class SearchHit:
    """One search result: similarity score plus the stored payload."""
    score: float
    payload: dict


class VectorStore:
    """Stores repair documents with their vectors, serves similarity search."""

    def __init__(self, location: str = ":memory:", collection: str = COLLECTION_NAME):
        if location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=location)
        self.collection = collection

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist (idempotent)."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )

    def upsert_documents(
        self, documents: list[RepairDocument], vectors: np.ndarray
    ) -> int:
        """
        Insert or overwrite documents with their vectors.

        Returns the number of points written. Raises ValueError on any
        mismatch between documents and vectors.
        """
        if len(documents) != len(vectors):
            raise ValueError(
                f"{len(documents)} documents but {len(vectors)} vectors"
            )
        if len(documents) == 0:
            return 0
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"expected dim {EMBEDDING_DIM}, got {vectors.shape[1]}"
            )
        points = [
            PointStruct(
                id=point_id_for(doc.doc_id),
                vector=vec.tolist(),
                payload=asdict(doc),
            )
            for doc, vec in zip(documents, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """Return the top_k most similar documents, best first."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(score=p.score, payload=p.payload) for p in response.points
        ]

    def count(self) -> int:
        """Number of points currently stored."""
        return self.client.count(self.collection, exact=True).count