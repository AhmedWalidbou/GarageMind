"""
Retriever - GarageMind M3 EBR-RAG
Assembles embedder + vector store into the single entry point the rest
of the system uses: query text in, ranked unique repair cases out.

Design decisions:
    - Search across all documents, no language filter: the multilingual
      embedding space maps FR and EN to the same region (verified
      empirically - a FR query ranks the relevant EN passage above an
      irrelevant FR one), so filtering by query language would only
      discard signal.
    - Deduplicate by case_id, keeping the best-scoring language variant:
      downstream (the M4 agent) reasons about repair *cases*, not about
      language variants of the same case.
    - To guarantee top_k unique cases after dedup, the raw search
      over-fetches (top_k * 2 + 4): each case exists in at most 2
      language variants, so top_k * 2 raw hits are enough in the worst
      case; +4 is headroom.
"""

from dataclasses import dataclass

from src.ebr.embedder import Embedder
from src.ebr.vectorstore import VectorStore


@dataclass
class RetrievedCase:
    """One unique repair case with its best similarity score."""
    case_id: str
    score: float
    lang: str
    text: str
    dtc_codes: list[str]
    brands: list[str]
    system: str
    engine_family: str


class Retriever:
    """Query text in, ranked unique repair cases out."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        """
        Return up to top_k unique cases, best first.

        Raises ValueError on empty query (propagated from the embedder).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        query_vector = self.embedder.embed_query(query)
        raw_hits = self.store.search(query_vector, top_k=top_k * 2 + 4)

        best_by_case: dict[str, RetrievedCase] = {}
        for hit in raw_hits:
            p = hit.payload
            case_id = p["case_id"]
            if case_id not in best_by_case:
                best_by_case[case_id] = RetrievedCase(
                    case_id=case_id,
                    score=hit.score,
                    lang=p["lang"],
                    text=p["text"],
                    dtc_codes=list(p["dtc_codes"]),
                    brands=list(p["brands"]),
                    system=p["system"],
                    engine_family=p["engine_family"],
                )

        ranked = sorted(
            best_by_case.values(), key=lambda c: c.score, reverse=True
        )
        return ranked[:top_k]