"""
Lexical retriever (BM25 baseline) - GarageMind M3 EBR-RAG
BM25 over the same 40 monolingual documents, exposed behind the exact
same interface as the dense retriever.

Design decisions:
    - Interface parity: retrieve(query, top_k) -> list[RetrievedCase],
      the same return type as the dense Retriever. The evaluation
      harness runs both retrievers through identical code, so the
      protocols cannot diverge by construction.
    - Fair tokenization, documented: lowercase + NFKD accent stripping
      + alphanumeric runs. DTC codes ("P0301" -> "p0301") survive as
      whole tokens - the exact lexical anchor this baseline exists to
      test (eval failure q-022).
    - BM25 has zero cross-lingual power: a FR query only matches FR
      tokens, plus tokens shared across languages (DTC codes, engine
      families, model names). Both language variants stay indexed and
      deduplicated by case_id, exactly like the dense side - measuring
      where that hurts is part of the benchmark.
    - BM25 scores are unbounded and corpus-dependent, not comparable
      to cosine similarities: only ranks and rank-based metrics are
      compared across retrievers, never raw scores.
"""

import re
import unicodedata

from rank_bm25 import BM25Okapi

from src.ebr.corpus import RepairDocument
from src.ebr.retriever import RetrievedCase

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (NFKD), keep alphanumeric runs."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _TOKEN_PATTERN.findall(text)


class LexicalRetriever:
    """BM25 baseline with the same interface as the dense Retriever."""

    def __init__(self, documents: list[RepairDocument]):
        if not documents:
            raise ValueError("documents list is empty")
        self.documents = documents
        self._bm25 = BM25Okapi([tokenize(d.text) for d in documents])

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        """
        Return up to top_k unique cases, best first.

        Raises ValueError on empty query, empty token stream or
        top_k < 1 (same contract as the dense retriever).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query or not query.strip():
            raise ValueError("query text is empty")

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query has no indexable tokens")

        scores = self._bm25.get_scores(query_tokens)
        order = sorted(
            range(len(self.documents)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )

        best_by_case: dict[str, RetrievedCase] = {}
        for i in order:
            doc = self.documents[i]
            if doc.case_id not in best_by_case:
                best_by_case[doc.case_id] = RetrievedCase(
                    case_id=doc.case_id,
                    score=float(scores[i]),
                    lang=doc.lang,
                    text=doc.text,
                    dtc_codes=list(doc.dtc_codes),
                    brands=list(doc.brands),
                    system=doc.system,
                    engine_family=doc.engine_family,
                )

        ranked = sorted(
            best_by_case.values(), key=lambda c: c.score, reverse=True
        )
        return ranked[:top_k]