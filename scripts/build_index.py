"""
Build the persistent Qdrant index - GarageMind M3 EBR-RAG

Embeds the 40 knowledge-base documents with the e5 model and writes
them to the on-disk Qdrant index (data/qdrant, gitignored), then runs
one end-to-end smoke retrieval against the freshly built index.

Design decisions:
    - Re-runnable by construction: deterministic UUIDv5 point ids make
      every run an overwrite, never a duplication. Rebuilding after a
      knowledge-base edit is the intended workflow.
    - The smoke retrieval at the end makes the script self-validating:
      it exercises corpus -> embedder -> store -> retriever on real
      data in one command.
"""

import time
from pathlib import Path

from src.ebr.corpus import load_documents
from src.ebr.embedder import Embedder
from src.ebr.retriever import Retriever
from src.ebr.vectorstore import VectorStore

KNOWLEDGE_PATH = Path("knowledge/repair_cases.json")
INDEX_PATH = Path("data/qdrant")
SMOKE_QUERY = "voyant moteur allume et perte de puissance en montee"


def main() -> None:
    print(f"Loading knowledge base from {KNOWLEDGE_PATH} ...")
    documents = load_documents(KNOWLEDGE_PATH)
    print(f"  {len(documents)} documents from {len(documents) // 2} cases")

    print("Loading embedding model ...")
    embedder = Embedder()

    print("Embedding passages ...")
    t0 = time.perf_counter()
    vectors = embedder.embed_passages([d.text for d in documents])
    elapsed = time.perf_counter() - t0
    print(f"  {vectors.shape[0]} vectors, dim {vectors.shape[1]}, {elapsed:.1f}s")

    print(f"Writing index to {INDEX_PATH} ...")
    store = VectorStore(location=str(INDEX_PATH))
    store.ensure_collection()
    written = store.upsert_documents(documents, vectors)
    print(f"  upserted {written} points, collection count = {store.count()}")

    print(f"\nSmoke retrieval: '{SMOKE_QUERY}'")
    retriever = Retriever(embedder, store)
    for rank, case in enumerate(retriever.retrieve(SMOKE_QUERY, top_k=3), 1):
        print(
            f"  {rank}. {case.case_id} [{case.lang}] "
            f"score={case.score:.4f} system={case.system}"
        )


if __name__ == "__main__":
    main()