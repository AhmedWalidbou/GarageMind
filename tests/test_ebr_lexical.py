"""
Lexical retriever tests - GarageMind M3 EBR-RAG

Handcrafted mini-corpora make BM25 outcomes predictable: a token
present in a single document dominates its ranking (IDF), so every
assertion is deterministic.

Signature scenarios:
    - Exact DTC code in the query must put the right case first -
      the property this baseline exists to test (eval failure q-022).
    - Accent-insensitive matching: unaccented queries must match the
      accented corpus (fair-tokenization guarantee).

Note: micro-corpora must stay non-degenerate. With N=2 documents and
a token present in one of them (df=1), BM25Okapi's IDF is
ln((2-1+0.5)/(1+0.5)) = ln(1) = 0 and every score collapses to 0.
Each retrieval test therefore includes at least one distractor from
another case.
"""

import pytest

from src.ebr.corpus import RepairDocument
from src.ebr.lexical import LexicalRetriever, tokenize


def make_doc(doc_id: str, case_id: str, lang: str, text: str) -> RepairDocument:
    return RepairDocument(
        doc_id=doc_id,
        case_id=case_id,
        lang=lang,
        text=text,
        dtc_codes=["P0301"],
        brands=["vw"],
        system="ignition",
        engine_family="1.4 TSI (CZCA)",
    )


class TestTokenize:
    def test_lowercase(self):
        assert tokenize("Voyant MOTEUR") == ["voyant", "moteur"]

    def test_accents_stripped(self):
        assert tokenize("régénération dégradé") == ["regeneration", "degrade"]

    def test_dtc_code_survives_whole(self):
        assert tokenize("code P0301 stocké") == ["code", "p0301", "stocke"]

    def test_punctuation_splits_tokens(self):
        assert tokenize("cale, puis... redémarre!") == [
            "cale", "puis", "redemarre"
        ]

    def test_symbols_only_yield_nothing(self):
        assert tokenize("!!! ???") == []


class TestRetrieval:
    def test_exact_dtc_code_dominates(self):
        docs = [
            make_doc("case-001-en", "case-001", "en",
                     "Fault codes: P0301. Misfire on cylinder one when warm."),
            make_doc("case-002-en", "case-002", "en",
                     "Fault codes: P0299. Turbo underboost and whistle."),
            make_doc("case-003-en", "case-003", "en",
                     "Fault codes: P2002. Particulate filter efficiency low."),
        ]
        results = LexicalRetriever(docs).retrieve(
            "what usually causes a P0301 code", top_k=3
        )
        assert results[0].case_id == "case-001"

    def test_accent_insensitive_match(self):
        docs = [
            make_doc("case-001-fr", "case-001", "fr",
                     "Régénérations fréquentes du filtre à particules."),
            make_doc("case-002-fr", "case-002", "fr",
                     "Boîte automatique à-coups entre les rapports."),
            make_doc("case-003-fr", "case-003", "fr",
                     "Alternateur en fin de vie, tension instable."),
        ]
        results = LexicalRetriever(docs).retrieve(
            "regenerations frequentes", top_k=1
        )
        assert results[0].case_id == "case-001"

    def test_dedup_one_result_per_case(self):
        docs = [
            make_doc("case-001-fr", "case-001", "fr",
                     "Codes défaut: P0301. Ratés d'allumage à chaud."),
            make_doc("case-001-en", "case-001", "en",
                     "Fault codes: P0301. Misfire when warm."),
            make_doc("case-002-fr", "case-002", "fr",
                     "Codes défaut: P0299. Turbo qui siffle."),
        ]
        results = LexicalRetriever(docs).retrieve("P0301", top_k=3)
        case_ids = [r.case_id for r in results]
        assert case_ids.count("case-001") == 1

    def test_dedup_keeps_best_variant(self):
        # The third document (another case) keeps IDF meaningful:
        # on a 2-doc corpus, df=1 out of N=2 gives IDF=0 and every
        # score collapses to 0 (degenerate micro-corpus).
        docs = [
            make_doc("case-001-fr", "case-001", "fr",
                     "Codes défaut: P0301. Ratés d'allumage cylindre un."),
            make_doc("case-001-en", "case-001", "en",
                     "Fault codes: P0301. Misfire on cylinder one when warm."),
            make_doc("case-002-fr", "case-002", "fr",
                     "Codes défaut: P0299. Turbo qui siffle en montée."),
        ]
        results = LexicalRetriever(docs).retrieve(
            "misfire cylinder one warm", top_k=1
        )
        assert results[0].case_id == "case-001"
        assert results[0].lang == "en"

    def test_ranked_best_first(self):
        docs = [
            make_doc("case-001-fr", "case-001", "fr",
                     "Turbo qui siffle, pression instable, mode dégradé."),
            make_doc("case-002-fr", "case-002", "fr",
                     "Remplacement du turbo effectué récemment."),
            make_doc("case-003-fr", "case-003", "fr",
                     "Batterie déchargée après une nuit au froid."),
        ]
        results = LexicalRetriever(docs).retrieve(
            "turbo siffle pression", top_k=2
        )
        assert [r.case_id for r in results] == ["case-001", "case-002"]
        assert results[0].score > results[1].score

    def test_top_k_limits_results(self):
        docs = [
            make_doc(f"case-00{i}-fr", f"case-00{i}", "fr",
                     f"Panne numero {i} moteur diesel.")
            for i in range(1, 4)
        ]
        assert len(LexicalRetriever(docs).retrieve("moteur", top_k=2)) == 2

    def test_payload_fields_carried_over(self):
        docs = [
            make_doc("case-001-en", "case-001", "en",
                     "Fault codes: P0301. Misfire on cylinder one."),
            make_doc("case-002-en", "case-002", "en",
                     "Fault codes: P0299. Turbo underboost on motorway."),
        ]
        r = LexicalRetriever(docs).retrieve("misfire", top_k=1)[0]
        assert r.text == "Fault codes: P0301. Misfire on cylinder one."
        assert r.dtc_codes == ["P0301"]
        assert r.brands == ["vw"]
        assert r.system == "ignition"
        assert r.engine_family == "1.4 TSI (CZCA)"


class TestValidation:
    def test_empty_documents_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            LexicalRetriever([])

    def test_empty_query_rejected(self):
        docs = [make_doc("case-001-fr", "case-001", "fr", "Panne moteur.")]
        with pytest.raises(ValueError):
            LexicalRetriever(docs).retrieve("   ")

    def test_symbol_only_query_rejected(self):
        docs = [make_doc("case-001-fr", "case-001", "fr", "Panne moteur.")]
        with pytest.raises(ValueError, match="no indexable tokens"):
            LexicalRetriever(docs).retrieve("!!! ???")

    def test_top_k_below_one_rejected(self):
        docs = [make_doc("case-001-fr", "case-001", "fr", "Panne moteur.")]
        with pytest.raises(ValueError, match="top_k"):
            LexicalRetriever(docs).retrieve("moteur", top_k=0)