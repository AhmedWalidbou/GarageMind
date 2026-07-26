"""
Tests for src/ebr/corpus.py
Covers: schema validation (every rejection path with precise messages),
document flattening, metadata propagation and the real knowledge base.
Validation is the quality boundary of the RAG corpus, so every rule
gets its own failure test.
"""

import json

import pytest

from src.ebr.corpus import (
    CorpusError,
    RepairDocument,
    case_to_documents,
    load_cases,
    load_documents,
)

KB_PATH = "knowledge/repair_cases.json"


def valid_case(case_id: str = "case-test") -> dict:
    return {
        "id": case_id,
        "dtc_codes": ["P0301"],
        "brands": ["renault"],
        "engine_family": "1.5 dCi (K9K)",
        "system": "ignition",
        "symptoms_fr": "Symptômes de test.",
        "symptoms_en": "Test symptoms.",
        "root_cause_fr": "Cause de test.",
        "root_cause_en": "Test cause.",
        "fix_fr": "Réparation de test.",
        "fix_en": "Test fix.",
        "confirmed": True,
    }


def write_kb(tmp_path, cases: list[dict]):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


class TestValidation:
    def test_valid_case_loads(self, tmp_path):
        path = write_kb(tmp_path, [valid_case()])
        assert len(load_cases(path)) == 1

    def test_missing_file_raises(self):
        with pytest.raises(CorpusError, match="not found"):
            load_cases("does/not/exist.json")

    def test_missing_cases_key_raises(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text(json.dumps({"other": []}), encoding="utf-8")
        with pytest.raises(CorpusError, match="'cases'"):
            load_cases(path)

    def test_empty_cases_raises(self, tmp_path):
        path = write_kb(tmp_path, [])
        with pytest.raises(CorpusError, match="no cases"):
            load_cases(path)

    def test_missing_field_raises_with_case_id(self, tmp_path):
        case = valid_case("case-x")
        del case["fix_en"]
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="case-x.*fix_en"):
            load_cases(path)

    def test_duplicate_id_raises(self, tmp_path):
        path = write_kb(tmp_path, [valid_case("dup"), valid_case("dup")])
        with pytest.raises(CorpusError, match="duplicate"):
            load_cases(path)

    def test_invalid_dtc_format_raises(self, tmp_path):
        case = valid_case()
        case["dtc_codes"] = ["Z9999"]
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="invalid DTC"):
            load_cases(path)

    def test_empty_dtc_list_raises(self, tmp_path):
        case = valid_case()
        case["dtc_codes"] = []
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="dtc_codes"):
            load_cases(path)

    def test_empty_brands_raises(self, tmp_path):
        case = valid_case()
        case["brands"] = []
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="brands"):
            load_cases(path)

    def test_blank_text_field_raises(self, tmp_path):
        case = valid_case()
        case["symptoms_fr"] = "   "
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="symptoms_fr"):
            load_cases(path)

    def test_non_bool_confirmed_raises(self, tmp_path):
        case = valid_case()
        case["confirmed"] = "yes"
        path = write_kb(tmp_path, [case])
        with pytest.raises(CorpusError, match="confirmed"):
            load_cases(path)


class TestDocuments:
    def test_two_documents_per_case(self):
        docs = case_to_documents(valid_case())
        assert len(docs) == 2
        assert {d.lang for d in docs} == {"fr", "en"}

    def test_doc_ids_are_case_id_plus_lang(self):
        docs = case_to_documents(valid_case("case-42"))
        assert {d.doc_id for d in docs} == {"case-42-fr", "case-42-en"}

    def test_text_is_monolingual(self):
        docs = {d.lang: d for d in case_to_documents(valid_case())}
        assert "Symptômes" in docs["fr"].text
        assert "Test symptoms." in docs["en"].text
        assert "Symptoms:" not in docs["fr"].text
        assert "Cause racine" not in docs["en"].text

    def test_metadata_propagated(self):
        doc = case_to_documents(valid_case())[0]
        assert doc.dtc_codes == ["P0301"]
        assert doc.brands == ["renault"]
        assert doc.system == "ignition"
        assert doc.engine_family == "1.5 dCi (K9K)"

    def test_dtc_codes_uppercased(self):
        case = valid_case()
        case["dtc_codes"] = ["p0301"]
        doc = case_to_documents(case)[0]
        assert doc.dtc_codes == ["P0301"]
        assert "P0301" in doc.text


class TestRealKnowledgeBase:
    def test_real_kb_loads_without_error(self):
        cases = load_cases(KB_PATH)
        assert len(cases) >= 10

    def test_real_kb_documents_complete(self):
        docs = load_documents(KB_PATH)
        cases = load_cases(KB_PATH)
        assert len(docs) == 2 * len(cases)
        assert all(isinstance(d, RepairDocument) for d in docs)
        assert all(d.text.strip() for d in docs)