"""
Tests for knowledge/eval_queries.json
The evaluation query set is what Recall@k and MRR will be computed against,
so its integrity conditions the validity of every retrieval metric:
every referenced case must exist, ids must be unique, queries non-empty.
"""

import json

import pytest

from src.ebr.corpus import load_cases

KB_PATH = "knowledge/repair_cases.json"
QUERIES_PATH = "knowledge/eval_queries.json"


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "queries" in data and isinstance(data["queries"], list)
    return data["queries"]


@pytest.fixture(scope="module")
def case_ids() -> set[str]:
    return {case["id"] for case in load_cases(KB_PATH)}


class TestQuerySetIntegrity:
    def test_at_least_20_queries(self, queries):
        assert len(queries) >= 20

    def test_query_ids_unique(self, queries):
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids))

    def test_required_fields_present(self, queries):
        for q in queries:
            assert set(q) >= {"id", "lang", "query", "relevant_cases"}, q.get("id")

    def test_queries_non_empty(self, queries):
        for q in queries:
            assert q["query"].strip(), q["id"]

    def test_langs_are_fr_or_en(self, queries):
        for q in queries:
            assert q["lang"] in ("fr", "en"), q["id"]

    def test_both_languages_represented(self, queries):
        langs = {q["lang"] for q in queries}
        assert langs == {"fr", "en"}

    def test_relevant_cases_non_empty(self, queries):
        for q in queries:
            assert isinstance(q["relevant_cases"], list) and q["relevant_cases"], q["id"]

    def test_every_relevant_case_exists(self, queries, case_ids):
        for q in queries:
            for case_id in q["relevant_cases"]:
                assert case_id in case_ids, (
                    f"{q['id']} references unknown case '{case_id}'"
                )

    def test_no_duplicate_relevant_cases_within_query(self, queries):
        for q in queries:
            assert len(q["relevant_cases"]) == len(set(q["relevant_cases"])), q["id"]

    def test_every_case_covered_by_at_least_one_query(self, queries, case_ids):
        covered = {cid for q in queries for cid in q["relevant_cases"]}
        uncovered = case_ids - covered
        assert not uncovered, f"cases never referenced by any query: {sorted(uncovered)}"