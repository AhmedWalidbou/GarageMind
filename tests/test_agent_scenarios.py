"""
Agent scenario integrity tests - GarageMind M4

The evaluation set is written before the agent exists, so nothing can be
tuned toward it after the fact. These tests guard the set itself: a
scenario referencing a case id that is not in the knowledge base would
score the agent against a phantom answer.

Tool names are checked for shape only at this stage; once the tool
registry lands they will be checked against it.
"""

import json
from pathlib import Path

import pytest

from src.ebr.corpus import load_cases

SCENARIOS_PATH = Path("knowledge/agent_scenarios.json")
KNOWLEDGE_PATH = Path("knowledge/repair_cases.json")

REQUIRED_FIELDS = [
    "id", "type", "input", "expected_tools",
    "expected_cases", "expected_keywords", "must_decline",
]

VALID_TYPES = {
    "simple",
    "composite",
    "trap_out_of_domain",
    "trap_insufficient_info",
    "trap_unknown_code",
}

PLANNED_TOOLS = {
    "search_repair_cases",
    "decode_dtc",
    "decode_vin",
    "analyze_can_log",
    "detect_anomalies",
}


@pytest.fixture(scope="module")
def scenarios():
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)["scenarios"]


@pytest.fixture(scope="module")
def known_case_ids():
    return {case["id"] for case in load_cases(KNOWLEDGE_PATH)}


class TestSchema:
    def test_set_is_not_empty(self, scenarios):
        assert len(scenarios) >= 10

    def test_all_required_fields_present(self, scenarios):
        for s in scenarios:
            for field in REQUIRED_FIELDS:
                assert field in s, f"{s.get('id', '?')}: missing '{field}'"

    def test_ids_are_unique(self, scenarios):
        ids = [s["id"] for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_types_are_valid(self, scenarios):
        for s in scenarios:
            assert s["type"] in VALID_TYPES, f"{s['id']}: unknown type"

    def test_inputs_are_substantial(self, scenarios):
        for s in scenarios:
            assert len(s["input"].strip()) >= 15, f"{s['id']}: input too short"

    def test_list_fields_are_lists(self, scenarios):
        for s in scenarios:
            assert isinstance(s["expected_tools"], list)
            assert isinstance(s["expected_cases"], list)
            assert isinstance(s["expected_keywords"], list)

    def test_must_decline_is_boolean(self, scenarios):
        for s in scenarios:
            assert isinstance(s["must_decline"], bool)


class TestGrounding:
    def test_expected_cases_exist_in_knowledge_base(
        self, scenarios, known_case_ids
    ):
        """The core guard: no scenario may point at a phantom case."""
        for s in scenarios:
            for case_id in s["expected_cases"]:
                assert case_id in known_case_ids, (
                    f"{s['id']}: case '{case_id}' is not in the knowledge base"
                )

    def test_expected_tools_are_planned(self, scenarios):
        for s in scenarios:
            for tool in s["expected_tools"]:
                assert tool in PLANNED_TOOLS, f"{s['id']}: unknown tool '{tool}'"

    def test_keywords_are_lowercase(self, scenarios):
        """Conclusion matching is case-insensitive; keywords stay lowercase."""
        for s in scenarios:
            for kw in s["expected_keywords"]:
                assert kw == kw.lower(), f"{s['id']}: keyword '{kw}' not lowercase"


class TestConsistency:
    def test_declining_scenarios_expect_no_case(self, scenarios):
        for s in scenarios:
            if s["must_decline"]:
                assert not s["expected_cases"], (
                    f"{s['id']}: a declining scenario cannot expect a case"
                )

    def test_declining_scenarios_expect_no_keyword(self, scenarios):
        for s in scenarios:
            if s["must_decline"]:
                assert not s["expected_keywords"], (
                    f"{s['id']}: a declining scenario cannot expect keywords"
                )

    def test_non_trap_scenarios_use_tools(self, scenarios):
        for s in scenarios:
            if s["type"] in {"simple", "composite"}:
                assert s["expected_tools"], f"{s['id']}: expected at least one tool"

    def test_composite_scenarios_chain_tools(self, scenarios):
        """A composite scenario is defined by needing more than one tool."""
        for s in scenarios:
            if s["type"] == "composite":
                assert len(s["expected_tools"]) >= 2, (
                    f"{s['id']}: composite but only one tool expected"
                )


class TestCoverage:
    def test_traps_are_present(self, scenarios):
        traps = [s for s in scenarios if s["type"].startswith("trap_")]
        assert len(traps) >= 3

    def test_both_solvable_types_present(self, scenarios):
        types = {s["type"] for s in scenarios}
        assert "simple" in types
        assert "composite" in types

    def test_every_planned_tool_is_exercised(self, scenarios):
        """Documents which tools the evaluation actually covers."""
        used = {t for s in scenarios for t in s["expected_tools"]}
        assert "search_repair_cases" in used
        assert "decode_dtc" in used
        assert "decode_vin" in used