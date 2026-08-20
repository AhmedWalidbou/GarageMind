"""
Tests for the ReAct graph - GarageMind M4

Every test runs on FakeBackend: no network, no API key, milliseconds.
The tool layer is stubbed where the test is about the graph's control
flow rather than about the tools themselves.
"""

import json
import pytest

from src.agent import graph as graph_module
from src.agent.backend import FakeBackend, LLMResponse
from src.agent.graph import (
    MAX_TURNS,
    SYSTEM_PROMPT,
    TURN_LIMIT_ANSWER,
    AgentResult,
    already_called,
    assistant_tool_message,
    build_initial_state,
    run_agent,
    tool_result_message,
)


@pytest.fixture
def stub_tool(monkeypatch):
    """Replace call_tool with a recorder returning a canned result."""
    calls = []

    def fake_call_tool(name, arguments):
        calls.append((name, arguments))
        return f"result for {name}"

    monkeypatch.setattr(graph_module, "call_tool", fake_call_tool)
    return calls


def tool_turn(name, arguments, call_id="tc_1"):
    return LLMResponse(tool_name=name, tool_arguments=arguments, tool_call_id=call_id)


# --- Initial state ---

class TestInitialState:

    def test_starts_with_the_system_prompt_then_the_question(self):
        state = build_initial_state("P0420 sur Clio")
        assert state["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert state["messages"][1] == {"role": "user", "content": "P0420 sur Clio"}

    def test_counters_start_empty(self):
        state = build_initial_state("hi")
        assert state["turns"] == 0
        assert state["trace"] == []
        assert state["answer"] == ""
        assert state["stop_reason"] == ""
        assert state["pending"] is None

    def test_the_system_prompt_carries_the_anti_trap_rules(self):
        """These four rules are what the trap scenarios score."""
        lowered = " ".join(SYSTEM_PROMPT.lower().split())
        assert "outside your scope" in lowered
        assert "ask for the specific missing" in lowered
        assert "do not have a definition" in lowered
        assert "never cite an identifier" in lowered


# --- Message builders ---

class TestMessageBuilders:

    def test_the_assistant_message_replays_the_call_id(self):
        response = tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_42")
        message = assistant_tool_message(response)
        assert message["role"] == "assistant"
        assert message["tool_calls"][0]["id"] == "tc_42"
        assert message["tool_calls"][0]["function"]["name"] == "decode_dtc"

    def test_the_assistant_message_serialises_arguments_as_json(self):
        message = assistant_tool_message(tool_turn("decode_dtc", {"code": "P0420"}))
        arguments = message["tool_calls"][0]["function"]["arguments"]
        assert json.loads(arguments) == {"code": "P0420"}

    def test_the_tool_message_answers_the_same_call_id(self):
        response = tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_42")
        message = tool_result_message(response, "catalyst efficiency")
        assert message["role"] == "tool"
        assert message["tool_call_id"] == "tc_42"
        assert message["name"] == "decode_dtc"
        assert message["content"] == "catalyst efficiency"

    def test_none_content_never_reaches_the_wire(self):
        """A tool-only turn has no text; the API expects a string, not null."""
        response = LLMResponse(content="", tool_name="decode_dtc",
                               tool_arguments={}, tool_call_id="tc_1")
        assert assistant_tool_message(response)["content"] == ""


# --- Loop detection helper ---

class TestAlreadyCalled:

    def test_detects_an_identical_call(self):
        trace = [{"tool": "decode_dtc", "arguments": {"code": "P0420"}}]
        assert already_called(trace, "decode_dtc", {"code": "P0420"}) is True

    def test_different_arguments_are_not_a_repeat(self):
        trace = [{"tool": "decode_dtc", "arguments": {"code": "P0420"}}]
        assert already_called(trace, "decode_dtc", {"code": "P0301"}) is False

    def test_different_tool_is_not_a_repeat(self):
        trace = [{"tool": "decode_dtc", "arguments": {"code": "P0420"}}]
        assert already_called(trace, "decode_vin", {"code": "P0420"}) is False

    def test_an_empty_trace_never_matches(self):
        assert already_called([], "decode_dtc", {"code": "P0420"}) is False


# --- Direct answer ---

class TestDirectAnswer:

    def test_a_text_answer_ends_the_run(self):
        backend = FakeBackend([LLMResponse(content="The DPF is clogged.")])
        result = run_agent("fumee noire", backend)
        assert result.answer == "The DPF is clogged."
        assert result.stop_reason == "answered"
        assert result.turns == 1
        assert result.trace == []

    def test_the_answer_is_appended_to_the_history(self):
        backend = FakeBackend([LLMResponse(content="Outside my scope.")])
        result = run_agent("recette du couscous", backend)
        assert result.messages[-1] == {"role": "assistant", "content": "Outside my scope."}

    def test_the_conclusion_survives_the_router(self):
        """
        Regression: LangGraph discards whatever the routing function
        assigns to the state, so the conclusion is written by a dedicated
        node. Moving it back into the router would empty the answer with
        no error at all - this test is the alarm.
        """
        backend = FakeBackend([LLMResponse(content="A real conclusion.")])
        result = run_agent("question", backend)
        assert result.answer != ""
        assert result.stop_reason != ""

    def test_no_tools_means_no_trace(self):
        backend = FakeBackend([LLMResponse(content="done")])
        assert run_agent("q", backend).tools_used == []


# --- One tool, then an answer ---

class TestSingleToolRun:

    def test_the_tool_is_executed_with_the_model_arguments(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}),
            LLMResponse(content="Catalyst below threshold."),
        ])
        result = run_agent("P0420?", backend)
        assert stub_tool == [("decode_dtc", {"code": "P0420"})]
        assert result.tools_used == ["decode_dtc"]
        assert result.answer == "Catalyst below threshold."

    def test_the_history_gets_the_assistant_and_tool_pair(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_9"),
            LLMResponse(content="done"),
        ])
        result = run_agent("P0420?", backend)
        roles = [message["role"] for message in result.messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_the_tool_result_references_the_assistant_call_id(self, stub_tool):
        """The wire protocol the API rejects if broken."""
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_9"),
            LLMResponse(content="done"),
        ])
        result = run_agent("P0420?", backend)
        assistant, tool = result.messages[2], result.messages[3]
        assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"] == "tc_9"

    def test_the_model_sees_the_tool_result_on_the_next_turn(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}),
            LLMResponse(content="done"),
        ])
        run_agent("P0420?", backend)
        second_call = backend.calls[1]
        assert second_call[-1]["role"] == "tool"
        assert second_call[-1]["content"] == "result for decode_dtc"

    def test_the_trace_records_arguments_and_result(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}),
            LLMResponse(content="done"),
        ])
        entry = run_agent("P0420?", backend).trace[0]
        assert entry["tool"] == "decode_dtc"
        assert entry["arguments"] == {"code": "P0420"}
        assert entry["result"] == "result for decode_dtc"
        assert entry["repeated"] is False


# --- Chaining several tools ---

class TestCompositeRun:

    def test_two_different_tools_run_in_order(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_vin", {"vin": "VF1RFB00X12345678"}, call_id="tc_1"),
            tool_turn("search_repair_cases", {"query": "fumee noire"}, call_id="tc_2"),
            LLMResponse(content="Likely the DPF (case-003)."),
        ])
        result = run_agent("VIN + fumee noire", backend)
        assert result.tools_used == ["decode_vin", "search_repair_cases"]
        assert result.turns == 3
        assert result.stop_reason == "answered"

    def test_each_call_id_is_paired_independently(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_vin", {"vin": "X"}, call_id="tc_1"),
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_2"),
            LLMResponse(content="done"),
        ])
        result = run_agent("q", backend)
        pairs = [
            (message["tool_calls"][0]["id"] if message["role"] == "assistant" and message.get("tool_calls")
             else message.get("tool_call_id"))
            for message in result.messages
            if message["role"] in ("assistant", "tool") and
            (message.get("tool_calls") or message.get("tool_call_id"))
        ]
        assert pairs == ["tc_1", "tc_1", "tc_2", "tc_2"]


# --- Loop detection in the graph ---

class TestRepeatedCall:

    def test_an_identical_call_is_not_executed_twice(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_1"),
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_2"),
            LLMResponse(content="done"),
        ])
        result = run_agent("P0420?", backend)
        assert len(stub_tool) == 1
        assert result.trace[1]["repeated"] is True

    def test_the_repeat_is_reported_back_to_the_model(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_1"),
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_2"),
            LLMResponse(content="done"),
        ])
        result = run_agent("P0420?", backend)
        assert "already called" in result.trace[1]["result"]

    def test_a_corrected_call_runs_normally(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}, call_id="tc_1"),
            tool_turn("decode_dtc", {"code": "P0301"}, call_id="tc_2"),
            LLMResponse(content="done"),
        ])
        run_agent("q", backend)
        assert len(stub_tool) == 2


# --- Turn budget ---

class TestTurnLimit:

    def test_a_looping_model_is_stopped(self, stub_tool):
        """FakeBackend repeats its last response forever once exhausted."""
        backend = FakeBackend([tool_turn("search_repair_cases", {"query": "x"})])
        result = run_agent("q", backend, max_turns=3)
        assert result.hit_turn_limit is True
        assert result.turns == 3

    def test_the_limit_answer_is_honest_not_empty(self, stub_tool):
        backend = FakeBackend([tool_turn("search_repair_cases", {"query": "x"})])
        result = run_agent("q", backend, max_turns=2)
        assert result.answer == TURN_LIMIT_ANSWER
        assert result.answer.strip() != ""

    def test_the_budget_is_not_overspent(self, stub_tool):
        backend = FakeBackend([tool_turn("decode_dtc", {"code": "P0420"})])
        run_agent("q", backend, max_turns=3)
        assert backend.call_count == 3

    def test_work_done_before_the_limit_is_kept(self, stub_tool):
        backend = FakeBackend([tool_turn("search_repair_cases", {"query": "x"})])
        result = run_agent("q", backend, max_turns=3)
        assert len(result.trace) >= 1

    def test_an_answer_within_budget_does_not_trip_the_limit(self, stub_tool):
        backend = FakeBackend([
            tool_turn("decode_dtc", {"code": "P0420"}),
            LLMResponse(content="done"),
        ])
        assert run_agent("q", backend, max_turns=5).hit_turn_limit is False

    def test_the_default_budget_is_five(self):
        assert MAX_TURNS == 5


# --- Degraded backend ---

class TestDegradedBackend:

    def test_an_unavailable_model_ends_cleanly(self):
        """The backend degrades an API failure into text; the graph must not loop."""
        backend = FakeBackend([LLMResponse(content="Error: the model is unavailable.")])
        result = run_agent("q", backend)
        assert result.stop_reason == "answered"
        assert "unavailable" in result.answer

    def test_a_tool_error_is_fed_back_instead_of_stopping(self, monkeypatch):
        monkeypatch.setattr(graph_module, "call_tool",
                            lambda name, arguments: "Error: unknown tool 'nope'.")
        backend = FakeBackend([
            tool_turn("nope", {}, call_id="tc_1"),
            LLMResponse(content="I could not use that tool."),
        ])
        result = run_agent("q", backend)
        assert result.stop_reason == "answered"
        assert "Error" in result.trace[0]["result"]


# --- Result object ---

class TestAgentResult:

    def test_tools_used_follows_call_order(self):
        result = AgentResult(answer="x", trace=[
            {"tool": "decode_vin", "arguments": {}, "result": "", "repeated": False},
            {"tool": "decode_dtc", "arguments": {}, "result": "", "repeated": False},
        ])
        assert result.tools_used == ["decode_vin", "decode_dtc"]

    def test_cases_cited_reads_the_tool_results_not_the_answer(self):
        """
        Citations are scored against what the tools actually returned, so a
        case_id invented by the model cannot count as grounded.
        """
        result = AgentResult(
            answer="It is case-999.",
            trace=[{"tool": "search_repair_cases", "arguments": {},
                    "result": "case-003: FAP colmate", "repeated": False}],
        )
        assert result.cases_cited == ["case-003"]

    def test_cases_cited_deduplicates_and_strips_punctuation(self):
        result = AgentResult(answer="", trace=[
            {"tool": "t", "arguments": {}, "result": "(case-003), case-003; case-007.",
             "repeated": False},
        ])
        assert result.cases_cited == ["case-003", "case-007"]

    def test_an_empty_trace_cites_nothing(self):
        assert AgentResult(answer="x").cases_cited == []

    def test_latency_is_measured(self):
        backend = FakeBackend([LLMResponse(content="done")])
        assert run_agent("q", backend).latency_ms > 0