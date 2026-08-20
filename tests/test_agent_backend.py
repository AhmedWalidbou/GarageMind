"""
Tests for the LLM backend layer - GarageMind M4

None of these tests touch the network or need an API key. The Mistral
path is exercised by injecting a fake client into the backend, which is
exactly what the lazy `_client` attribute is there for.
"""

import json
import pytest

from src.agent.backend import (
    LLMResponse,
    LLMBackend,
    FakeBackend,
    MistralBackend,
    to_mistral_tools,
    parse_mistral_response,
    get_backend,
)
from src.agent.tools import TOOL_SCHEMAS


# --- Test doubles mimicking the shape of a Mistral SDK response ---

class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeCompletion:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeChat:
    """Records the payload it was called with, or raises on demand."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.payloads = []

    def complete(self, **payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, chat):
        self.chat = chat


# --- LLMResponse ---

class TestLLMResponse:

    def test_final_answer_is_not_a_tool_call(self):
        response = LLMResponse(content="the DPF is clogged")
        assert response.is_tool_call is False

    def test_tool_call_is_flagged(self):
        response = LLMResponse(tool_name="decode_dtc", tool_arguments={"code": "P0420"})
        assert response.is_tool_call is True

    def test_arguments_default_to_an_empty_dict_not_shared(self):
        first = LLMResponse()
        second = LLMResponse()
        first.tool_arguments["x"] = 1
        assert second.tool_arguments == {}

    def test_base_backend_is_abstract(self):
        with pytest.raises(NotImplementedError):
            LLMBackend().complete([{"role": "user", "content": "hi"}])


# --- FakeBackend ---

class TestFakeBackend:

    def test_rejects_an_empty_script(self):
        with pytest.raises(ValueError):
            FakeBackend([])

    def test_replays_responses_in_order(self):
        backend = FakeBackend([
            LLMResponse(tool_name="decode_dtc", tool_arguments={"code": "P0420"}),
            LLMResponse(content="final answer"),
        ])
        first = backend.complete([{"role": "user", "content": "P0420?"}])
        second = backend.complete([{"role": "user", "content": "P0420?"}])
        assert first.tool_name == "decode_dtc"
        assert second.content == "final answer"

    def test_repeats_the_last_response_once_exhausted(self):
        backend = FakeBackend([LLMResponse(content="done")])
        backend.complete([])
        extra = backend.complete([])
        assert extra.content == "done"

    def test_records_every_message_list_it_receives(self):
        backend = FakeBackend([LLMResponse(content="ok")])
        backend.complete([{"role": "user", "content": "first"}])
        backend.complete([{"role": "user", "content": "second"}])
        assert backend.call_count == 2
        assert backend.calls[0][0]["content"] == "first"
        assert backend.calls[1][0]["content"] == "second"

    def test_recorded_messages_are_snapshots_not_references(self):
        """A later mutation of the caller's list must not rewrite history."""
        backend = FakeBackend([LLMResponse(content="ok")])
        messages = [{"role": "user", "content": "first"}]
        backend.complete(messages)
        messages.append({"role": "assistant", "content": "later"})
        assert len(backend.calls[0]) == 1

    def test_records_the_tools_it_was_offered(self):
        backend = FakeBackend([LLMResponse(content="ok")])
        backend.complete([], tools=TOOL_SCHEMAS)
        assert backend.tools_seen[0] == TOOL_SCHEMAS


# --- Schema translation ---

class TestToMistralTools:

    def test_translates_every_registry_schema(self):
        functions = to_mistral_tools(TOOL_SCHEMAS)
        assert len(functions) == len(TOOL_SCHEMAS)
        assert {f["function"]["name"] for f in functions} == {s["name"] for s in TOOL_SCHEMAS}

    def test_every_entry_has_the_function_envelope(self):
        for function in to_mistral_tools(TOOL_SCHEMAS):
            assert function["type"] == "function"
            assert set(function["function"]) == {"name", "description", "parameters"}
            assert function["function"]["parameters"]["type"] == "object"

    def test_required_flags_become_a_required_list(self):
        schema = [{
            "name": "demo",
            "description": "demo tool",
            "parameters": {
                "query": {"type": "string", "description": "the query", "required": True},
                "top_k": {"type": "integer", "description": "how many"},
            },
        }]
        parameters = to_mistral_tools(schema)[0]["function"]["parameters"]
        assert parameters["required"] == ["query"]
        assert set(parameters["properties"]) == {"query", "top_k"}

    def test_required_flag_is_stripped_from_the_property(self):
        schema = [{
            "name": "demo",
            "description": "demo tool",
            "parameters": {"query": {"type": "string", "description": "q", "required": True}},
        }]
        query = to_mistral_tools(schema)[0]["function"]["parameters"]["properties"]["query"]
        assert set(query) == {"type", "description"}

    def test_search_repair_cases_marks_query_as_required(self):
        """Grounded on the real registry, not a hand-made schema."""
        functions = {f["function"]["name"]: f for f in to_mistral_tools(TOOL_SCHEMAS)}
        parameters = functions["search_repair_cases"]["function"]["parameters"]
        assert "query" in parameters["required"]

    def test_empty_schema_list_gives_empty_functions(self):
        assert to_mistral_tools([]) == []


# --- Response parsing ---

class TestParseMistralResponse:

    def test_plain_text_becomes_a_final_answer(self):
        completion = FakeCompletion(FakeMessage(content="the DPF is clogged"))
        response = parse_mistral_response(completion)
        assert response.is_tool_call is False
        assert response.content == "the DPF is clogged"

    def test_none_content_becomes_an_empty_string(self):
        response = parse_mistral_response(FakeCompletion(FakeMessage(content=None)))
        assert response.content == ""

    def test_tool_call_arguments_are_parsed_from_json(self):
        completion = FakeCompletion(FakeMessage(
            tool_calls=[FakeToolCall("decode_dtc", json.dumps({"code": "P0420"}))],
        ))
        response = parse_mistral_response(completion)
        assert response.tool_name == "decode_dtc"
        assert response.tool_arguments == {"code": "P0420"}

    def test_arguments_already_decoded_are_accepted(self):
        completion = FakeCompletion(FakeMessage(
            tool_calls=[FakeToolCall("decode_dtc", {"code": "P0420"})],
        ))
        assert parse_mistral_response(completion).tool_arguments == {"code": "P0420"}

    def test_only_the_first_tool_call_is_taken(self):
        """The graph runs one tool per turn by design."""
        completion = FakeCompletion(FakeMessage(tool_calls=[
            FakeToolCall("decode_dtc", '{"code": "P0420"}'),
            FakeToolCall("decode_vin", '{"vin": "VF1RFB00X12345678"}'),
        ]))
        assert parse_mistral_response(completion).tool_name == "decode_dtc"

    def test_malformed_arguments_degrade_into_a_final_answer(self):
        completion = FakeCompletion(FakeMessage(
            tool_calls=[FakeToolCall("decode_dtc", "{not json")],
        ))
        response = parse_mistral_response(completion)
        assert response.is_tool_call is False
        assert "decode_dtc" in response.content

    def test_unreadable_response_degrades_instead_of_raising(self):
        response = parse_mistral_response(object())
        assert response.is_tool_call is False
        assert response.content.startswith("Error")

    def test_empty_choices_degrade_instead_of_raising(self):
        class Empty:
            choices = []
        response = parse_mistral_response(Empty())
        assert response.content.startswith("Error")

    def test_the_raw_response_is_kept_for_inspection(self):
        completion = FakeCompletion(FakeMessage(content="hi"))
        assert parse_mistral_response(completion).raw is completion


# --- MistralBackend with an injected client ---

class TestMistralBackend:

    def test_constructor_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
        assert MistralBackend(api_key="explicit").api_key == "explicit"

    def test_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
        assert MistralBackend().api_key == "from-env"

    def test_a_text_answer_comes_back_as_a_final_answer(self):
        backend = MistralBackend(api_key="test")
        backend._client = FakeClient(FakeChat(FakeCompletion(FakeMessage(content="hello"))))
        response = backend.complete([{"role": "user", "content": "hi"}])
        assert response.content == "hello"
        assert response.is_tool_call is False

    def test_a_tool_call_comes_back_parsed(self):
        backend = MistralBackend(api_key="test")
        completion = FakeCompletion(FakeMessage(
            tool_calls=[FakeToolCall("decode_dtc", '{"code": "P0420"}')],
        ))
        backend._client = FakeClient(FakeChat(completion))
        response = backend.complete([{"role": "user", "content": "P0420?"}])
        assert response.tool_name == "decode_dtc"
        assert response.tool_arguments == {"code": "P0420"}

    def test_tools_are_translated_before_being_sent(self):
        backend = MistralBackend(api_key="test")
        chat = FakeChat(FakeCompletion(FakeMessage(content="ok")))
        backend._client = FakeClient(chat)
        backend.complete([{"role": "user", "content": "hi"}], tools=TOOL_SCHEMAS)
        payload = chat.payloads[0]
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["type"] == "function"

    def test_no_tools_means_no_tool_keys_in_the_payload(self):
        backend = MistralBackend(api_key="test")
        chat = FakeChat(FakeCompletion(FakeMessage(content="ok")))
        backend._client = FakeClient(chat)
        backend.complete([{"role": "user", "content": "hi"}])
        assert "tools" not in chat.payloads[0]
        assert "tool_choice" not in chat.payloads[0]

    def test_generation_settings_are_forwarded(self):
        backend = MistralBackend(api_key="test", model="mistral-tiny", temperature=0.7, max_tokens=42)
        chat = FakeChat(FakeCompletion(FakeMessage(content="ok")))
        backend._client = FakeClient(chat)
        backend.complete([])
        payload = chat.payloads[0]
        assert payload["model"] == "mistral-tiny"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 42

    def test_default_temperature_is_deterministic(self):
        """Reproducible evaluation runs depend on this."""
        assert MistralBackend().temperature == 0.0

    def test_an_api_failure_degrades_into_a_final_answer(self):
        backend = MistralBackend(api_key="test")
        backend._client = FakeClient(FakeChat(error=ConnectionError("no network")))
        response = backend.complete([{"role": "user", "content": "hi"}])
        assert response.is_tool_call is False
        assert "unavailable" in response.content
        assert "ConnectionError" in response.content

    def test_the_client_is_not_built_at_construction_time(self):
        """Importing and constructing must cost nothing and need no key."""
        assert MistralBackend()._client is None


# --- Factory ---

class TestGetBackend:

    def test_returns_a_mistral_backend_by_default(self):
        assert isinstance(get_backend(), MistralBackend)

    def test_forwards_keyword_arguments(self):
        backend = get_backend("mistral", model="mistral-tiny", api_key="test")
        assert backend.model == "mistral-tiny"

    def test_fake_backend_cannot_be_built_by_name(self):
        with pytest.raises(ValueError):
            get_backend("fake")

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("gpt-9")