"""
Agent tool tests - GarageMind M4

Most of these exercise error paths, because the module's central promise
is that no tool ever raises: bad input must come back as a readable
message the agent can act on.

The repair-case search needs the e5 model and the Qdrant index, so it is
isolated and skipped when the index has not been built - a fresh clone
must not fail the suite for a missing artifact.
"""

from pathlib import Path

from src.agent import tools

import pytest

from src.agent.tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    analyze_can_log,
    call_tool,
    decode_dtc,
    decode_vin,
    search_repair_cases,
)

INDEX_PATH = Path("data/qdrant")
index_missing = not INDEX_PATH.exists()


class TestRegistry:
    def test_registry_and_schemas_agree(self):
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        assert schema_names == set(TOOL_REGISTRY)

    def test_every_schema_is_complete(self):
        for schema in TOOL_SCHEMAS:
            assert schema["name"]
            assert len(schema["description"]) > 40, (
                f"{schema['name']}: description too thin for a model to choose on"
            )
            assert isinstance(schema["parameters"], dict)

    def test_every_parameter_is_typed_and_described(self):
        for schema in TOOL_SCHEMAS:
            for param, spec in schema["parameters"].items():
                assert "type" in spec, f"{schema['name']}.{param}: no type"
                assert "description" in spec, f"{schema['name']}.{param}: no description"
                assert "required" in spec, f"{schema['name']}.{param}: no required flag"

    def test_registry_entries_are_callable(self):
        for name, func in TOOL_REGISTRY.items():
            assert callable(func), f"{name} is not callable"


class TestDecodeDtc:
    def test_valid_code_is_interpreted(self):
        out = decode_dtc("P0301")
        assert "P0301" in out
        assert not out.startswith("Error")

    def test_lowercase_is_accepted(self):
        assert "P0301" in decode_dtc("p0301")

    def test_surrounding_spaces_are_trimmed(self):
        assert "P0301" in decode_dtc("  P0301  ")

    def test_malformed_code_returns_message(self):
        out = decode_dtc("XYZ")
        assert out.startswith("Error")
        assert "P0301" in out, "the error should show a valid example"

    def test_empty_code_returns_message(self):
        assert decode_dtc("").startswith("Error")

    def test_body_code_is_recognised(self):
        """B-codes are body-system codes; the tool must not reject them."""
        out = decode_dtc("B1234")
        assert not out.startswith("Error")


class TestDecodeVin:
    def test_valid_vin_is_decoded(self):
        out = decode_vin("WAUZZZ8V1JA123456")
        assert not out.startswith("Error")
        assert "WAUZZZ8V1JA123456" in out

    def test_too_short_vin_returns_message(self):
        assert decode_vin("ABC123").startswith("Error")

    def test_empty_vin_returns_message(self):
        assert decode_vin("").startswith("Error")


class TestAnalyzeCanLog:
    def test_missing_file_returns_message(self):
        out = analyze_can_log("data/raw/does_not_exist.csv")
        assert out.startswith("Error")
        assert "not found" in out

    def test_empty_path_returns_message(self):
        assert analyze_can_log("").startswith("Error")

    def test_unreadable_file_returns_message(self, tmp_path):
        """A file that exists but is not a CAN log must not raise."""
        bogus = tmp_path / "notalog.csv"
        bogus.write_text("this is not a CAN capture", encoding="utf-8")
        out = analyze_can_log(str(bogus))
        assert out.startswith("Error")


class TestSearchValidation:
    """Input validation runs before the retriever is ever built."""

    def test_empty_query_returns_message(self):
        out = search_repair_cases("")
        assert out.startswith("Error")

    def test_blank_query_returns_message(self):
        assert search_repair_cases("   ").startswith("Error")


class TestDispatch:
    def test_unknown_tool_lists_available_ones(self):
        out = call_tool("teleport", {})
        assert out.startswith("Error")
        assert "decode_dtc" in out

    def test_non_dict_arguments_return_message(self):
        assert call_tool("decode_dtc", "P0301").startswith("Error")

    def test_wrong_argument_name_returns_message(self):
        out = call_tool("decode_dtc", {"wrong_name": "P0301"})
        assert out.startswith("Error")

    def test_valid_dispatch_reaches_the_tool(self):
        assert "P0301" in call_tool("decode_dtc", {"code": "P0301"})

    def test_dispatch_never_raises_on_garbage(self):
        """The contract in one test: whatever comes in, a string comes out."""
        for name, args in [
            ("decode_dtc", {"code": None}),
            ("decode_vin", {"vin": 12345}),
            ("analyze_can_log", {"path": None}),
            ("search_repair_cases", {"query": None}),
            ("nonexistent", {"a": 1}),
        ]:
            assert isinstance(call_tool(name, args), str)


@pytest.mark.skipif(index_missing, reason="Qdrant index not built")
class TestSearchIntegration:
    def test_known_symptoms_return_the_expected_case(self):
        out = search_repair_cases(
            "voyant moteur allume, perte de puissance, regenerations FAP frequentes",
            top_k=3,
        )
        assert not out.startswith("Error")
        assert "case-001" in out

    def test_top_k_is_respected(self):
        out = search_repair_cases("turbo qui siffle", top_k=1)
        assert out.count("[case-") == 1

    def test_top_k_is_clamped(self):
        out = search_repair_cases("turbo qui siffle", top_k=99)
        assert out.count("[case-") <= 5

    def test_english_query_works(self):
        out = search_repair_cases("misfire on cylinder one when warm", top_k=2)
        assert not out.startswith("Error")


class TestRetrieverShutdown:
    """
    close_retriever exists to stop the Qdrant client being collected
    during interpreter teardown, where its __del__ prints an ignored
    ImportError. The tests below pin the contract; the absence of the
    warning itself was verified by running the CLI, since pytest never
    tears the interpreter down mid-suite.
    """

    def test_closing_an_empty_cache_is_a_no_op(self):
        tools.reset_retriever()
        tools.close_retriever()  # must not raise

    def test_the_cached_retriever_is_dropped(self):
        tools._retriever = object()
        tools.close_retriever()
        assert tools._retriever is None

    def test_the_store_is_closed_when_it_can_be(self):
        closed = []

        class FakeStore:
            def close(self):
                closed.append(True)

        class FakeDense:
            store = FakeStore()

        class FakeRetriever:
            dense = FakeDense()

        tools._retriever = FakeRetriever()
        tools.close_retriever()
        assert closed == [True]

    def test_a_failing_close_never_propagates(self):
        """Shutdown must stay silent even when the client misbehaves."""

        class ExplodingStore:
            def close(self):
                raise RuntimeError("client already gone")

        class FakeDense:
            store = ExplodingStore()

        class FakeRetriever:
            dense = FakeDense()

        tools._retriever = FakeRetriever()
        tools.close_retriever()  # must not raise
        assert tools._retriever is None

    def test_a_retriever_without_a_closable_store_is_still_dropped(self):
        class FakeRetriever:
            dense = None

        tools._retriever = FakeRetriever()
        tools.close_retriever()
        assert tools._retriever is None

    def test_reset_retriever_closes_as_well(self):
        closed = []

        class FakeStore:
            def close(self):
                closed.append(True)

        class FakeDense:
            store = FakeStore()

        class FakeRetriever:
            dense = FakeDense()

        tools._retriever = FakeRetriever()
        tools.reset_retriever()
        assert closed == [True]
        assert tools._retriever is None

    def test_the_close_is_registered_for_interpreter_exit(self):
        """Without the atexit hook the fix only works when called by hand."""
        # atexit exposes no portable way to inspect what is registered,
        # so the registration is asserted on the source itself.
        source = Path("src/agent/tools.py").read_text(encoding="utf-8")
        assert "atexit.register(close_retriever)" in source
