"""
LLM backends - GarageMind M4

One interface, three implementations. The graph talks to LLMBackend and
never knows which model is behind it.

Design decisions:
    - A single response shape (LLMResponse): either a tool call or a
      final answer. Each backend translates its provider's raw format
      into that shape, so provider quirks stop at this boundary.
    - LLMResponse carries the provider's tool_call_id. The chat APIs
      require the tool result message to reference the id of the call it
      answers; dropping it here would make the conversation history
      invalid and fail only at the first real inference, with an opaque
      400. Providers that emit no id get a deterministic fallback so the
      graph needs no special case.
    - FakeBackend replays a scripted sequence and records what it was
      sent. It makes the whole agent testable in milliseconds, with no
      API key and no network - which is also what lets anyone clone the
      repository and run the test suite.
    - MistralBackend never raises. A network failure or an API error
      comes back as a final answer carrying the message, exactly like
      the tool layer: the agent must degrade, not crash.
    - Mistral was chosen for consistency with the other portfolio
      projects (AgentForge, TempoRAG), which already run on it.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


@dataclass
class LLMResponse:
    """
    One model turn. Either it asks for a tool, or it answers.

    tool_name/tool_arguments/tool_call_id are set when the model wants to
    act; content holds the final answer otherwise.
    """
    content: str = ""
    tool_name: str | None = None
    tool_arguments: dict = field(default_factory=dict)
    tool_call_id: str | None = None
    raw: Any = None

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None


class LLMBackend:
    """Interface every backend implements."""

    name = "base"

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError


class FakeBackend(LLMBackend):
    """
    Replays a preset sequence of responses, in order.

    Records every messages list it receives, so tests can assert on what
    the graph actually sent to the model. Once the script is exhausted it
    keeps returning the last response rather than raising, so a runaway
    loop shows up as a stuck agent in tests instead of an exception.
    """

    name = "fake"

    def __init__(self, responses: list[LLMResponse]):
        if not responses:
            raise ValueError("FakeBackend needs at least one response")
        self.responses = responses
        self.calls: list[list[dict]] = []
        self.tools_seen: list[list[dict] | None] = []

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    @property
    def call_count(self) -> int:
        return len(self.calls)


class MistralBackend(LLMBackend):
    """
    Mistral API backend with native function calling.

    The client is created lazily so that importing this module costs
    nothing and does not require an API key.
    """

    name = "mistral"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    @property
    def api_key(self) -> str | None:
        """Read the key from the constructor, the environment or .env."""
        if self._api_key:
            return self._api_key
        key = os.environ.get("MISTRAL_API_KEY")
        if key:
            return key
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            return None
        return os.environ.get("MISTRAL_API_KEY")

    @property
    def client(self):
        if self._client is None:
            from mistralai import Mistral
            key = self.api_key
            if not key:
                raise RuntimeError(
                    "MISTRAL_API_KEY is not set. Put it in a .env file at the "
                    "project root or export it in the environment."
                )
            self._client = Mistral(api_key=key)
        return self._client

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = to_mistral_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            response = self.client.chat.complete(**payload)
        except Exception as exc:
            return LLMResponse(
                content=f"Error: the language model is unavailable ({type(exc).__name__}).",
                raw=exc,
            )

        return parse_mistral_response(response)


# --- Translation helpers, kept module-level so they can be tested alone ---

def to_mistral_tools(schemas: list[dict]) -> list[dict]:
    """
    Convert the tool registry schemas into Mistral's function format.

    The registry uses a compact shape (name, description, parameters with
    a required flag per parameter); Mistral expects JSON Schema with a
    separate required list.
    """
    functions = []
    for schema in schemas:
        properties = {}
        required = []
        for param, spec in schema["parameters"].items():
            properties[param] = {
                "type": spec["type"],
                "description": spec["description"],
            }
            if spec.get("required"):
                required.append(param)
        functions.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return functions


def parse_mistral_response(response: Any) -> LLMResponse:
    """
    Turn a Mistral chat completion into an LLMResponse.

    Malformed tool arguments are treated as a final answer describing the
    problem rather than an exception: the agent can then report it.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return LLMResponse(content="Error: unreadable model response.", raw=response)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        call = tool_calls[0]
        name = call.function.name
        # The tool result must reference this id. Fall back to a stable
        # synthetic id when the provider does not supply one.
        call_id = getattr(call, "id", None) or f"call_{name}"
        raw_args = call.function.arguments
        if isinstance(raw_args, dict):
            arguments = raw_args
        else:
            try:
                arguments = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                return LLMResponse(
                    content=f"Error: the model sent malformed arguments for '{name}'.",
                    raw=response,
                )
        return LLMResponse(
            content=message.content or "",
            tool_name=name,
            tool_arguments=arguments,
            tool_call_id=call_id,
            raw=response,
        )

    return LLMResponse(content=message.content or "", raw=response)


def get_backend(name: str = "mistral", **kwargs) -> LLMBackend:
    """Build a backend by name. Used by the CLI and the evaluation script."""
    if name == "mistral":
        return MistralBackend(**kwargs)
    if name == "fake":
        raise ValueError("FakeBackend must be constructed directly with responses")
    raise ValueError(f"Unknown backend '{name}'. Available: mistral.")