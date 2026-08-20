"""
ReAct diagnostic agent - GarageMind M4

A LangGraph state machine that alternates between reasoning (calling the
model) and acting (running a tool), until the model answers or the turn
budget runs out.

    START -> reason -> [route] -> act -> reason -> ...
                          |
                          +-> finish -> END

Design decisions:
    - Only StateGraph, nodes and the conditional edge are taken from
      LangGraph. The LangChain abstractions (ChatMistralAI, ToolNode) are
      deliberately not used: the backend and the tool layer already exist
      with their own tested contracts, and going through LangChain would
      hide them behind a second, untested translation.
    - The state carries the full message history (standard ReAct). That
      history *is* the evidence of how the agent reasoned, and it feeds
      the evaluation script directly. A summarised state would have
      invented a private protocol and destroyed traceability.
    - Nodes write state, the router only reads it. LangGraph calls the
      routing function to pick an edge and discards anything it assigns,
      so the conclusion is written by a dedicated `finish` node. Doing it
      in the router would have failed silently, with an empty answer and
      no error - which is why it is pinned by a test here.
    - Every key that must survive a step is declared in AgentState;
      undeclared keys are dropped between supersteps.
    - The turn budget is checked *before* looping back to the model, so
      an exhausted agent never pays for one last completion.
    - A tool error is fed back as a normal tool result. The agent must be
      able to correct itself (retry with a fixed argument) rather than
      stop on the first bad call.
    - Repeated identical calls are detected and answered with a notice
      instead of the same result. Without it a looping model burns the
      whole budget re-reading what it already has.
    - No checkpointer: this agent answers one question per run, there is
      no multi-turn conversation to persist.
"""

import json
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.backend import LLMBackend, LLMResponse
from src.agent.tools import TOOL_SCHEMAS, call_tool

MAX_TURNS = 5

TURN_LIMIT_ANSWER = (
    "I reached my analysis limit before reaching a conclusion. Please narrow "
    "the question, or give me the fault code and the vehicle."
)

SYSTEM_PROMPT = """You are GarageMind, a diagnostic assistant for car mechanics.

You help diagnose vehicle faults using the tools available to you. You do
not guess and you do not invent facts about a vehicle.

Rules you must follow:

1. Ground every diagnostic claim in a tool result. When you use a repair
   case, cite its identifier explicitly, like (case-007). Never cite an
   identifier that a tool did not return to you.
2. If the question is not about vehicle diagnostics, say briefly that it
   is outside your scope. Do not answer it and do not call any tool.
3. If the question lacks the information needed to diagnose anything -
   no symptom, no fault code, no vehicle - ask for the specific missing
   detail instead of searching. One short question, no speculation.
4. If a fault code is unknown to the tools, say plainly that you do not
   have a definition for it. Do not infer its meaning from its number.
5. When a tool returns an error, read it and correct your call if you
   can. Do not repeat an identical call.
6. Answer in the language the mechanic used.

Be concise and practical: the likely cause, what it is based on, and the
checks to run next."""


class AgentState(TypedDict):
    """
    What flows through the graph.

    messages     full chat history, in the provider's wire format
    trace        one entry per executed tool, for evaluation
    turns        model calls made so far
    pending      the model turn awaiting routing
    answer       final text, written by the finish node
    stop_reason  "answered" or "turn_limit"
    """
    messages: list[dict]
    trace: list[dict]
    turns: int
    pending: LLMResponse | None
    answer: str
    stop_reason: str


@dataclass
class AgentResult:
    """The outcome of one run, in a shape the CLI and the evaluator can use."""
    answer: str
    trace: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    turns: int = 0
    stop_reason: str = ""
    latency_ms: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        """Tool names in call order - what the evaluation scores."""
        return [entry["tool"] for entry in self.trace]

    @property
    def cases_cited(self) -> list[str]:
        """Case identifiers returned by the tools during this run."""
        seen = []
        for entry in self.trace:
            for token in entry["result"].split():
                cleaned = token.strip("(),.:;[]")
                if cleaned.startswith("case-") and cleaned not in seen:
                    seen.append(cleaned)
        return seen

    @property
    def hit_turn_limit(self) -> bool:
        return self.stop_reason == "turn_limit"


def build_initial_state(question: str) -> AgentState:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "trace": [],
        "turns": 0,
        "pending": None,
        "answer": "",
        "stop_reason": "",
    }


def assistant_tool_message(response: LLMResponse) -> dict:
    """
    Rebuild the assistant turn that requested a tool.

    The API requires the assistant message that issued the call to be
    present in the history, carrying the same id the tool result
    references.
    """
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [{
            "id": response.tool_call_id,
            "type": "function",
            "function": {
                "name": response.tool_name,
                "arguments": json.dumps(response.tool_arguments),
            },
        }],
    }


def tool_result_message(response: LLMResponse, result: str) -> dict:
    return {
        "role": "tool",
        "name": response.tool_name,
        "tool_call_id": response.tool_call_id,
        "content": result,
    }


def already_called(trace: list[dict], name: str, arguments: dict) -> bool:
    """True if this exact call was already executed in this run."""
    return any(
        entry["tool"] == name and entry["arguments"] == arguments
        for entry in trace
    )


def build_graph(backend: LLMBackend, max_turns: int = MAX_TURNS):
    """
    Compile the agent graph for a given backend.

    The backend is captured by the nodes rather than stored in the state,
    so the state stays serialisable and comparable in tests.
    """

    def reason(state: AgentState) -> dict:
        """Ask the model what to do next."""
        response = backend.complete(state["messages"], tools=TOOL_SCHEMAS)
        return {"turns": state["turns"] + 1, "pending": response}

    def act(state: AgentState) -> dict:
        """Run the requested tool and feed the result back into the history."""
        response = state["pending"]
        name = response.tool_name
        arguments = response.tool_arguments

        repeated = already_called(state["trace"], name, arguments)
        if repeated:
            result = (
                f"Notice: '{name}' was already called with these exact "
                "arguments in this session. Use the previous result, or "
                "change the arguments."
            )
        else:
            result = call_tool(name, arguments)

        return {
            "messages": state["messages"] + [
                assistant_tool_message(response),
                tool_result_message(response, result),
            ],
            "trace": state["trace"] + [{
                "tool": name,
                "arguments": arguments,
                "result": result,
                "repeated": repeated,
            }],
        }

    def finish(state: AgentState) -> dict:
        """Write the conclusion. The router cannot: its writes are discarded."""
        response = state["pending"]

        if response is not None and not response.is_tool_call:
            return {
                "answer": response.content,
                "stop_reason": "answered",
                "messages": state["messages"] + [
                    {"role": "assistant", "content": response.content}
                ],
            }

        return {"answer": TURN_LIMIT_ANSWER, "stop_reason": "turn_limit"}

    def route(state: AgentState) -> str:
        """Read-only decision: run a tool, or conclude."""
        response = state["pending"]
        if response is None or not response.is_tool_call:
            return "finish"
        if state["turns"] >= max_turns:
            return "finish"
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("finish", finish)
    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", route, {"act": "act", "finish": "finish"})
    graph.add_edge("act", "reason")
    graph.add_edge("finish", END)
    return graph.compile()


def run_agent(question: str, backend: LLMBackend, max_turns: int = MAX_TURNS) -> AgentResult:
    """Run one diagnostic question end to end."""
    graph = build_graph(backend, max_turns=max_turns)
    state = build_initial_state(question)

    started = time.perf_counter()
    final = graph.invoke(state, config={"recursion_limit": max_turns * 2 + 10})
    latency_ms = (time.perf_counter() - started) * 1000

    return AgentResult(
        answer=final["answer"],
        trace=final["trace"],
        messages=final["messages"],
        turns=final["turns"],
        stop_reason=final["stop_reason"],
        latency_ms=latency_ms,
    )