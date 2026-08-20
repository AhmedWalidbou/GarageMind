"""
Agent tools - GarageMind M4

Wraps the Module 1 and Module 3 capabilities as functions an LLM can
call, each with a JSON schema describing its parameters.

Design decisions:
    - No tool ever raises. Bad input, a missing file or an unknown code
      returns a readable message instead of an exception. An agent that
      crashes on malformed input is unusable; an agent that reads
      "invalid DTC code" can correct itself on the next turn.
    - Every tool returns plain text, not Python objects: the model
      consumes text, so formatting belongs here rather than in the
      graph.
    - The retriever is loaded lazily and cached at module level. It
      pulls the e5 model and the Qdrant index, which cost seconds to
      open and must not be paid on every call.
    - decode_vin runs with use_api=False. An agent that depends on an
      external HTTP call is neither deterministic nor testable offline.
    - No anomaly-detection tool. Module 2 exposes training primitives
      that need a full fit-and-calibrate pipeline, far too slow for a
      tool call; the CAN identifier distribution returned by
      analyze_can_log already surfaces flood attacks, which was the
      Module 1 finding.
"""

import atexit
from pathlib import Path
from typing import Callable

from src.ebr.corpus import load_documents
from src.ebr.hybrid import HybridRetriever
from src.ebr.lexical import LexicalRetriever
from src.ebr.retriever import Retriever
from src.scan_engine.can_parser import bus_summary, load_can_log
from src.scan_engine.dtc_reader import interpret_dtc, validate_dtc
from src.scan_engine.vin_decoder import decode_vin as _decode_vin
from src.scan_engine.vin_decoder import validate_vin

KNOWLEDGE_PATH = Path("knowledge/repair_cases.json")
INDEX_PATH = Path("data/qdrant")
CAN_LOG_SAMPLE_ROWS = 50000

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Build the hybrid retriever once and reuse it across calls."""
    global _retriever
    if _retriever is None:
        from src.ebr.embedder import Embedder
        from src.ebr.vectorstore import VectorStore

        store = VectorStore(location=str(INDEX_PATH))
        documents = load_documents(KNOWLEDGE_PATH)
        _retriever = HybridRetriever(
            dense=Retriever(Embedder(), store),
            lexical=LexicalRetriever(documents),
        )
    return _retriever


def close_retriever() -> None:
    """
    Release the Qdrant client held by the cached retriever.

    Without this, the client is only collected when the interpreter is
    already tearing down: its __del__ then runs after sys.meta_path is
    gone and prints an ignored ImportError. Harmless, but it looks like
    a crash to anyone running the CLI. Registered with atexit so the
    close happens while imports still work.
    """
    global _retriever
    if _retriever is None:
        return
    store = getattr(getattr(_retriever, "dense", None), "store", None)
    for candidate in (store, getattr(store, "client", None)):
        closer = getattr(candidate, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass  # Shutdown must never raise.
            break
    _retriever = None


def reset_retriever() -> None:
    """Drop the cached retriever, closing it first. For tests only."""
    close_retriever()


atexit.register(close_retriever)


# --- Tools ---

def search_repair_cases(query: str, top_k: int = 3) -> str:
    """Search the curated repair-case knowledge base."""
    if not query or not query.strip():
        return "Error: the query is empty. Provide a description of the symptoms."
    try:
        top_k = max(1, min(int(top_k), 5))
    except (TypeError, ValueError):
        top_k = 3

    try:
        cases = _get_retriever().retrieve(query, top_k=top_k)
    except Exception as exc:
        return f"Error: repair-case search unavailable ({type(exc).__name__})."

    if not cases:
        return "No repair case matched this description."

    lines = [f"{len(cases)} matching repair case(s):"]
    for rank, case in enumerate(cases, start=1):
        lines.append(
            f"{rank}. [{case.case_id}] system={case.system} "
            f"engine={case.engine_family} dtc={','.join(case.dtc_codes)}"
        )
        lines.append(f"   {case.text}")
    return "\n".join(lines)


def decode_dtc(code: str) -> str:
    """Interpret one OBD-II diagnostic trouble code."""
    if not code or not code.strip():
        return "Error: no DTC code provided."

    code = code.strip().upper()
    if not validate_dtc(code):
        return (
            f"Error: '{code}' is not a valid DTC code. "
            "Expected one letter (P, C, B or U) followed by four hex digits, "
            "for example P0301."
        )

    try:
        info = interpret_dtc(code, lang="fr")
    except Exception as exc:
        return f"Error: could not interpret '{code}' ({type(exc).__name__})."

    parts = [f"DTC {code}"]
    for key in ("system", "description", "severity", "type", "subsystem"):
        value = info.get(key)
        if value:
            parts.append(f"{key}: {value}")
    causes = info.get("causes") or info.get("likely_causes")
    if causes:
        joined = ", ".join(causes) if isinstance(causes, list) else str(causes)
        parts.append(f"likely causes: {joined}")
    if not info.get("description"):
        parts.append(
            "note: this code is not in the local database, only its "
            "structure could be decoded."
        )
    return " | ".join(parts)


def decode_vin(vin: str) -> str:
    """Identify a vehicle from its 17-character VIN, offline only."""
    if not vin or not vin.strip():
        return "Error: no VIN provided."

    vin = vin.strip().upper()
    if not validate_vin(vin):
        return f"Error: '{vin}' is not a valid 17-character VIN."

    try:
        info = _decode_vin(vin, use_api=False)
    except Exception as exc:
        return f"Error: could not decode VIN ({type(exc).__name__})."

    parts = [f"VIN {vin}"]
    for key in ("manufacturer", "country", "year", "wmi"):
        value = info.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def analyze_can_log(path: str) -> str:
    """Summarise a raw CAN log: rate, identifiers and their distribution."""
    if not path or not path.strip():
        return "Error: no log path provided."

    log_path = Path(path.strip())
    if not log_path.exists():
        return f"Error: file not found at '{log_path}'."

    try:
        df = load_can_log(str(log_path), nrows=CAN_LOG_SAMPLE_ROWS)
        summary = bus_summary(df)
    except Exception as exc:
        return f"Error: could not read the CAN log ({type(exc).__name__})."

    top = summary.get("top_10_can_ids", {})
    total = summary.get("total_frames", 0) or 1
    top_lines = [
        f"   {can_id}: {count} frames ({100 * count / total:.1f}%)"
        for can_id, count in list(top.items())[:5]
    ]

    return "\n".join([
        f"CAN log summary for {log_path.name} "
        f"(first {CAN_LOG_SAMPLE_ROWS} frames at most):",
        f"total frames: {summary.get('total_frames')}",
        f"duration: {summary.get('duration_seconds')} s "
        f"({summary.get('frames_per_second')} frames/s)",
        f"unique CAN identifiers: {summary.get('unique_can_ids')}",
        "most frequent identifiers:",
        *top_lines,
        "note: an identifier taking a very large share of the traffic "
        "suggests an injection flood.",
    ])


# --- Registry ---

TOOL_SCHEMAS = [
    {
        "name": "search_repair_cases",
        "description": (
            "Search a curated knowledge base of documented repair cases "
            "from real workshops. Use this to find how similar symptoms "
            "were diagnosed and fixed. Works in French and English."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Symptoms, in the customer's or mechanic's words.",
                "required": True,
            },
            "top_k": {
                "type": "integer",
                "description": "How many cases to return, 1 to 5. Default 3.",
                "required": False,
            },
        },
    },
    {
        "name": "decode_dtc",
        "description": (
            "Interpret an OBD-II diagnostic trouble code such as P0301: "
            "affected system, meaning, severity and likely causes."
        ),
        "parameters": {
            "code": {
                "type": "string",
                "description": "The DTC code, for example P0301 or U0100.",
                "required": True,
            },
        },
    },
    {
        "name": "decode_vin",
        "description": (
            "Identify a vehicle from its 17-character VIN: manufacturer, "
            "country of origin and model year. Offline, local tables only."
        ),
        "parameters": {
            "vin": {
                "type": "string",
                "description": "The 17-character vehicle identification number.",
                "required": True,
            },
        },
    },
    {
        "name": "analyze_can_log",
        "description": (
            "Summarise a raw CAN bus log file: frame rate, number of "
            "identifiers and how the traffic is distributed across them. "
            "Use this when the user points at a capture file."
        ),
        "parameters": {
            "path": {
                "type": "string",
                "description": "Path to the CAN log file.",
                "required": True,
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "search_repair_cases": search_repair_cases,
    "decode_dtc": decode_dtc,
    "decode_vin": decode_vin,
    "analyze_can_log": analyze_can_log,
}


def call_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call by name. Unknown tools and bad arguments come
    back as readable messages, never as exceptions, so the agent can
    recover on its next turn.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        available = ", ".join(sorted(TOOL_REGISTRY))
        return f"Error: unknown tool '{name}'. Available tools: {available}."
    if not isinstance(arguments, dict):
        return f"Error: arguments for '{name}' must be an object."

    try:
        return func(**arguments)
    except TypeError as exc:
        return f"Error: wrong arguments for '{name}' ({exc})."
    except Exception as exc:
        return f"Error: '{name}' failed ({type(exc).__name__})."