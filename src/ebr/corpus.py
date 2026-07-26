"""
Corpus loader - GarageMind M3 EBR-RAG
Loads and validates the curated repair-case knowledge base, then flattens
each case into language-specific documents ready for embedding.

Design decisions:
    - Validation at the boundary: malformed cases are rejected at load time
      with precise error messages (case id + field), never silently indexed.
      A RAG system's answer quality is capped by its corpus quality.
    - One document per case per language (fr, en). Bilingual retrieval works
      best when each document is monolingual: mixed-language documents pull
      embeddings toward the middle of both languages.
    - Each document carries filterable metadata (dtc_codes, brands, system,
      engine_family) so retrieval can be constrained (e.g. same brand first)
      without relying on the embedding to encode it.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DTC_PATTERN = re.compile(r"^[PCBU][0-9A-F]{4}$", re.IGNORECASE)

REQUIRED_CASE_FIELDS = [
    "id", "dtc_codes", "brands", "engine_family", "system",
    "symptoms_fr", "symptoms_en",
    "root_cause_fr", "root_cause_en",
    "fix_fr", "fix_en",
    "confirmed",
]

LANGS = ("fr", "en")


@dataclass
class RepairDocument:
    """One indexable, monolingual document derived from a repair case."""
    doc_id: str
    case_id: str
    lang: str
    text: str
    dtc_codes: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    system: str = ""
    engine_family: str = ""


class CorpusError(ValueError):
    """Raised when the knowledge base fails validation."""


def _validate_case(case: dict, seen_ids: set[str]) -> None:
    """Validate one case dict; raise CorpusError with a precise message."""
    case_id = case.get("id", "<missing id>")

    for key in REQUIRED_CASE_FIELDS:
        if key not in case:
            raise CorpusError(f"{case_id}: missing required field '{key}'")

    if case["id"] in seen_ids:
        raise CorpusError(f"{case_id}: duplicate case id")

    if not isinstance(case["dtc_codes"], list) or not case["dtc_codes"]:
        raise CorpusError(f"{case_id}: dtc_codes must be a non-empty list")
    for code in case["dtc_codes"]:
        if not DTC_PATTERN.match(code):
            raise CorpusError(f"{case_id}: invalid DTC code format '{code}'")

    if not isinstance(case["brands"], list) or not case["brands"]:
        raise CorpusError(f"{case_id}: brands must be a non-empty list")

    for lang in LANGS:
        for prefix in ("symptoms", "root_cause", "fix"):
            key = f"{prefix}_{lang}"
            if not isinstance(case[key], str) or not case[key].strip():
                raise CorpusError(f"{case_id}: field '{key}' is empty")

    if not isinstance(case["confirmed"], bool):
        raise CorpusError(f"{case_id}: 'confirmed' must be a boolean")


def load_cases(path: str | Path) -> list[dict]:
    """
    Load and validate the knowledge base JSON.

    Returns the list of validated case dicts.
    Raises CorpusError on any structural or content problem.
    """
    path = Path(path)
    if not path.exists():
        raise CorpusError(f"knowledge base not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "cases" not in data or not isinstance(data["cases"], list):
        raise CorpusError("knowledge base must contain a 'cases' list")
    if not data["cases"]:
        raise CorpusError("knowledge base contains no cases")

    seen_ids: set[str] = set()
    for case in data["cases"]:
        _validate_case(case, seen_ids)
        seen_ids.add(case["id"])

    return data["cases"]


def case_to_documents(case: dict) -> list[RepairDocument]:
    """
    Flatten one validated case into monolingual documents (fr, en).

    The embedded text concatenates symptoms, root cause and fix with
    lightweight section markers: dense enough for retrieval, structured
    enough for display.
    """
    docs = []
    dtc_str = ", ".join(code.upper() for code in case["dtc_codes"])
    for lang in LANGS:
        if lang == "fr":
            text = (
                f"Codes défaut: {dtc_str}. "
                f"Moteur: {case['engine_family']}. "
                f"Symptômes: {case['symptoms_fr']} "
                f"Cause racine: {case['root_cause_fr']} "
                f"Réparation: {case['fix_fr']}"
            )
        else:
            text = (
                f"Fault codes: {dtc_str}. "
                f"Engine: {case['engine_family']}. "
                f"Symptoms: {case['symptoms_en']} "
                f"Root cause: {case['root_cause_en']} "
                f"Fix: {case['fix_en']}"
            )
        docs.append(
            RepairDocument(
                doc_id=f"{case['id']}-{lang}",
                case_id=case["id"],
                lang=lang,
                text=text,
                dtc_codes=[c.upper() for c in case["dtc_codes"]],
                brands=list(case["brands"]),
                system=case["system"],
                engine_family=case["engine_family"],
            )
        )
    return docs


def load_documents(path: str | Path) -> list[RepairDocument]:
    """Full pipeline: JSON knowledge base -> validated indexable documents."""
    cases = load_cases(path)
    docs: list[RepairDocument] = []
    for case in cases:
        docs.extend(case_to_documents(case))
    return docs


if __name__ == "__main__":
    PATH = "knowledge/repair_cases.json"
    documents = load_documents(PATH)
    langs = {d.lang for d in documents}
    systems = sorted({d.system for d in documents})
    print(f"Loaded {len(documents)} documents "
          f"from {len(documents) // len(LANGS)} cases")
    print(f"Languages: {sorted(langs)}")
    print(f"Systems covered: {systems}")
    print(f"\nSample document ({documents[0].doc_id}):")
    print(f"  {documents[0].text[:200]}...")