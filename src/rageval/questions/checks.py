"""Rule-based filters for generated questions. No LLM calls, so they cost nothing and are exact."""

from __future__ import annotations

import json
import re

from rageval.text import normalize

# A retrieval question has to stand on its own: it cannot point at a passage the user never saw.
_CONTEXT_EN = re.compile(
    r"\b(?:this|the above|the present|the aforementioned|the said|above-mentioned)\s+"
    r"(?:passage|text|excerpt|document|report|resolution|letter|note|paragraph|section|chapter|annex|decision)\b"
    r"|\b(?:in|from|according to)\s+the\s+(?:passage|text|excerpt)\b"
    r"|\b(?:mentioned|referred to|cited)\s+above\b"
    r"|\bpassage\s+[ab12]\b",
    re.IGNORECASE,
)
_CONTEXT_AR = tuple(
    normalize(phrase, "ar")
    for phrase in (
        "هذا التقرير",
        "هذه الوثيقة",
        "هذا القرار",
        "هذه الرسالة",
        "هذه المذكرة",
        "هذا النص",
        "هذا المقطع",
        "المقتطف",
        "التقرير الحالي",
        "المذكور أعلاه",
        "المذكورة أعلاه",
        "الوارد أعلاه",
        "الواردة أعلاه",
        "المشار إليه أعلاه",
        "المشار إليها أعلاه",
    )
)

# "In the document" on its own points at a passage the user never saw; "in the document that the
# Assembly requested..." describes one, which is how a bridge question must refer to a document.
_DESCRIBED_EN = r"(?:that|which|entitled|on|of|by|concerning|submitted|issued|dated|prepared|in which)"
_CONTEXT_EN_DOCUMENT = re.compile(
    rf"\b(?:in|from|according to|mentioned in|referred to in)\s+the\s+(?:document|report)\b(?!\s+{_DESCRIBED_EN}\b)",
    re.IGNORECASE,
)
_CONTEXT_AR_DOCUMENT = re.compile(
    normalize("في", "ar")
    + r" (?:الوثيقه|النص|المقطع|المقتطف)(?!\s+(?:التي|الذي|المتعلقه|المعنونه|الصادره|المقدمه|المؤرخه|بشان|عن|حول)\b)"
)

# A document symbol or "n/m" resolution number in a bridge question gives BM25 an exact-match
# shortcut to the second document, which defeats the point of a two-step question.
_DOCUMENT_NUMBER = re.compile(r"[a-z]+/[\w.]*\d|\b\d{1,4}\s*/\s*\d{1,4}\b")


def has_context_reference(question: str, lang: str) -> bool:
    if lang == "en":
        return bool(_CONTEXT_EN.search(question) or _CONTEXT_EN_DOCUMENT.search(question))
    text = normalize(question, "ar")
    return any(phrase in text for phrase in _CONTEXT_AR) or bool(_CONTEXT_AR_DOCUMENT.search(text))


def mentions_document_number(question: str, lang: str) -> bool:
    return bool(_DOCUMENT_NUMBER.search(normalize(question, lang)))


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object, tolerating code fences or prose around it."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def is_true(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")
