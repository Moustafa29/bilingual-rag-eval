"""Appropriation records from UN General Assembly mission-financing resolutions.

Financing resolutions follow a fixed template: "Decides to appropriate to the Special Account for the
United Nations Mission in Liberia the amount of 585,682,100 dollars for the period from 1 July 2009 to
30 June 2010, inclusive of ...". Two missions' appropriations for the same budget period are
comparable by construction, which is what a deterministic comparison question needs.

Extraction rules, each fixing an error found in the first probe's sample:
- The paragraph must say "to appropriate". "Decides also to apportion ... representing the balance of
  the appropriation" is an apportionment and is rejected.
- The amount is the first amount after "amount of" in either form: grouped ("585,682,100") or words
  ("563 million"). The first probe only knew the grouped form, so for "the amount of 563 million
  dollars ... inclusive of the amount of 292,069,000 dollars" it took the quoted earlier figure.
- The period is the first "for the period from ... to ..." after the amount and before any "inclusive
  of" / "including" clause, so a period quoted inside that clause is never used.
- Paragraphs appropriating an additional amount or a reduction are labelled and not treated as a
  period's appropriation.
- The mission name comes from "Special Account for the <mission>" in the paragraph when present,
  otherwise from the most frequent full mission name in the document. The first probe used the title
  line, which TEI sentence splitting truncates ("United Nations Disengagement").
"""

from __future__ import annotations

import re
from collections import Counter

from rageval.text import normalize

# A capitalised name token, but not the word that starts the next sentence of a resolution: the title line
# runs straight into "The General Assembly," or a preamble opener, which would otherwise extend the name.
_SENTENCE_OPENERS = (
    "The|Having|Recalling|Noting|Decides|Requests|Reaffirming|Taking|Expressing|Welcoming|Concerned|"
    "Bearing|Mindful|Emphasizing|Stressing|Aware|Acknowledging|Recognizing|Endorses|Approves|Authorizes"
)
_WORD = rf"(?!(?:{_SENTENCE_OPENERS})\b)[A-Z][\w'’\-]*"
_CONNECTOR = r"(?:of|the|in|for|and|on|d'[A-Z]\w*|d’[A-Z]\w*)"
_MISSION = rf"{_WORD}(?:\s+(?:{_WORD}|{_CONNECTOR}))*"
_TRAILING_CONNECTORS = {"of", "the", "in", "for", "and", "on"}

_APPROPRIATE = re.compile(r"\bto appropriate\b")
_AMOUNT = re.compile(
    r"\b(?:the|an) (?:total |additional |gross )?amount of (?:US\s?)?\$?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?\s(?:million|billion))"
)
_PERIOD = re.compile(r"\bperiod from (\d{1,2} [A-Z][a-z]+ \d{4}) to (\d{1,2} [A-Z][a-z]+ \d{4})")
_CLAUSE_END = re.compile(r"\binclusive\b|\bincluding\b|;")
_PARAGRAPH_MISSION = re.compile(rf"Special Account for the ({_MISSION})\s+(?:the|an) (?:total |additional |gross )?amount")
_DOCUMENT_MISSION = re.compile(rf"(?:[Ff]inancing of|Special Account for) the ({_MISSION})")


def parse_amount(text: str) -> int:
    """"585,682,100" -> 585682100; "563 million" -> 563000000; "6.5 million" -> 6500000."""
    text = text.strip()
    scaled = re.fullmatch(r"(\d+(?:\.\d+)?)\s(million|billion)", text)
    if scaled:
        return round(float(scaled.group(1)) * (1_000_000 if scaled.group(2) == "million" else 1_000_000_000))
    return int(float(text.replace(",", "")))


def _trim(name: str) -> str:
    tokens = name.split()
    while tokens and tokens[-1] in _TRAILING_CONNECTORS:
        tokens.pop()
    return " ".join(tokens)


def canonical_mission(name: str) -> str:
    name = re.sub(r"\s*\(continued\)", "", name, flags=re.IGNORECASE)
    name = name.lower().replace("peace-keeping", "peacekeeping").replace("’", "'")
    return re.sub(r"\s+", " ", name).strip()


def document_mission(text: str) -> str | None:
    """Most frequent full mission name (3+ tokens) in the document; ties go to the longer name."""
    names = Counter(_trim(m.group(1)) for m in _DOCUMENT_MISSION.finditer(text))
    names = Counter({n: c for n, c in names.items() if len(n.split()) >= 3})
    if not names:
        return None
    return max(names, key=lambda n: (names[n], len(n)))


def extract_appropriation(paragraph: str) -> dict | None:
    verb = _APPROPRIATE.search(paragraph)
    if not verb:
        return None
    amount = _AMOUNT.search(paragraph, verb.end())
    if not amount:
        return None
    clause_end = _CLAUSE_END.search(paragraph, amount.end())
    period = _PERIOD.search(paragraph, amount.end(), clause_end.start() if clause_end else len(paragraph))
    if not period:
        return None
    main_clause = paragraph[verb.start() : period.end()]
    kind = "additional" if re.search(r"\badditional\b", main_clause) else "reduction" if "reduc" in main_clause else "appropriation"
    mission = _PARAGRAPH_MISSION.search(paragraph, 0, amount.end() + 1)
    return {
        "amount_text": amount.group("num"),
        "amount_usd": parse_amount(amount.group("num")),
        "period": f"{period.group(1)} - {period.group(2)}",
        "kind": kind,
        "mission_in_paragraph": _trim(mission.group(1)) if mission else None,
    }


def arabic_confirms(ar_paragraphs: list[str], amount_text: str) -> bool:
    """The amount appears in an Arabic paragraph about an amount (مبلغ), grouped or as '<n> مليون'."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", amount_text):
        groups = amount_text.split(".")[0].split(",")
        patterns = [re.escape(sep.join(groups)) for sep in (" ", ",", "")]
    else:
        number, scale = amount_text.split()
        word = r"(?:مليون|ملايين)" if scale == "million" else r"(?:بليون|مليار|بلايين)"
        patterns = [re.escape(number).replace(r"\.", "[.,٫]") + rf"\s*(?:من\s*)?{word}"]
    amount_word = normalize("مبلغ", "ar")
    for paragraph in ar_paragraphs:
        text = normalize(paragraph, "ar")
        if amount_word in text and any(re.search(rf"(?<!\d){p}(?!\d)", text) for p in patterns):
            return True
    return False
