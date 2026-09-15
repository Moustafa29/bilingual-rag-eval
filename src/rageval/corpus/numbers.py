"""Digit-group reversal in the Arabic side of the UN corpus, and its alignment-guided correction.

In the aligned pilot corpus, most numbers that English writes with thousands separators appear in
the Arabic chunk with their digit groups in reverse order: English "50,000" is stored as
"000 50", "278,707" as "707 278". The cause is not verified; a right-to-left display order fixed
into the text during legacy document conversion is one candidate.

Left in place, this is an Arabic-only defect that would look like a language effect: an Arabic
query containing "50,000" cannot match "000 50" lexically, embedding models read the digits out of
order, and numeric answers fail exact-match scoring.

The correction is alignment-guided. A reversed digit sequence in an Arabic chunk is rewritten only
when the aligned English chunk contains the same number in normal order, so no Arabic number is
changed on the strength of the Arabic text alone. Digit script (Western or Arabic-Indic) and the
separator characters are preserved; only the order of the groups changes.
"""

from __future__ import annotations

import re
from collections import Counter

from rageval.text import normalize

# An English number with thousands separators: 2,000 or 1,974,200, not part of a longer digit
# token. A comma before or after the number only disqualifies it when a digit is on its other
# side: "$1,974,200, and" is a number followed by a clause comma. The first version rejected any
# adjacent comma, which silently skipped numbers at the end of clauses and list items.
EN_THOUSANDS = re.compile(r"(?<!\d)(?<!\d[.,])\d{1,3}(?:,\d{3})+(?!\d|,\d)")

_DIGIT = "[0-9٠-٩۰-۹]"
_SEP = "[   -  ]"


def english_numbers(en: str) -> list[list[str]]:
    return [m.group(0).split(",") for m in EN_THOUSANDS.finditer(en)]


def has_thousands_number(text: str) -> bool:
    return bool(EN_THOUSANDS.search(text))


# A grouped number in a generated or translated *answer*: groups separated by a comma or by a space-like
# character. English corpus text writes 21,456, but a translation of an Arabic-source answer keeps the
# Arabic grouping, "21 456". The first numeric run rejected all 42 such answers because only commas were
# accepted. A plain year such as 1995 is not grouped and does not match. EN_THOUSANDS, which drives the
# corpus audit and correction, is deliberately unchanged.
ANSWER_GROUPED_NUMBER = re.compile(rf"(?<!{_DIGIT})(?<![.,]){_DIGIT}{{1,3}}(?:(?:,|{_SEP}){_DIGIT}{{3}})+(?!{_DIGIT}|,{_DIGIT})")


def has_grouped_number(text: str) -> bool:
    return bool(ANSWER_GROUPED_NUMBER.search(text))


def is_numeric_question(question: dict) -> bool:
    """Questions touched by the reversal: a grouped number in the English question or answer."""
    return has_grouped_number(question["question"]["en"]) or has_grouped_number(question["answer"]["en"])


def audit_numbers(en: str, ar: str) -> Counter:
    """For each thousands-separated English number, how the aligned Arabic text writes it.

    Forms: forward_space ("278 707"), reversed_space ("707 278"), comma ("278,707"), joined
    ("278707"), ambiguous (forward and reversed are the same string, e.g. "200 200"), not_found.
    """
    ar_norm = normalize(ar, "ar")
    counts: Counter = Counter()
    for groups in english_numbers(en):
        written = {
            "forward_space": " ".join(groups),
            "reversed_space": " ".join(reversed(groups)),
            "comma": ",".join(groups),
            "joined": "".join(groups),
        }
        found = [name for name, s in written.items() if re.search(rf"(?<!\d){re.escape(s)}(?!\d)", ar_norm)]
        if "reversed_space" in found and "forward_space" in found:
            found = ["ambiguous"]
        counts[found[0] if found else "not_found"] += 1
    return counts


def _digit(ch: str) -> str:
    d = int(ch)
    return f"[{d}{chr(0x0660 + d)}{chr(0x06F0 + d)}]"


def _reversed_pattern(groups: list[str]) -> re.Pattern:
    parts = []
    for i, group in enumerate(reversed(groups)):
        if i:
            parts.append(f"({_SEP}+)")
        parts.append("(" + "".join(_digit(ch) for ch in group) + ")")
    return re.compile(rf"(?<!{_DIGIT})(?<!{_DIGIT}{_SEP})" + "".join(parts) + rf"(?!{_SEP}*{_DIGIT})")


def correct_reversed_numbers(en: str, ar: str) -> tuple[str, int]:
    """Rewrite reversed digit groups in `ar` for numbers the aligned `en` text contains. Returns (text, count)."""
    corrected = 0
    for groups in sorted({tuple(g) for g in english_numbers(en)}, key=lambda g: (-len(g), g)):
        if list(groups) == list(reversed(groups)):
            continue

        def swap(match: re.Match) -> str:
            captured = match.groups()
            ordered = list(reversed(captured[0::2]))
            out = [ordered[0]]
            for separator, group in zip(captured[1::2], ordered[1:]):
                out += [separator, group]
            return "".join(out)

        ar, n = _reversed_pattern(list(groups)).subn(swap, ar)
        corrected += n
    return ar, corrected
