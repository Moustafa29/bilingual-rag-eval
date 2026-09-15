"""References between UN documents, used to build bridge (multi-hop) questions.

Every UN document has a symbol such as A/CN.4/452, DP/2002/13/Add.1 or S/RES/1711(2006). The
corpus stores each document at a path derived from its symbol: `1992/a/cn_4/452`,
`2002/dp/2002/13/add_1`, `2006/s/res/1711_2006_` (year folder, then the symbol lower-cased,
"/" as folders, and each ".", space, hyphen or parenthesis as "_"). `symbol_key` applies that
rule so a reference found in text can be looked up by path. The corpus build reports how often
the rule reproduces each fetched document's own header symbol.
"""

from __future__ import annotations

import re

# A capitalised body prefix followed by one or more "/segment" parts, e.g. A/CN.4/452 or
# S/RES/1711(2006). Parentheses are allowed inside segments; unbalanced trailing ones are the
# surrounding sentence's and are stripped.
_SYMBOL = re.compile(r"(?<![\w/])[A-Z][A-Z-]{0,11}(?:/[A-Za-z0-9.()\-]+)+")

# General Assembly shorthand: "resolution 47/33" means A/RES/47/33. Restricted to sessions
# 40-69 (the corpus covers 1990-2014) because other bodies use the same "n/m" form: Human
# Rights Council resolution 7/19 is not A/RES/7/19.
_GA_RESOLUTION = re.compile(r"\bresolutions?\s+((?:4|5|6)\d/\d{1,3})\b")

# Security Council shorthand: "resolution 1325 (2000)" means S/RES/1325(2000).
_SC_RESOLUTION = re.compile(r"\bresolutions?\s+(\d{3,4})\s*\(((?:19|20)\d\d)\)")


def _strip_unbalanced(symbol: str) -> str:
    symbol = symbol.rstrip(".-")
    while symbol.endswith(")") and symbol.count(")") > symbol.count("("):
        symbol = symbol[:-1].rstrip(".-")
    return symbol


# Joint symbols name one document issued under two series, e.g. A/55/432-S/2000/921. Split at a
# hyphen followed by another body prefix and "/"; A/ES-10/648 is not split (digits follow).
_JOINT = re.compile(r"-(?=[A-Z][A-Z-]{0,11}/)")


def find_references(text: str) -> set[str]:
    refs = set()
    for match in _SYMBOL.finditer(text):
        for part in _JOINT.split(match.group(0)):
            symbol = _strip_unbalanced(part)
            if any(ch.isdigit() for ch in symbol) and "/" in symbol:
                refs.add(symbol)
    for match in _GA_RESOLUTION.finditer(text):
        refs.add(f"A/RES/{match.group(1)}")
    for match in _SC_RESOLUTION.finditer(text):
        refs.add(f"S/RES/{match.group(1)}({match.group(2)})")
    return refs


def symbol_key(symbol: str) -> str:
    segments = symbol.strip().rstrip(".").lower().split("/")
    return "/".join(re.sub(r"[.\s()\-]", "_", segment) for segment in segments)


def doc_key(doc_id: str) -> str:
    """Drop the year folder: `"1992/a/cn_4/452"` -> `"a/cn_4/452"`."""
    return doc_id.split("/", 1)[1]
