"""Which chunks questions are written from, and in what order.

Every chunk stays in the retrieval corpus as a potential distractor. Questions are only written
from *content* chunks, meaning prose that states facts. Cover pages, tables of contents and
distribution lists are skipped. Candidates are drawn at most one per document, so a handful of
long reports cannot dominate the question set.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

from rageval.corpus.symbols import find_references, symbol_key

_TOC_LINE = re.compile(r"(?:\.\s*){2,}\d+\s*$|\s\.\s+\d+\s*$")


def is_content_chunk(chunk: dict, rules: dict) -> bool:
    en = chunk["en"]
    if chunk.get("truncated_e5") or not chunk["ar"].strip():
        return False
    if len(en) < rules["min_chars"] or chunk["one_to_one_ratio"] < rules["min_one_to_one_ratio"]:
        return False
    if sum(ch.isalpha() for ch in en) / len(en) < rules["min_letter_ratio"]:
        return False
    lines = [line for line in en.split("\n") if line.strip()]
    if sum(bool(_TOC_LINE.search(line)) for line in lines) / len(lines) > rules["max_toc_line_ratio"]:
        return False
    return max(len(line.split()) for line in lines) >= rules["min_paragraph_words"]


def group_by_doc(chunks: list[dict]) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    return by_doc


def single_candidates(by_doc: dict[str, list[dict]], rules: dict, seed: int) -> Iterator[dict]:
    rng = random.Random(f"{seed}:single")
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc in docs:
        options = [c for c in by_doc[doc] if is_content_chunk(c, rules)]
        if options:
            yield rng.choice(options)


def bridge_candidates(
    by_doc: dict[str, list[dict]],
    key_to_docs: dict[str, list[str]],
    rules: dict,
    seed: int,
    max_b_options: int = 5,
) -> Iterator[tuple[dict, dict, str]]:
    """Yield (chunk A, chunk B, symbol): A cites the document B belongs to.

    B is drawn from the first few content chunks of the cited document, where UN documents
    state what they are and what they cover.
    """
    rng = random.Random(f"{seed}:bridge")
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc in docs:
        for a in by_doc[doc]:
            if not is_content_chunk(a, rules):
                continue
            targets = sorted(
                (symbol, target)
                for symbol in find_references(a["en"])
                for target in key_to_docs.get(symbol_key(symbol), ())
                if target != doc and target in by_doc
            )
            if not targets:
                continue
            symbol, target = rng.choice(targets)
            b_options = [c for c in by_doc[target] if is_content_chunk(c, rules)][:max_b_options]
            if not b_options:
                continue
            yield a, rng.choice(b_options), symbol
            break
