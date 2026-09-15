"""Which chunks questions are written from, and in what order.

Every chunk stays in the retrieval corpus as a potential distractor. Questions are only written
from *content* chunks, meaning prose that states facts. Cover pages, tables of contents and
distribution lists are skipped. Candidates are drawn at most one per document, so a handful of
long reports cannot dominate the question set.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterator

from rageval.corpus.numbers import has_thousands_number
from rageval.corpus.symbols import find_references, symbol_key
from rageval.text import contains_span

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


def numeric_candidates(by_doc: dict[str, list[dict]], rules: dict, seed: int) -> Iterator[dict]:
    """Content chunks holding a number whose Arabic digit groups were corrected, one per document.

    These questions carry the corrected-vs-uncorrected comparison: without them, a random question
    set contains too few numbers for that comparison to have a confidence interval.
    """
    rng = random.Random(f"{seed}:numeric")
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc in docs:
        options = [
            c
            for c in by_doc[doc]
            if is_content_chunk(c, rules) and c.get("numbers_corrected", 0) > 0 and has_thousands_number(c["en"])
        ]
        if options:
            yield rng.choice(options)


def _subject_term(keyword: str) -> str:
    """UNBIS terms can carry a subdivision ("SIERRA LEONE--POLITICAL CONDITIONS") or qualifier ("(AFRICA)")."""
    return re.sub(r"\s*\([^)]*\)", "", keyword.split("--")[0]).strip()


def comparison_candidates(
    by_doc: dict[str, list[dict]], rules: dict, seed: int, max_keyword_docs: int = 20
) -> Iterator[tuple[dict, dict, str]]:
    """Yield (chunk 1, chunk 2, term): two documents indexed under the same UNBIS subject term, each
    with a content chunk that mentions the term.

    Rarer shared terms are tried first: two documents under "RWANDA" or "MARITIME TRANSPORT" are far
    more likely to state comparable facts than two under "HUMAN RIGHTS" (127 documents). Terms shared
    by more than `max_keyword_docs` documents are not used. Each document is paired at most once.
    """
    rng = random.Random(f"{seed}:comparison")
    doc_terms = {
        doc: sorted({_subject_term(k) for k in chunks[0].get("keywords", []) if _subject_term(k)})
        for doc, chunks in by_doc.items()
    }
    docs_by_term: dict[str, list[str]] = defaultdict(list)
    for doc in sorted(doc_terms):
        for term in doc_terms[doc]:
            docs_by_term[term].append(doc)

    def mentioning(doc: str, term: str) -> list[dict]:
        return [c for c in by_doc[doc] if is_content_chunk(c, rules) and contains_span(c["en"], term, "en")]

    used: set[str] = set()
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc in docs:
        if doc in used:
            continue
        terms = sorted(
            (t for t in doc_terms[doc] if 2 <= len(docs_by_term[t]) <= max_keyword_docs),
            key=lambda t: (len(docs_by_term[t]), t),
        )
        for term in terms:
            own = mentioning(doc, term)
            if not own:
                continue
            partners = [p for p in docs_by_term[term] if p != doc and p not in used]
            rng.shuffle(partners)
            pair = next(((p, other) for p in partners if (other := mentioning(p, term))), None)
            if pair is None:
                continue
            partner, other = pair
            used.update({doc, partner})
            yield rng.choice(own), rng.choice(other), term
            break


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
