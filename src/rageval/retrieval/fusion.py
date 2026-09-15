"""Reciprocal rank fusion (Cormack, Clarke & Büttcher, 2009).

BM25 scores are unbounded and query-dependent; cosine similarities sit in a narrow band. Adding
them is meaningless, so RRF ignores scores and uses only rank positions:

    RRF(d) = sum over rankings r of 1 / (k + rank_r(d)),   rank starting at 1, k = 60

A document ranked 1st in one list gets 1/61, one ranked 10th gets 1/70: k flattens the curve so
no single retriever's top hit dominates, and documents ranked well by *both* rise. k = 60 is the
paper's value, used untuned; tuning it on the test questions would inflate the result. A
document absent from a ranking contributes nothing from it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60, depth: int | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking[:depth] if depth else ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
