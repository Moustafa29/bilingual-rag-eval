"""Retrieval metrics over relevance *groups*.

A question's relevance is a list of groups: one group per gold chunk, holding that chunk and its
near-duplicates (`rageval.questions.dedup`). Retrieving any member of a group retrieves the
group, and a group is credited once, at its first position. Crediting every duplicate would let
a retriever that returns five copies of one paragraph look better than one that returns the two
different paragraphs a bridge question needs.

- recall@k      share of groups with a member in the top k
- all_recall@k  1 if every group has a member in the top k, else 0. For bridge questions this is
                the one that matters: with one of two passages, the answer cannot be composed.
- mrr@k         1 / rank of the first retrieved member of any group, 0 if none in the top k
- ndcg@k        binary gain per group at its first position, discounted by log2(rank + 1),
                divided by the best possible (all groups at the top positions)
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def first_hit_ranks(ranking: Sequence[str], groups: Sequence[Sequence[str]]) -> list[int | None]:
    """1-based rank at which each group is first retrieved, or None."""
    position: dict[str, int] = {}
    for rank, chunk_id in enumerate(ranking, start=1):
        position.setdefault(chunk_id, rank)
    return [min((position[c] for c in group if c in position), default=None) for group in groups]


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def all_recall_at(ranks: Sequence[int | None], k: int) -> float:
    return float(all(r is not None and r <= k for r in ranks))


def mrr_at(ranks: Sequence[int | None], k: int) -> float:
    hits = [r for r in ranks if r is not None and r <= k]
    return 1.0 / min(hits) if hits else 0.0


def ndcg_at(ranks: Sequence[int | None], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ranks), k) + 1))
    return dcg / ideal


def question_metrics(ranking: Sequence[str], groups: Sequence[Sequence[str]], ks: Sequence[int] = (1, 5, 10, 20)) -> dict[str, float]:
    if not groups:
        raise ValueError("a question needs at least one relevance group")
    ranks = first_hit_ranks(ranking, groups)
    out = {f"recall@{k}": recall_at(ranks, k) for k in ks}
    out.update({f"all_recall@{k}": all_recall_at(ranks, k) for k in ks})
    out["mrr@10"] = mrr_at(ranks, 10)
    out["ndcg@10"] = ndcg_at(ranks, 10)
    return out
