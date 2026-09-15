"""Which passages the answering model sees, and where the gold passage sits among them.

Three conditions per question and language:
- closed_book: no passages. What the model answers from memory, which is what the "missed but
  correct" attribution cell can only be read against.
- oracle: the gold chunk(s) plus randomly drawn distractors, k passages in a seeded shuffled order.
  The generation ceiling: accuracy when retrieval is perfect.
- rag: the top k of a retrieval run.

"Retrieved" means every relevance group has a member in the context actually given to the model,
not somewhere in a longer ranking.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def gold_positions(context_ids: Sequence[str], relevant_groups: Sequence[Sequence[str]]) -> list[int | None]:
    """1-based position of each relevance group's first member in the context, or None if absent."""
    position = {}
    for i, chunk_id in enumerate(context_ids, start=1):
        position.setdefault(chunk_id, i)
    return [min((position[c] for c in group if c in position), default=None) for group in relevant_groups]


def retrieved_all(context_ids: Sequence[str], relevant_groups: Sequence[Sequence[str]]) -> bool:
    return all(p is not None for p in gold_positions(context_ids, relevant_groups))


def oracle_context(
    qid: str,
    gold_chunks: Sequence[str],
    relevant_groups: Sequence[Sequence[str]],
    pool_ids: Sequence[str],
    k: int,
    seed: int,
) -> list[str]:
    """Gold chunks plus distractors that belong to no relevance group, shuffled deterministically per question."""
    if len(gold_chunks) > k:
        raise ValueError(f"{qid}: {len(gold_chunks)} gold chunks do not fit in a context of {k}")
    rng = random.Random(f"{seed}:oracle:{qid}")
    excluded = {chunk_id for group in relevant_groups for chunk_id in group} | set(gold_chunks)
    candidates = sorted(set(pool_ids) - excluded)
    ids = list(gold_chunks) + rng.sample(candidates, k - len(gold_chunks))
    rng.shuffle(ids)
    return ids


def rag_context(ranking: Sequence[str], k: int) -> list[str]:
    return list(ranking[:k])
