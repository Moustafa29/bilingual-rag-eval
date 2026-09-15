"""Four-cell attribution of answer errors to retrieval or generation.

For every question and language, two facts: was the gold passage in the context given to the model
(`rageval.generation.contexts.retrieved_all`), and was the answer correct.

                     answer correct        answer wrong
    gold retrieved   system worked         generation failure: had the evidence, didn't use it
    gold missed      parametric answer     retrieval failure (generation unidentifiable)

"Missed but correct" is the model answering from memory or by chance, a cell most RAG evaluations
cannot see. It is read against the closed-book condition: if closed-book accuracy is high, that cell
is memory, not luck.

Abstentions count as wrong answers in the cells and are also reported separately, because a model
that abstains often looks safe on hallucination for a bad reason.

Hallucination: a non-abstaining answer that no passage in its context supports (a judge decision,
recorded per row as `supported`).

Decomposition across conditions, per language:
    retrieval_cost  = oracle accuracy - rag accuracy
    generation_cost = 1 - oracle accuracy
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

CELLS = ("retrieved_correct", "retrieved_wrong", "missed_correct", "missed_wrong")


def cell(retrieved: bool, correct: bool) -> str:
    return f"{'retrieved' if retrieved else 'missed'}_{'correct' if correct else 'wrong'}"


def attribution_table(rows: Sequence[dict]) -> dict:
    """`rows` need `retrieved`, `correct`, `abstained`; `supported` is optional (None if not judged)."""
    if not rows:
        raise ValueError("no rows")
    counts = Counter(cell(r["retrieved"], r["correct"]) for r in rows)
    n = len(rows)
    table = {
        "n": n,
        "cells": {c: counts[c] for c in CELLS},
        "cell_shares": {c: counts[c] / n for c in CELLS},
        "accuracy": sum(r["correct"] for r in rows) / n,
        "retrieval_rate": sum(r["retrieved"] for r in rows) / n,
        "abstention_rate": sum(r["abstained"] for r in rows) / n,
        "abstention_when_retrieved": _rate([r["abstained"] for r in rows if r["retrieved"]]),
        "abstention_when_missed": _rate([r["abstained"] for r in rows if not r["retrieved"]]),
    }
    judged = [r for r in rows if not r["abstained"] and r.get("supported") is not None]
    table["hallucination_rate"] = _rate([not r["supported"] for r in judged])
    table["hallucination_judged_n"] = len(judged)
    return table


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def decomposition(oracle_accuracy: float, rag_accuracy: float) -> dict[str, float]:
    return {"retrieval_cost": oracle_accuracy - rag_accuracy, "generation_cost": 1.0 - oracle_accuracy}
