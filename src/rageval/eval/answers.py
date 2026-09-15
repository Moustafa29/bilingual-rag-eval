"""Scoring a generated answer against the reference answer, without an LLM.

Both measures use `rageval.text.answer_tokens`: normalized tokens with Arabic orthography and digits
folded ("٢٠٠٩" matches "2009") and English articles and the Arabic definite article dropped. Without
that, correct Arabic answers are scored wrong for how they are written.

- exact_match: identical answer tokens.
- token_f1: SQuAD-style overlap. Reported, never used as a gate: a 0.5 token-overlap threshold
  accepted a wrong answer during question building (docs/corpus.md §3).
"""

from __future__ import annotations

from rageval.text import answer_tokens, token_f1


def exact_match(prediction: str, reference: str, lang: str) -> bool:
    pred = answer_tokens(prediction, lang)
    return bool(pred) and pred == answer_tokens(reference, lang)


def answer_scores(prediction: str, reference: str, lang: str) -> dict[str, float]:
    return {"exact_match": float(exact_match(prediction, reference, lang)), "token_f1": token_f1(prediction, reference, lang)}
