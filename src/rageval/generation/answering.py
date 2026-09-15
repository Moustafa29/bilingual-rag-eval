"""Ask the answering model one question, with or without passages, and detect abstentions.

Answers are plain text, not JSON, so small or local models can be used. Abstention uses a fixed ASCII
token in both languages (NOT_IN_CONTEXT with passages, UNKNOWN without), so it is detected exactly
rather than by guessing at phrasings such as "the passages do not say" or "لا تذكر النصوص".
"""

from __future__ import annotations

from rageval.questions.builder import LANG_NAMES, Prompts, format_passages

ABSTAIN_TOKEN = {"closed_book": "UNKNOWN", "oracle": "NOT_IN_CONTEXT", "rag": "NOT_IN_CONTEXT"}


def is_abstention(answer: str, condition: str) -> bool:
    return ABSTAIN_TOKEN[condition] in answer.upper()


def answer_question(client, prompts: Prompts, question: str, lang: str, condition: str, passages: list[str]) -> dict:
    if condition == "closed_book":
        if passages:
            raise ValueError("closed_book answers must not see passages")
        prompt = prompts.render("answer_closed_book", lang_name=LANG_NAMES[lang], question=question)
    elif condition in ("oracle", "rag"):
        prompt = prompts.render("answer_rag", lang_name=LANG_NAMES[lang], question=question, passages=format_passages(passages))
    else:
        raise ValueError(f"unknown condition: {condition}")
    result = client.chat([{"role": "user", "content": prompt}])
    answer = result.text.strip()
    return {
        "answer": answer,
        "abstained": is_abstention(answer, condition),
        "cache_key": result.cache_key,
        "cached": result.cached,
        "usage": result.usage,
    }
