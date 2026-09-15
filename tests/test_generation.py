from pathlib import Path

import pytest

from rageval.eval.answers import answer_scores, exact_match
from rageval.eval.attribution import CELLS, attribution_table, cell, decomposition
from rageval.generation.answering import answer_question, is_abstention
from rageval.generation.contexts import gold_positions, oracle_context, rag_context, retrieved_all
from rageval.llm.client import ChatResult
from rageval.questions.builder import Prompts

PROMPTS = Prompts(Path(__file__).resolve().parents[1] / "prompts")


def test_gold_positions_and_retrieval_use_the_context_given_to_the_model():
    groups = [["gold", "copy"], ["second"]]
    assert gold_positions(["x", "copy", "gold", "y"], groups) == [2, None]
    assert not retrieved_all(["x", "copy", "gold", "y"], groups)
    assert retrieved_all(["second", "x", "gold"], groups)
    assert rag_context(["a", "b", "c", "d", "e", "f"], 5) == ["a", "b", "c", "d", "e"]


def test_oracle_context_contains_gold_excludes_relevant_and_is_deterministic():
    pool = [f"c{i}" for i in range(50)] + ["gold", "copy"]
    first = oracle_context("q1", ["gold"], [["gold", "copy"]], pool, k=5, seed=7)
    assert len(first) == 5 and "gold" in first and "copy" not in first
    assert first == oracle_context("q1", ["gold"], [["gold", "copy"]], pool, k=5, seed=7)
    assert first != oracle_context("q2", ["gold"], [["gold", "copy"]], pool, k=5, seed=7)
    with pytest.raises(ValueError):
        oracle_context("q3", ["a", "b"], [["a"], ["b"]], pool, k=1, seed=7)


class FakeClient:
    model = "fake"

    def __init__(self, text):
        self.text, self.prompts = text, []

    def chat(self, messages):
        self.prompts.append(messages[0]["content"])
        return ChatResult(self.text, "key", False, {})


def test_closed_book_prompt_has_no_passages_and_detects_unknown():
    client = FakeClient("UNKNOWN")
    out = answer_question(client, PROMPTS, "When was it adopted?", "en", "closed_book", [])
    assert out["abstained"] and "Passage 1" not in client.prompts[0]
    with pytest.raises(ValueError):
        answer_question(client, PROMPTS, "q", "en", "closed_book", ["a passage"])


def test_rag_prompt_includes_passages_and_detects_not_in_context_in_arabic_answers():
    client = FakeClient("not_in_context")
    out = answer_question(client, PROMPTS, "متى اعتمد؟", "ar", "rag", ["نص أول", "نص ثان"])
    assert out["abstained"]
    assert "Passage 2:" in client.prompts[0] and "Arabic" in client.prompts[0]
    assert not is_abstention("25 نوفمبر 1992", "rag")


def test_exact_match_folds_arabic_digits_articles_and_orthography():
    assert exact_match("٢٠٠٩", "2009", "ar")
    assert exact_match("الجمعية العامة", "جمعية عامة", "ar")
    assert exact_match("the Security Council", "Security Council", "en")
    assert not exact_match("", "", "en")
    assert answer_scores("25 November 1992", "25 November", "en")["token_f1"] == pytest.approx(0.8)


def row(retrieved, correct, abstained=False, supported=None):
    return {"retrieved": retrieved, "correct": correct, "abstained": abstained, "supported": supported}


def test_attribution_cells_and_rates():
    rows = [row(True, True, supported=True), row(True, False, supported=False), row(False, True, supported=False), row(False, False, abstained=True)]
    table = attribution_table(rows)
    assert table["cells"] == {c: 1 for c in CELLS}
    assert table["accuracy"] == 0.5 and table["retrieval_rate"] == 0.5
    assert table["abstention_rate"] == 0.25 and table["abstention_when_missed"] == 0.5 and table["abstention_when_retrieved"] == 0.0
    # Hallucination is judged only on non-abstaining answers: 2 unsupported of 3.
    assert table["hallucination_rate"] == pytest.approx(2 / 3) and table["hallucination_judged_n"] == 3
    assert cell(False, True) == "missed_correct"


def test_attribution_without_support_judgements_and_decomposition():
    table = attribution_table([row(True, True), row(False, False)])
    assert table["hallucination_rate"] is None and table["hallucination_judged_n"] == 0
    assert decomposition(oracle_accuracy=0.9, rag_accuracy=0.6) == pytest.approx({"retrieval_cost": 0.3, "generation_cost": 0.1})
    with pytest.raises(ValueError):
        attribution_table([])
