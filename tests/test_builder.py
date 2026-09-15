import json
import re
from pathlib import Path

from rageval.llm.client import ChatResult, DailyLimitReached
from rageval.questions.builder import Attempt, Prompts, QuestionBuilder, replace_kinds, run_build

PROMPTS = Prompts(Path(__file__).resolve().parents[1] / "prompts")
JUDGE = "Two short answers"

SINGLE = {
    "chunk_id": "1992/a/1#0003",
    "en": "The General Assembly adopted resolution 47/33 on 25 November 1992 concerning the International Law Commission.",
    "ar": "اعتمدت الجمعية العامة القرار ٤٧/٣٣ في ٢٥ تشرين الثاني/نوفمبر ١٩٩٢ بشأن لجنة القانون الدولي.",
}
NUMERIC = {
    "chunk_id": "2010/a/64/819#0005",
    "en": "As of January 2010, 21,456 individuals displaced during the August 2008 conflict had been relocated.",
    "ar": "وحتى كانون الثاني/يناير 2010، تم نقل 21 456 شخصا من بين الذين شردوا أثناء نزاع آب/أغسطس 2008.",
}
CHUNK_A = {
    "chunk_id": "2006/a/61/1#0002",
    "header": "Document A/61/1",
    "en": "The Assembly requested the Secretary-General to report on illicit trade in small arms (A/60/88) at its next session.",
    "ar": "طلبت الجمعية إلى الأمين العام تقديم تقرير عن الاتجار غير المشروع بالأسلحة الصغيرة (A/60/88) في دورتها المقبلة.",
}
CHUNK_B = {
    "chunk_id": "2005/a/60/88#0001",
    "header": "Document A/60/88",
    "en": "The report recommends that States mark all small arms at the point of manufacture.",
    "ar": "يوصي التقرير بأن تقوم الدول بوسم جميع الأسلحة الصغيرة عند صنعها.",
}
HAITI = {
    "chunk_id": "2010/a/res/64/1#0001",
    "en": "The Assembly appropriated 50 million dollars for the Mission in Haiti for the period.",
    "ar": "خصصت الجمعية 50 مليون دولار للبعثة في هايتي للفترة.",
}
LIBERIA = {
    "chunk_id": "2010/a/res/64/2#0001",
    "en": "The Assembly appropriated 80 million dollars for the Mission in Liberia for the period.",
    "ar": "خصصت الجمعية 80 مليون دولار للبعثة في ليبريا للفترة.",
}


class FakeClient:
    def __init__(self, model, respond):
        self.model = model
        self.respond = respond
        self.prompts = []

    def chat(self, messages):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return ChatResult(json.dumps(self.respond(prompt), ensure_ascii=False), f"k{len(self.prompts)}", False, {"prompt_tokens": 7, "completion_tokens": 3})


def single_generator(prompt):
    if prompt.startswith("Translate"):
        return {"question": "في أي تاريخ اعتمدت الجمعية العامة قرارها بشأن لجنة القانون الدولي؟", "answer": "25 نوفمبر 1992"}
    return {
        "question": "On what date did the General Assembly adopt its decision on the International Law Commission?",
        "answer": "25 November 1992",
        "evidence": SINGLE["en"],
    }


def single_verifier(same=True, answer="٢٥ تشرين الثاني/نوفمبر ١٩٩٢"):
    def respond(prompt):
        if prompt.startswith(JUDGE):
            return {"same": same}
        return {"answerable": True, "answer": answer}

    return respond


def test_single_hop_accepted_after_equivalence_judgement():
    generator, verifier = FakeClient("g", single_generator), FakeClient("v", single_verifier())
    attempt = QuestionBuilder(generator, verifier, PROMPTS).single(SINGLE, "en->ar")
    assert attempt.status == "accepted"
    record = attempt.record
    assert record["answer"] == {"en": "25 November 1992", "ar": "٢٥ تشرين الثاني/نوفمبر ١٩٩٢"}
    assert record["kind"] == "single" and record["type"] == "single"
    assert 0 < record["answer_f1_vs_translation"] < 1  # recorded, not gating
    assert len(record["llm_calls"]) == 4  # generate, translate, verify, judge
    assert verifier.prompts[-1].startswith(JUDGE)
    translate_prompt = next(p for p in generator.prompts if p.startswith("Translate"))
    assert SINGLE["en"] not in translate_prompt


def test_single_hop_rejected_when_answers_state_different_facts():
    # The false-accept case: token overlap would pass, the judgement says the facts differ.
    attempt = QuestionBuilder(FakeClient("g", single_generator), FakeClient("v", single_verifier(same=False)), PROMPTS).single(SINGLE, "en->ar")
    assert attempt.status == "answer_not_equivalent"
    assert attempt.record is None


def test_single_hop_rejects_hallucinated_evidence_and_keeps_question_text():
    def generator(prompt):
        out = single_generator(prompt)
        if "evidence" in out:
            out["evidence"] = "The Assembly adopted this on 1 January 1990."
        return out

    attempt = QuestionBuilder(FakeClient("g", generator), FakeClient("v", single_verifier()), PROMPTS).single(SINGLE, "en->ar")
    assert attempt.status == "evidence_not_in_passage"
    assert len(attempt.calls) == 1
    assert attempt.question["en"].startswith("On what date")


def test_single_hop_rejects_unanswerable_translation():
    verifier = FakeClient("v", lambda p: {"answerable": False, "answer": ""})
    attempt = QuestionBuilder(FakeClient("g", single_generator), verifier, PROMPTS).single(SINGLE, "en->ar")
    assert attempt.status == "not_answerable_translated"


def test_context_reference_rejection_keeps_question_for_audit():
    def generator(prompt):
        out = single_generator(prompt)
        if "evidence" in out:
            out["question"] = "According to the mentioned decision, when was it adopted?"
        return out

    attempt = QuestionBuilder(FakeClient("g", generator), FakeClient("v", single_verifier()), PROMPTS).single(SINGLE, "en->ar")
    assert attempt.status == "context_reference"
    assert attempt.question == {"en": "According to the mentioned decision, when was it adopted?"}


def test_numeric_mode_requires_a_thousands_separated_answer():
    builder = QuestionBuilder(FakeClient("g", single_generator), FakeClient("v", single_verifier()), PROMPTS)
    attempt = builder.single(SINGLE, "en->ar", kind="numeric", prompt="generate_numeric", require_number=True)
    assert attempt.status == "answer_not_a_thousands_number"
    assert len(attempt.calls) == 1  # rejected before translation


def test_numeric_mode_accepts_a_corrected_number():
    def generator(prompt):
        if prompt.startswith("Translate"):
            return {"question": "كم عدد الأشخاص الذين تم نقلهم حتى كانون الثاني/يناير 2010؟", "answer": "21 456 شخصا"}
        return {"question": "How many people displaced in the August 2008 conflict had been relocated by January 2010?", "answer": "21,456 individuals", "evidence": NUMERIC["en"]}

    generator_client = FakeClient("g", generator)
    attempt = QuestionBuilder(generator_client, FakeClient("v", single_verifier(answer="21 456 شخصا")), PROMPTS).single(
        NUMERIC, "en->ar", kind="numeric", prompt="generate_numeric", require_number=True
    )
    assert attempt.status == "accepted"
    assert attempt.record["kind"] == "numeric" and attempt.record["type"] == "single"
    assert "digit groups" in generator_client.prompts[0]


def choose_by_text(prompt, wanted):
    shown = dict(re.findall(r"^Option ([12]): (.*)$", prompt, re.M))
    return {"choice": next((int(k) for k, v in shown.items() if v == wanted), None)}


def comparison_generator(prompt):
    if prompt.startswith("Translate"):
        return {"question": "أي البعثتين حصلت على اعتماد أكبر، البعثة في هايتي أم البعثة في ليبريا؟", "options": ["البعثة في هايتي", "البعثة في ليبريا"]}
    return {
        "question": "Which received the larger appropriation, the Mission in Haiti or the Mission in Liberia?",
        "options": ["the Mission in Haiti", "the Mission in Liberia"],
        "answer": 2,
        "evidence_1": HAITI["en"],
        "evidence_2": LIBERIA["en"],
    }


def test_comparison_accepted_when_only_both_passages_decide():
    def verifier(prompt):
        return choose_by_text(prompt, "البعثة في ليبريا") if "Passage 2:" in prompt else {"choice": None}

    ver = FakeClient("v", verifier)
    attempt = QuestionBuilder(FakeClient("g", comparison_generator), ver, PROMPTS).comparison(HAITI, LIBERIA, "PEACEKEEPING FINANCING", "en->ar")
    assert attempt.status == "accepted"
    record = attempt.record
    assert record["answer"] == {"en": "the Mission in Liberia", "ar": "البعثة في ليبريا"}
    assert record["answer_index"] == 2 and record["type"] == "comparison"
    assert record["gold_chunks"] == [HAITI["chunk_id"], LIBERIA["chunk_id"]]
    assert len(attempt.calls) == 5  # generate, translate, shortcut 1, shortcut 2, verify both


def test_comparison_strict_shortcut_rejects_any_pick_from_one_passage():
    # With one passage the verifier guesses Haiti, which is wrong; any pick still rejects.
    def verifier(prompt):
        return choose_by_text(prompt, "the Mission in Haiti") if "Passage 2:" not in prompt else choose_by_text(prompt, "البعثة في ليبريا")

    attempt = QuestionBuilder(FakeClient("g", comparison_generator), FakeClient("v", verifier), PROMPTS).comparison(HAITI, LIBERIA, "T", "en->ar")
    assert attempt.status == "shortcut_1_alone"


def test_comparison_rejects_wrong_choice_regardless_of_display_order():
    def verifier(prompt):
        return choose_by_text(prompt, "البعثة في هايتي") if "Passage 2:" in prompt else {"choice": None}

    attempt = QuestionBuilder(FakeClient("g", comparison_generator), FakeClient("v", verifier), PROMPTS).comparison(HAITI, LIBERIA, "T", "en->ar")
    assert attempt.status == "verified_choice_mismatch"


def test_comparison_rejects_options_missing_from_question():
    def generator(prompt):
        out = comparison_generator(prompt)
        if "evidence_1" in out:
            out["question"] = "Which mission received more money?"
        return out

    attempt = QuestionBuilder(FakeClient("g", generator), FakeClient("v", lambda p: {"choice": None}), PROMPTS).comparison(HAITI, LIBERIA, "T", "en->ar")
    assert attempt.status == "options_not_in_question"


def bridge_generator(prompt):
    if prompt.startswith("Translate"):
        return {"question": "ما الذي أوصى به تقرير الأمين العام عن الاتجار غير المشروع بالأسلحة الصغيرة بأن تسمه الدول؟", "answer": "جميع الأسلحة الصغيرة"}
    return {
        "question": "What did the Secretary-General's report on illicit small arms trade recommend that States mark?",
        "answer": "all small arms",
        "evidence_a": CHUNK_A["en"],
        "evidence_b": CHUNK_B["en"],
    }


def test_bridge_accepted_with_headers_strict_shortcuts_and_judgement():
    def verifier(prompt):
        if prompt.startswith(JUDGE):
            return {"same": True}
        if "Passage 2:" in prompt:
            return {"answerable": True, "answer": "جميع الأسلحة الصغيرة"}
        return {"answerable": False, "answer": ""}

    generator, ver = FakeClient("g", bridge_generator), FakeClient("v", verifier)
    attempt = QuestionBuilder(generator, ver, PROMPTS).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "accepted"
    assert len(attempt.calls) == 6  # generate, translate, shortcut B, shortcut A, verify both, judge
    assert "Passage B (Document A/60/88)" in generator.prompts[0]
    verify_prompts = [p for p in ver.prompts if not p.startswith(JUDGE)]
    assert all("Document A/60/88" in p or "Document A/61/1" in p for p in verify_prompts)


def test_bridge_strict_shortcut_rejects_even_a_different_answer():
    verifier = FakeClient("v", lambda p: {"answerable": True, "answer": "something else entirely"})
    attempt = QuestionBuilder(FakeClient("g", bridge_generator), verifier, PROMPTS).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "shortcut_b_alone"


def test_bridge_rejects_symbol_in_question():
    def generator(prompt):
        out = bridge_generator(prompt)
        if "evidence_a" in out:
            out["question"] = "What did report A/60/88 recommend that States mark?"
        return out

    attempt = QuestionBuilder(FakeClient("g", generator), FakeClient("v", lambda p: {}), PROMPTS).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "document_number_in_question"


def test_run_build_balances_directions_stops_on_daily_limit_and_records_questions():
    def attempt_fn(candidate, direction):
        if candidate == "limit":
            raise DailyLimitReached("TPD")
        return Attempt(status="accepted", record={"direction": direction}, question={"en": candidate})

    result = run_build("single", attempt_fn, ["c1", "c2", "c3", "c4"], n_target=4, max_candidates=10)
    assert [r["direction"] for r in result.accepted] == ["en->ar", "ar->en", "en->ar", "ar->en"]
    assert result.stopped is None
    assert result.outcomes[0]["question"] == {"en": "c1"}

    result = run_build("single", attempt_fn, ["c1", "limit", "c3"], n_target=3, max_candidates=10)
    assert len(result.accepted) == 1 and result.stopped.startswith("daily limit")


def test_replace_kinds_keeps_other_kinds():
    previous = [{"kind": "single", "qid": "s1"}, {"kind": "comparison", "qid": "c1"}]
    new = [{"kind": "comparison", "qid": "c2"}]
    assert replace_kinds(previous, new, {"comparison"}, "kind") == [{"kind": "single", "qid": "s1"}, {"kind": "comparison", "qid": "c2"}]
    assert replace_kinds(previous, [], {"single", "comparison"}, "kind") == []


def test_run_build_uses_other_direction_when_quota_full():
    def attempt_fn(candidate, direction):
        ok = direction == "en->ar" or candidate >= 3
        return Attempt(status="accepted" if ok else "rejected", record={"direction": direction} if ok else None)

    result = run_build("single", attempt_fn, range(10), n_target=2, max_candidates=10)
    assert sorted(r["direction"] for r in result.accepted) == ["ar->en", "en->ar"]
