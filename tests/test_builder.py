import json
from pathlib import Path

from rageval.llm.client import ChatResult, DailyLimitReached
from rageval.questions.builder import Prompts, QuestionBuilder, run_build

PROMPTS = Prompts(Path(__file__).resolve().parents[1] / "prompts")

SINGLE = {
    "chunk_id": "1992/a/1#0003",
    "en": "The General Assembly adopted resolution 47/33 on 25 November 1992 concerning the International Law Commission.",
    "ar": "اعتمدت الجمعية العامة القرار ٤٧/٣٣ في ٢٥ تشرين الثاني/نوفمبر ١٩٩٢ بشأن لجنة القانون الدولي.",
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


def test_single_hop_accepted_with_verified_arabic_answer():
    verifier = FakeClient("verifier", lambda p: {"answerable": True, "answer": "٢٥ تشرين الثاني/نوفمبر ١٩٩٢"})
    generator = FakeClient("generator", single_generator)
    attempt = QuestionBuilder(generator, verifier, PROMPTS, 0.5).single(SINGLE, "en->ar")
    assert attempt.status == "accepted"
    record = attempt.record
    assert record["answer"] == {"en": "25 November 1992", "ar": "٢٥ تشرين الثاني/نوفمبر ١٩٩٢"}
    assert record["gold_chunks"] == ["1992/a/1#0003"]
    assert len(record["llm_calls"]) == 3
    # The translation prompt must not contain the passage (it would leak wording into the question).
    translate_prompt = next(p for p in generator.prompts if p.startswith("Translate"))
    assert "لجنة القانون الدولي." not in translate_prompt and SINGLE["en"] not in translate_prompt


def test_single_hop_rejects_hallucinated_evidence():
    def generator(prompt):
        out = single_generator(prompt)
        if "evidence" in out:
            out["evidence"] = "The Assembly adopted this on 1 January 1990."
        return out

    verifier = FakeClient("verifier", lambda p: {"answerable": True, "answer": "x"})
    attempt = QuestionBuilder(FakeClient("g", generator), verifier, PROMPTS, 0.5).single(SINGLE, "en->ar")
    assert attempt.status == "evidence_not_in_passage"
    assert attempt.record is None and len(attempt.calls) == 1


def test_single_hop_rejects_unanswerable_translation():
    verifier = FakeClient("verifier", lambda p: {"answerable": False, "answer": ""})
    attempt = QuestionBuilder(FakeClient("g", single_generator), verifier, PROMPTS, 0.5).single(SINGLE, "en->ar")
    assert attempt.status == "not_answerable_translated"


def bridge_generator(prompt):
    if prompt.startswith("Translate"):
        return {"question": "ما الذي أوصى به تقرير الأمين العام عن الاتجار غير المشروع بالأسلحة الصغيرة بأن تسمه الدول؟", "answer": "جميع الأسلحة الصغيرة"}
    return {
        "question": "What did the Secretary-General's report on illicit small arms trade recommend that States mark?",
        "answer": "all small arms",
        "evidence_a": CHUNK_A["en"],
        "evidence_b": CHUNK_B["en"],
    }


def test_bridge_accepted_when_neither_passage_alone_suffices():
    def verifier(prompt):
        if "Passage 2:" in prompt:
            return {"answerable": True, "answer": "جميع الأسلحة الصغيرة"}
        return {"answerable": False, "answer": ""}

    generator, ver = FakeClient("g", bridge_generator), FakeClient("v", verifier)
    attempt = QuestionBuilder(generator, ver, PROMPTS, 0.5).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "accepted"
    assert attempt.record["gold_chunks"] == [CHUNK_A["chunk_id"], CHUNK_B["chunk_id"]]
    assert len(attempt.calls) == 5  # generate, translate, shortcut B, shortcut A, verify both
    # Document headers make the citation link visible to the generator and every verifier call.
    assert "Passage B (Document A/60/88)" in generator.prompts[0]
    assert all("Document A/60/88" in p or "Document A/61/1" in p for p in ver.prompts)
    assert "Document A/61/1" in ver.prompts[-1] and "Document A/60/88" in ver.prompts[-1]


def test_bridge_rejected_when_passage_b_alone_answers():
    verifier = FakeClient("v", lambda p: {"answerable": True, "answer": "all small arms" if "Passage 2:" not in p else "جميع الأسلحة الصغيرة"})
    attempt = QuestionBuilder(FakeClient("g", bridge_generator), verifier, PROMPTS, 0.5).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "shortcut_b_alone"


def test_bridge_rejects_symbol_in_question():
    def generator(prompt):
        out = bridge_generator(prompt)
        if "evidence_a" in out:
            out["question"] = "What did report A/60/88 recommend that States mark?"
        return out

    attempt = QuestionBuilder(FakeClient("g", generator), FakeClient("v", lambda p: {}), PROMPTS, 0.5).bridge(CHUNK_A, CHUNK_B, "A/60/88", "en->ar")
    assert attempt.status == "document_number_in_question"


def test_run_build_balances_directions_and_stops_on_daily_limit():
    from rageval.questions.builder import Attempt

    seen = []

    def attempt_fn(candidate, direction):
        seen.append(direction)
        if candidate == "limit":
            raise DailyLimitReached("TPD")
        return Attempt(status="accepted", record={"direction": direction})

    result = run_build("single", attempt_fn, ["c1", "c2", "c3", "c4"], n_target=4, max_candidates=10)
    assert [r["direction"] for r in result.accepted] == ["en->ar", "ar->en", "en->ar", "ar->en"]
    assert result.stopped is None

    result = run_build("single", attempt_fn, ["c1", "limit", "c3"], n_target=3, max_candidates=10)
    assert len(result.accepted) == 1 and result.stopped.startswith("daily limit")


def test_replace_kinds_keeps_other_kinds():
    from rageval.questions.builder import replace_kinds

    previous = [{"type": "single", "qid": "s1"}, {"type": "bridge", "qid": "b1"}]
    new = [{"type": "bridge", "qid": "b2"}]
    assert replace_kinds(previous, new, {"bridge"}, "type") == [{"type": "single", "qid": "s1"}, {"type": "bridge", "qid": "b2"}]
    assert replace_kinds(previous, [], {"single", "bridge"}, "type") == []


def test_run_build_uses_other_direction_when_quota_full():
    from rageval.questions.builder import Attempt

    def attempt_fn(candidate, direction):
        ok = direction == "en->ar" or candidate >= 3
        return Attempt(status="accepted" if ok else "rejected", record={"direction": direction} if ok else None)

    result = run_build("single", attempt_fn, range(10), n_target=2, max_candidates=10)
    assert sorted(r["direction"] for r in result.accepted) == ["ar->en", "en->ar"]
