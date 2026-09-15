import json

import pytest

from rageval.corpus.xquad import load_xquad


def squad(questions, context="ctx"):
    return {"data": [{"title": "t", "paragraphs": [{"context": context, "qas": [{"id": qid, "question": q, "answers": [{"text": a, "answer_start": 0}]} for qid, q, a in questions]}]}]}


def write(tmp_path, en, ar):
    (tmp_path / "en.json").write_text(json.dumps(en), encoding="utf-8")
    (tmp_path / "ar.json").write_text(json.dumps(ar, ensure_ascii=False), encoding="utf-8")
    return tmp_path / "en.json", tmp_path / "ar.json"


def test_load_pairs_passages_and_questions(tmp_path):
    en = squad([("q1", "Who?", "Denver")], context="Denver won.")
    ar = squad([("q1", "من؟", "دنفر")], context="فاز دنفر.")
    passages, questions = load_xquad(*write(tmp_path, en, ar))
    assert passages == [{"chunk_id": "xquad/00/00", "doc_id": "xquad/00", "en": "Denver won.", "ar": "فاز دنفر."}]
    assert questions[0]["qid"] == "xquad-q1"
    assert questions[0]["question"] == {"en": "Who?", "ar": "من؟"}
    assert questions[0]["relevant_groups"] == [["xquad/00/00"]]


def test_mismatched_question_ids_raise(tmp_path):
    with pytest.raises(ValueError):
        load_xquad(*write(tmp_path, squad([("q1", "Who?", "x")]), squad([("q2", "من؟", "x")])))
