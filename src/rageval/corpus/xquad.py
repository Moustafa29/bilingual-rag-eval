"""XQuAD (English + Arabic) as a retrieval corpus with human-written questions.

XQuAD's Arabic file is a professional translation of the English one: same articles, same
paragraph order, same question ids. Each paragraph becomes one passage; each question's gold
passage is the paragraph it was written on.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_xquad(en_path: Path, ar_path: Path) -> tuple[list[dict], list[dict]]:
    en = json.loads(Path(en_path).read_text(encoding="utf-8"))["data"]
    ar = json.loads(Path(ar_path).read_text(encoding="utf-8"))["data"]
    if len(en) != len(ar):
        raise ValueError("XQuAD en/ar article counts differ")
    passages, questions = [], []
    for a, (en_article, ar_article) in enumerate(zip(en, ar)):
        if len(en_article["paragraphs"]) != len(ar_article["paragraphs"]):
            raise ValueError(f"XQuAD article {a}: paragraph counts differ")
        for p, (en_par, ar_par) in enumerate(zip(en_article["paragraphs"], ar_article["paragraphs"])):
            chunk_id = f"xquad/{a:02d}/{p:02d}"
            passages.append(
                {"chunk_id": chunk_id, "doc_id": f"xquad/{a:02d}", "en": en_par["context"], "ar": ar_par["context"]}
            )
            if len(en_par["qas"]) != len(ar_par["qas"]):
                raise ValueError(f"XQuAD {chunk_id}: question counts differ")
            for en_q, ar_q in zip(en_par["qas"], ar_par["qas"]):
                if en_q["id"] != ar_q["id"]:
                    raise ValueError(f"XQuAD {chunk_id}: question ids differ ({en_q['id']} vs {ar_q['id']})")
                questions.append(
                    {
                        "qid": f"xquad-{en_q['id']}",
                        "source": "xquad",
                        "type": "single",
                        "direction": "en->ar",
                        "translation": "human",
                        "question": {"en": en_q["question"], "ar": ar_q["question"]},
                        "answer": {"en": en_q["answers"][0]["text"], "ar": ar_q["answers"][0]["text"]},
                        "gold_chunks": [chunk_id],
                        "relevant_groups": [[chunk_id]],
                    }
                )
    return passages, questions
