"""Export a seeded sample of questions for human audit, as a UTF-8 CSV that opens in Excel.

    python scripts/export_audit.py --config configs/pilot.yaml --n 100

Rubric (fill each column with 1 = yes, 0 = no):
  q_en_ok / q_ar_ok      question is fluent, unambiguous and grammatical in that language
  answer_ok              reference answers are correct for the gold passage(s), in both languages
  self_contained         question makes sense without seeing the passage
  needs_both (bridge)    neither passage alone answers it
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from rageval.io import load_config, read_jsonl

RUBRIC = ["q_en_ok", "q_ar_ok", "answer_ok", "self_contained", "needs_both", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data"])
    questions = read_jsonl(data / "questions" / "unpc_pilot.jsonl")
    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / "unpc" / "chunks.jsonl")}

    sample = sorted(random.Random(f"{cfg['seed']}:audit").sample(questions, min(args.n, len(questions))), key=lambda r: r["qid"])
    out = Path("audit") / "questions_audit.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "type", "direction", "question_en", "question_ar", "answer_en", "answer_ar", "passages_en", "passages_ar", *RUBRIC])
        for r in sample:
            gold = [chunks[cid] for cid in r["gold_chunks"]]
            writer.writerow(
                [
                    r["qid"],
                    r["type"],
                    r["direction"],
                    r["question"]["en"],
                    r["question"]["ar"],
                    r["answer"]["en"],
                    r["answer"]["ar"],
                    "\n\n---\n\n".join(c["en"] for c in gold),
                    "\n\n---\n\n".join(c["ar"] for c in gold),
                    *([""] * len(RUBRIC)),
                ]
            )
    print(f"wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()
