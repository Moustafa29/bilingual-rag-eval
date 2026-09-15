"""Export questions for human audit, as a UTF-8 CSV that opens in Excel.

    python scripts/export_audit.py --config configs/pilot.yaml

Rows:
  sample     a seeded random sample of accepted questions (`audit.n` in the config)
  flagged    accepted questions whose gold chunk is listed in `audit.flag_chunks`, always included
             with the note from the config
  filtered   every attempt the context-reference filter rejected, with the question text it had.
             Auditing these measures the filter's false rejections; auditing the sample measures
             what it still misses.

Rubric (fill each column with 1 = yes, 0 = no):
  q_en_ok / q_ar_ok      question is fluent, unambiguous and grammatical in that language
  answer_ok              reference answers are correct for the gold passage(s), in both languages
  self_contained         question makes sense without seeing the passage
  needs_both             (comparison) neither passage alone answers it
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from rageval.io import load_config, read_jsonl

RUBRIC = ["q_en_ok", "q_ar_ok", "answer_ok", "self_contained", "needs_both", "notes"]
FILTER_STATUSES = {"context_reference", "context_reference_translated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    audit = cfg["audit"]
    data = Path(cfg["paths"]["data"])
    questions = read_jsonl(data / "questions" / "unpc_pilot.jsonl")
    attempts = read_jsonl(data / "questions" / "unpc_pilot_attempts.jsonl")
    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / "unpc" / "chunks.jsonl")}
    flags = audit.get("flag_chunks") or {}

    def flag_for(chunk_ids: list[str]) -> str:
        return "; ".join(flags[c] for c in chunk_ids if c in flags)

    sample = random.Random(f"{cfg['seed']}:audit").sample(questions, min(audit["n"], len(questions)))
    rows = []
    seen = set()
    for group, records in (("sample", sample), ("flagged", [r for r in questions if flag_for(r["gold_chunks"])])):
        for r in sorted(records, key=lambda r: r["qid"]):
            if r["qid"] in seen:
                continue
            seen.add(r["qid"])
            rows.append((group, r["qid"], r["kind"], "accepted", r["direction"], r["question"], r.get("options", {}), r["answer"], r["gold_chunks"]))
    for o in attempts:
        if o["status"] in FILTER_STATUSES:
            rows.append(("filtered", f"attempt-{o['kind']}-{o['candidate']}", o["kind"], o["status"], o["direction"], o.get("question", {}), {}, {}, o["chunks"]))

    out = Path("audit") / "questions_audit.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group", "id", "kind", "status", "flag", "direction", "question_en", "question_ar", "options_en", "options_ar",
             "answer_en", "answer_ar", "passages_en", "passages_ar", *RUBRIC]
        )
        for group, rid, kind, status, direction, question, options, answer, chunk_ids in rows:
            gold = [chunks[c] for c in chunk_ids if c in chunks]
            writer.writerow(
                [
                    group, rid, kind, status, flag_for(chunk_ids), direction,
                    question.get("en", ""), question.get("ar", ""),
                    " | ".join(options.get("en", [])), " | ".join(options.get("ar", [])),
                    answer.get("en", ""), answer.get("ar", ""),
                    "\n\n---\n\n".join(c["en"] for c in gold), "\n\n---\n\n".join(c["ar"] for c in gold),
                    *([""] * len(RUBRIC)),
                ]
            )
    counts = {g: sum(r[0] == g for r in rows) for g in ("sample", "flagged", "filtered")}
    print(f"wrote {len(rows)} rows to {out}: {counts}")


if __name__ == "__main__":
    main()
