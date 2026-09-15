"""Re-verify recorded single-hop and numeric attempts with the current verifier, candidate by candidate.

The verifier model the question set was built with, qwen/qwen3.6-27b, was withdrawn by Groq with no
deprecation notice. This replays every recorded attempt with the same chunk, the same translation
direction and the same cached generator responses, changing only the verifier model, and compares the
old status with the new one.

The normal build is not used for this: it balances translation directions by the number of questions
accepted so far, so a single flipped verdict changes the direction of later candidates, and those would
no longer be the same candidates. Here every attempt keeps its recorded direction.

Rejections that happen before any verifier call must come out identical; the summary checks that.

    python scripts/reverify.py --old-attempts data/questions/verifier_qwen3.6/unpc_pilot_attempts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DailyLimitReached, DiskCache
from rageval.questions.builder import Prompts, QuestionBuilder

KINDS = ("single", "numeric")
# Statuses decided before the verifier is ever called; they cannot depend on the verifier model.
PRE_VERIFIER = {
    "bad_json_generation",
    "generator_skipped",
    "missing_fields",
    "evidence_not_in_passage",
    "answer_not_in_passage",
    "context_reference",
    "answer_not_a_thousands_number",
    "translation_missing_fields",
    "context_reference_translated",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--old-attempts", required=True)
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    q = cfg["questions"]
    data = Path(cfg["paths"]["data"])

    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / "unpc" / "chunks.jsonl")}
    cache = DiskCache(data / "llm_cache")
    ledger = data / "llm_usage.jsonl"
    generator = ChatClient.from_config(q["generator"], cache, ledger)
    verifier = ChatClient.from_config(q["verifier"], cache, ledger)
    builder = QuestionBuilder(generator, verifier, Prompts(cfg["paths"]["prompts"]))

    old = [o for o in read_jsonl(args.old_attempts) if o["kind"] in KINDS]
    rows, stopped = [], None
    for o in old:
        chunk = chunks[o["chunks"][0]]
        try:
            if o["kind"] == "numeric":
                attempt = builder.single(chunk, o["direction"], kind="numeric", prompt="generate_numeric", require_number=True)
            else:
                attempt = builder.single(chunk, o["direction"])
        except DailyLimitReached as limit:
            stopped = f"daily limit: {str(limit)[:200]}"
            break
        rows.append({"kind": o["kind"], "candidate": o["candidate"], "chunks": o["chunks"], "direction": o["direction"], "old_status": o["status"], "new_status": attempt.status})

    summary = {"old_verifier": "qwen/qwen3.6-27b", "new_verifier": verifier.model, "stopped": stopped, "by_kind": {}}
    for kind in KINDS:
        kr = [r for r in rows if r["kind"] == kind]
        reached = [r for r in kr if r["old_status"] not in PRE_VERIFIER]
        summary["by_kind"][kind] = {
            "attempts_replayed": len(kr),
            "pre_verifier_rejections": len(kr) - len(reached),
            "pre_verifier_status_changed": sum(r["old_status"] != r["new_status"] for r in kr if r["old_status"] in PRE_VERIFIER),
            "reached_verifier": len(reached),
            "verdict_unchanged": sum(r["old_status"] == r["new_status"] for r in reached),
            "accept_to_reject": sum(r["old_status"] == "accepted" and r["new_status"] != "accepted" for r in reached),
            "reject_to_accept": sum(r["old_status"] != "accepted" and r["new_status"] == "accepted" for r in reached),
            "reject_to_reject_other_reason": sum(
                r["old_status"] != "accepted" and r["new_status"] not in ("accepted", r["old_status"]) for r in reached
            ),
            "accepted_old": sum(r["old_status"] == "accepted" for r in kr),
            "accepted_new": sum(r["new_status"] == "accepted" for r in kr),
            "transitions": {f"{a} -> {b}": n for (a, b), n in Counter((r["old_status"], r["new_status"]) for r in reached if r["old_status"] != r["new_status"]).most_common()},
        }

    out = data / "questions"
    write_jsonl(out / "reverify_rows.jsonl", rows)
    (out / "reverify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if stopped:
        sys.exit(3)


if __name__ == "__main__":
    main()
