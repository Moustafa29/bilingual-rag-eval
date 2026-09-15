"""Generate the UN pilot question set with Groq (free tier) and write it with its rejection funnel.

    python scripts/build_questions.py --dry-run
    python scripts/build_questions.py --kind comparison --max-candidates 20
    python scripts/build_questions.py                  # all: single, numeric, comparison

Kinds:
  single      single-hop questions from random content chunks
  numeric     single-hop questions whose answer is a thousands-separated number, from chunks where
              the Arabic digit groups were corrected (for the corrected-vs-uncorrected comparison)
  comparison  two-passage questions over documents that share a subject term
  bridge      two-passage questions via document citations; kept for the documented negative
              result, not part of "all"

Every LLM response is cached, so re-running after a daily-limit stop only spends quota on candidates
not yet tried. Exit code 3 means the run stopped early and should be re-run later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from rageval.corpus.symbols import doc_key
from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DiskCache
from rageval.questions.builder import BuildResult, Prompts, QuestionBuilder, replace_kinds, run_build
from rageval.questions.dedup import near_duplicate_groups
from rageval.questions.sampling import (
    bridge_candidates,
    comparison_candidates,
    group_by_doc,
    numeric_candidates,
    single_candidates,
)

KINDS = ("single", "numeric", "comparison", "bridge")
ALL = ("single", "numeric", "comparison")


def prompts_version(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.txt")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count candidates; no API calls")
    parser.add_argument("--kind", choices=[*KINDS, "all"], default="all")
    parser.add_argument("--max-candidates", type=int, help="override every kind's candidate cap, e.g. 20 for a test")
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    q = cfg["questions"]
    if q.get("frozen") and not args.dry_run:
        raise SystemExit(f"the question set is frozen ({q['frozen']}); see questions.frozen in the config")
    data = Path(cfg["paths"]["data"])
    seed = cfg["seed"]
    if args.max_candidates is not None:
        for kind in KINDS:
            q[f"max_candidates_{kind}"] = args.max_candidates

    chunks = read_jsonl(data / "corpus" / "unpc" / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    by_doc = group_by_doc(chunks)
    key_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(by_doc):
        key_to_docs[doc_key(doc_id)].append(doc_id)
    rules = q["content_filter"]
    candidates = {
        "single": lambda: single_candidates(by_doc, rules, seed),
        "numeric": lambda: numeric_candidates(by_doc, rules, seed),
        "comparison": lambda: comparison_candidates(by_doc, rules, seed, q["comparison_max_keyword_docs"]),
        "bridge": lambda: bridge_candidates(by_doc, key_to_docs, rules, seed),
    }
    selected = ALL if args.kind == "all" else (args.kind,)

    if args.dry_run:
        for kind in selected:
            cap = q[f"max_candidates_{kind}"]
            print(f"{kind}: {len(list(islice(candidates[kind](), cap)))} candidates available (cap {cap}), target {q[f'n_{kind}']}")
        return

    prompts_dir = Path(cfg["paths"]["prompts"])
    cache = DiskCache(data / "llm_cache")
    ledger = data / "llm_usage.jsonl"
    generator = ChatClient.from_config(q["generator"], cache, ledger)
    verifier = ChatClient.from_config(q["verifier"], cache, ledger)
    builder = QuestionBuilder(generator, verifier, Prompts(prompts_dir))
    attempt_fns = {
        "single": lambda cand, d: builder.single(cand, d),
        "numeric": lambda cand, d: builder.single(cand, d, kind="numeric", prompt="generate_numeric", require_number=True),
        "comparison": lambda cand, d: builder.comparison(*cand, d),
        "bridge": lambda cand, d: builder.bridge(*cand, d),
    }

    ran: dict[str, BuildResult] = {}
    for kind in selected:
        ran[kind] = run_build(kind, attempt_fns[kind], candidates[kind](), q[f"n_{kind}"], q[f"max_candidates_{kind}"])
        if ran[kind].stopped and ran[kind].stopped.startswith("daily limit"):
            break

    new_records = [r for result in ran.values() for r in result.accepted]
    gold = {cid for r in new_records for cid in r["gold_chunks"]}
    groups = near_duplicate_groups(gold, by_id, q["dedup_jaccard"], q["shingle_size"])
    version = prompts_version(prompts_dir)
    counters = Counter()
    for r in new_records:
        counters[r["kind"]] += 1
        r["qid"] = f"unpc-{r['kind']}-{counters[r['kind']]:04d}"
        r["relevant_groups"] = [groups[cid] for cid in r["gold_chunks"]]
        r["prompts_version"] = version
        r["models"] = {"generator": generator.model, "verifier": verifier.model}
    new_records = [{"qid": r.pop("qid"), **r} for r in new_records]

    # A run of some kinds must not erase the others' results.
    out = data / "questions"
    questions_path = out / "unpc_pilot.jsonl"
    attempts_path = out / "unpc_pilot_attempts.jsonl"
    funnel_path = out / "unpc_pilot_funnel.json"
    previous_records = read_jsonl(questions_path) if questions_path.exists() else []
    for r in previous_records:
        r.setdefault("kind", r["type"])  # records written before kinds were separate from types
    records = replace_kinds(previous_records, new_records, set(ran), "kind")
    outcomes = replace_kinds(
        read_jsonl(attempts_path) if attempts_path.exists() else [],
        [o for result in ran.values() for o in result.outcomes],
        set(ran),
        "kind",
    )
    write_jsonl(questions_path, records)
    write_jsonl(attempts_path, outcomes)

    previous = json.loads(funnel_path.read_text(encoding="utf-8")).get("funnel", {}) if funnel_path.exists() else {}
    funnel = {}
    for kind in KINDS:
        if kind not in ran:
            if kind in previous:
                funnel[kind] = previous[kind]
            continue
        result = ran[kind]
        funnel[kind] = {
            "attempted": len(result.outcomes),
            "by_status": dict(Counter(o["status"] for o in result.outcomes).most_common()),
            "accepted_by_direction": dict(Counter(r["direction"] for r in result.accepted)),
            "stopped": result.stopped,
            "network_tokens": dict(result.usage),
            "prompts_version": version,
        }
    summary = {
        "funnel": funnel,
        "questions": dict(Counter(r["kind"] for r in records)),
        "relevance_group_sizes": dict(Counter(len(g) for r in records for g in r["relevant_groups"])),
    }
    funnel_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(r.stopped and r.stopped.startswith("daily limit") for r in ran.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
