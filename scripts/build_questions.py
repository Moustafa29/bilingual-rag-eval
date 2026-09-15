"""Generate the UN pilot question set with Groq (free tier) and write it with its rejection funnel.

    python scripts/build_questions.py --config configs/pilot.yaml --dry-run
    python scripts/build_questions.py --config configs/pilot.yaml

Every LLM response is cached, so re-running after a daily-limit stop only spends quota on
candidates not yet tried. Exit code 3 means the run stopped early and should be re-run later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from functools import partial
from itertools import islice
from pathlib import Path

from rageval.corpus.symbols import doc_key
from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DiskCache
from rageval.questions.builder import BuildResult, Prompts, QuestionBuilder, replace_kinds, run_build
from rageval.questions.dedup import near_duplicate_groups
from rageval.questions.sampling import bridge_candidates, group_by_doc, single_candidates


def prompts_version(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.txt")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count candidates and prompt sizes; no API calls")
    parser.add_argument("--kind", choices=["single", "bridge", "all"], default="all")
    parser.add_argument("--max-candidates", type=int, help="override both candidate caps, e.g. 20 for a smoke run")
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    q = cfg["questions"]
    if args.max_candidates is not None:
        q["max_candidates_single"] = q["max_candidates_bridge"] = args.max_candidates
    data = Path(cfg["paths"]["data"])
    seed = cfg["seed"]

    chunks = read_jsonl(data / "corpus" / "unpc" / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    by_doc = group_by_doc(chunks)
    key_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(by_doc):
        key_to_docs[doc_key(doc_id)].append(doc_id)
    rules = q["content_filter"]

    def singles():
        return single_candidates(by_doc, rules, seed)

    def bridges():
        return bridge_candidates(by_doc, key_to_docs, rules, seed)

    if args.dry_run:
        single_list = list(islice(singles(), q["max_candidates_single"]))
        bridge_list = list(islice(bridges(), q["max_candidates_bridge"]))
        chars = [len(c["en"]) + len(c["ar"]) for c in single_list]
        print(f"single-hop candidates available (capped at {q['max_candidates_single']}): {len(single_list)}")
        print(f"bridge candidates available (capped at {q['max_candidates_bridge']}): {len(bridge_list)}")
        if chars:
            print(f"mean en+ar characters per single-hop candidate chunk: {sum(chars) / len(chars):.0f}")
        return

    prompts_dir = Path(cfg["paths"]["prompts"])
    cache = DiskCache(data / "llm_cache")
    ledger = data / "llm_usage.jsonl"
    generator = ChatClient.from_config(q["generator"], cache, ledger)
    verifier = ChatClient.from_config(q["verifier"], cache, ledger)
    builder = QuestionBuilder(generator, verifier, Prompts(prompts_dir), q["answer_f1_threshold"])

    ran: dict[str, BuildResult] = {}
    if args.kind in ("single", "all"):
        ran["single"] = run_build("single", builder.single, singles(), q["n_single"], q["max_candidates_single"])
    daily_limit = any(r.stopped and r.stopped.startswith("daily limit") for r in ran.values())
    if args.kind in ("bridge", "all") and not daily_limit:
        ran["bridge"] = run_build(
            "bridge", lambda cand, d: builder.bridge(*cand, d), bridges(), q["n_bridge"], q["max_candidates_bridge"]
        )

    new_records = [r for result in ran.values() for r in result.accepted]
    gold = {cid for r in new_records for cid in r["gold_chunks"]}
    groups = near_duplicate_groups(gold, by_id, q["dedup_jaccard"], q["shingle_size"])
    version = prompts_version(prompts_dir)
    counters = Counter()
    for r in new_records:
        counters[r["type"]] += 1
        r["qid"] = f"unpc-{r['type']}-{counters[r['type']]:04d}"
        r["relevant_groups"] = [groups[cid] for cid in r["gold_chunks"]]
        r["prompts_version"] = version
        r["models"] = {"generator": generator.model, "verifier": verifier.model}
    new_records = [{"qid": r.pop("qid"), **r} for r in new_records]

    # A run of one kind must not erase the other kind's results: the first smoke runs did exactly
    # that, when the bridge run overwrote the 14 accepted single-hop questions.
    out = data / "questions"
    questions_path = out / "unpc_pilot.jsonl"
    attempts_path = out / "unpc_pilot_attempts.jsonl"
    funnel_path = out / "unpc_pilot_funnel.json"
    run_kinds = set(ran)
    records = replace_kinds(read_jsonl(questions_path) if questions_path.exists() else [], new_records, run_kinds, "type")
    outcomes = replace_kinds(
        read_jsonl(attempts_path) if attempts_path.exists() else [],
        [o for result in ran.values() for o in result.outcomes],
        run_kinds,
        "kind",
    )
    write_jsonl(questions_path, records)
    write_jsonl(attempts_path, outcomes)

    previous = json.loads(funnel_path.read_text(encoding="utf-8")).get("funnel", {}) if funnel_path.exists() else {}
    funnel = {}
    for kind in ("single", "bridge"):
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
        "questions": dict(Counter(r["type"] for r in records)),
        "relevance_group_sizes": dict(Counter(len(g) for r in records for g in r["relevant_groups"])),
    }
    funnel_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(r.stopped and r.stopped.startswith("daily limit") for r in ran.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
