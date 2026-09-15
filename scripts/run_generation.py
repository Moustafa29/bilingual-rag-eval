"""Generate answers for every question and language under one or more context conditions.

    python scripts/run_generation.py --corpus unpc --conditions closed_book,oracle,rag:hybrid_bm25-light__bge-m3
    python scripts/run_generation.py --corpus unpc --conditions rag:bm25-light --context-corpus unpc_uncorrected

Conditions:
  closed_book      no passages
  oracle           gold chunk(s) plus seeded distractors, k passages
  rag:<run>        top k of data/runs/<corpus>/<run>/<lang>-<lang>.jsonl

--context-corpus reads passage *text* from another corpus with identical chunk ids. Retrieval and the
gold labels stay those of --corpus. Use `unpc_uncorrected` to answer from the Arabic digit groups as
distributed, for the corrected-vs-uncorrected answer-scoring comparison.

Writes data/generation/<corpus>/<condition>[@<context-corpus>]/<lang>.jsonl with, per question: the
context ids, where each gold group sits in the context, whether all gold groups were retrieved, and the
answer. Every LLM call is cached; exit code 3 means the daily limit stopped the run (re-run to resume).

The answering model comes from `generation.answerer` in the config, which must be set explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rageval.eval.corpora import QUESTION_FILES
from rageval.generation.answering import answer_question
from rageval.generation.contexts import gold_positions, oracle_context, rag_context, retrieved_all
from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DailyLimitReached, DiskCache
from rageval.questions.builder import Prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--context-corpus", choices=sorted(QUESTION_FILES))
    parser.add_argument("--langs", default="en,ar")
    parser.add_argument("--limit", type=int, help="first N questions only, for a cost check")
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    gen_cfg = cfg.get("generation") or {}
    if not gen_cfg.get("answerer"):
        raise SystemExit("set generation.answerer in the config before generating (see docs/design.md, Phase 4)")
    k, seed = gen_cfg["k"], cfg["seed"]
    data = Path(cfg["paths"]["data"])

    questions = read_jsonl(data / "questions" / QUESTION_FILES[args.corpus])[: args.limit]
    context_corpus = args.context_corpus or args.corpus
    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / context_corpus / "chunks.jsonl")}
    pool = sorted(chunks)
    cache = DiskCache(data / "llm_cache")
    answerer = ChatClient.from_config(gen_cfg["answerer"], cache, data / "llm_usage.jsonl")
    prompts = Prompts(cfg["paths"]["prompts"])

    stopped = False
    for condition_spec in args.conditions.split(","):
        condition, _, run_name = condition_spec.partition(":")
        for lang in args.langs.split(","):
            ranking = {}
            if condition == "rag":
                ranking = {r["qid"]: [c for c, _ in r["ranking"]] for r in read_jsonl(data / "runs" / args.corpus / run_name / f"{lang}-{lang}.jsonl")}
            rows = []
            for q in questions:
                if condition == "closed_book":
                    context = []
                elif condition == "oracle":
                    context = oracle_context(q["qid"], q["gold_chunks"], q["relevant_groups"], pool, k, seed)
                elif condition == "rag":
                    context = rag_context(ranking[q["qid"]], k)
                else:
                    raise SystemExit(f"unknown condition: {condition}")
                try:
                    out = answer_question(answerer, prompts, q["question"][lang], lang, condition, [chunks[c][lang] for c in context])
                except DailyLimitReached as limit:
                    print(f"stopped: {limit}")
                    stopped = True
                    break
                rows.append(
                    {
                        "qid": q["qid"],
                        "kind": q.get("kind", q["type"]),
                        "lang": lang,
                        "condition": condition_spec,
                        "context_corpus": context_corpus,
                        "context_ids": context,
                        "gold_positions": gold_positions(context, q["relevant_groups"]) if context else [None] * len(q["relevant_groups"]),
                        "retrieved": retrieved_all(context, q["relevant_groups"]) if context else False,
                        "answer": out["answer"],
                        "abstained": out["abstained"],
                        "model": answerer.model,
                        "cache_key": out["cache_key"],
                    }
                )
            name = condition_spec.replace(":", "_") + (f"@{args.context_corpus}" if args.context_corpus else "")
            write_jsonl(data / "generation" / args.corpus / name / f"{lang}.jsonl", rows)
            print(f"{name} {lang}: {len(rows)} answers, abstained {sum(r['abstained'] for r in rows)}, retrieved {sum(r['retrieved'] for r in rows)}")
            if stopped:
                sys.exit(3)


if __name__ == "__main__":
    main()
