"""Generate answers for every question and language under one or more context conditions.

    python scripts/run_generation.py --corpus unpc --conditions closed_book,oracle,rag:bm25-light --dry-run
    python scripts/run_generation.py --corpus unpc --conditions closed_book,oracle,rag:hybrid_bm25-light__bge-m3
    python scripts/run_generation.py --corpus unpc --conditions oracle --answerer contamination_answerer --sample 60
    python scripts/run_generation.py --corpus unpc --conditions rag:bm25-light --context-corpus unpc_uncorrected

Conditions:
  closed_book      no passages
  oracle           gold chunk(s) plus seeded distractors, k passages
  rag:<run>        top k of data/runs/<corpus>/<run>/<lang>-<lang>.jsonl

--answerer selects the config key under `generation` (answerer, or contamination_answerer for the
family-contamination check). --sample N answers a seeded random subset of N questions; the same seed
gives the same subset for every answerer, so their answers pair up question by question.

--context-corpus reads passage *text* from another corpus with identical chunk ids. Retrieval and gold
labels stay those of --corpus. Use `unpc_uncorrected` for the answer-scoring half of the digit-group
correction comparison.

--dry-run makes no LLM calls. It renders every answering prompt, and the judge prompts that scoring would
send, and reports call counts (and how many are not cached) and approximate tokens. Prompt tokens are
estimated at 3.5 characters per token. Output tokens are an assumption until real calls measure them
(the usage ledger records actual counts).

Writes data/generation/<corpus>/<answerer model>/<condition>[@<context-corpus>]/<lang>.jsonl. Every call is
cached; exit code 3 means the daily limit stopped the run (re-run to resume).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from rageval.eval.corpora import QUESTION_FILES
from rageval.generation.answering import answer_question
from rageval.generation.contexts import gold_positions, oracle_context, rag_context, retrieved_all
from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DailyLimitReached, DiskCache, cache_key
from rageval.questions.builder import LANG_NAMES, Prompts, format_passages

# Assumed output tokens per call, used only by --dry-run until the ledger has real numbers.
ASSUMED_OUTPUT_TOKENS = {"answer_reasoning": 150, "answer_plain": 20, "judge": 15}


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def model_dir(model: str) -> str:
    return model.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--answerer", default="answerer", choices=["answerer", "contamination_answerer"])
    parser.add_argument("--context-corpus", choices=sorted(QUESTION_FILES))
    parser.add_argument("--langs", default="en,ar")
    parser.add_argument("--sample", type=int, help="seeded random subset of N questions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    gen_cfg = cfg.get("generation") or {}
    if not gen_cfg.get(args.answerer):
        raise SystemExit(f"set generation.{args.answerer} in the config before generating")
    k, seed = gen_cfg["k"], cfg["seed"]
    data = Path(cfg["paths"]["data"])

    questions = read_jsonl(data / "questions" / QUESTION_FILES[args.corpus])
    if args.sample is not None:
        questions = sorted(random.Random(f"{seed}:generation-sample").sample(questions, min(args.sample, len(questions))), key=lambda q: q["qid"])
    context_corpus = args.context_corpus or args.corpus
    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / context_corpus / "chunks.jsonl")}
    pool = sorted(chunks)
    cache = DiskCache(data / "llm_cache")
    answerer = ChatClient.from_config(gen_cfg[args.answerer], cache, data / "llm_usage.jsonl")
    prompts = Prompts(cfg["paths"]["prompts"])
    reasoning = "reasoning_effort" in answerer.params and answerer.params.get("reasoning_effort") != "none"

    dry = {"answerer": answerer.model, "questions": len(questions), "by_condition": {}, "assumed_output_tokens_per_call": ASSUMED_OUTPUT_TOKENS}
    stopped = False
    for condition_spec in args.conditions.split(","):
        condition, _, run_name = condition_spec.partition(":")
        for lang in args.langs.split(","):
            ranking = {}
            if condition == "rag":
                ranking = {r["qid"]: [c for c, _ in r["ranking"]] for r in read_jsonl(data / "runs" / args.corpus / run_name / f"{lang}-{lang}.jsonl")}
            rows = []
            stats = {"answer_calls": 0, "answer_uncached": 0, "answer_prompt_tokens": 0, "judge_correctness_calls": 0, "judge_support_calls": 0, "judge_prompt_tokens": 0}
            for q in questions:
                if condition == "closed_book":
                    context = []
                elif condition == "oracle":
                    context = oracle_context(q["qid"], q["gold_chunks"], q["relevant_groups"], pool, k, seed)
                elif condition == "rag":
                    context = rag_context(ranking[q["qid"]], k)
                else:
                    raise SystemExit(f"unknown condition: {condition}")
                passages = [chunks[c][lang] for c in context]

                if args.dry_run:
                    if condition == "closed_book":
                        prompt = prompts.render("answer_closed_book", lang_name=LANG_NAMES[lang], question=q["question"][lang])
                    else:
                        prompt = prompts.render("answer_rag", lang_name=LANG_NAMES[lang], question=q["question"][lang], passages=format_passages(passages))
                    key = cache_key(answerer.base_url, answerer.model, [{"role": "user", "content": prompt}], answerer.params)
                    stats["answer_calls"] += 1
                    stats["answer_uncached"] += cache.get(key) is None
                    stats["answer_prompt_tokens"] += approx_tokens(prompt)
                    # The judge sees the reference answer as a stand-in for the unknown candidate answer.
                    reference = q["answer"][lang]
                    stats["judge_correctness_calls"] += 1
                    stats["judge_prompt_tokens"] += approx_tokens(prompts.render("judge_correctness", lang_name=LANG_NAMES[lang], question=q["question"][lang], reference=reference, candidate=reference))
                    if context:
                        stats["judge_support_calls"] += 1
                        stats["judge_prompt_tokens"] += approx_tokens(prompts.render("judge_support", question=q["question"][lang], answer=reference, passages=format_passages(passages)))
                    continue

                try:
                    out = answer_question(answerer, prompts, q["question"][lang], lang, condition, passages)
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
            if args.dry_run:
                stats["answer_output_tokens_assumed"] = stats["answer_calls"] * ASSUMED_OUTPUT_TOKENS["answer_reasoning" if reasoning else "answer_plain"]
                stats["judge_output_tokens_assumed"] = (stats["judge_correctness_calls"] + stats["judge_support_calls"]) * ASSUMED_OUTPUT_TOKENS["judge"]
                dry["by_condition"][f"{name}/{lang}"] = stats
                continue
            write_jsonl(data / "generation" / args.corpus / model_dir(answerer.model) / name / f"{lang}.jsonl", rows)
            print(f"{model_dir(answerer.model)}/{name} {lang}: {len(rows)} answers, abstained {sum(r['abstained'] for r in rows)}, retrieved {sum(r['retrieved'] for r in rows)}")
            if stopped:
                sys.exit(3)

    if args.dry_run:
        totals = {key: sum(s[key] for s in dry["by_condition"].values()) for key in next(iter(dry["by_condition"].values()))}
        dry["totals"] = totals
        dry["answerer_tokens_total"] = totals["answer_prompt_tokens"] + totals["answer_output_tokens_assumed"]
        dry["judge_tokens_total"] = totals["judge_prompt_tokens"] + totals["judge_output_tokens_assumed"]
        print(json.dumps(dry, indent=2))


if __name__ == "__main__":
    main()
