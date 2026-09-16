"""Score generated answers: four-cell attribution, and the answerer family-contamination comparison.

    python scripts/score_generation.py --corpus unpc --correctness exact_match
    python scripts/score_generation.py --corpus unpc --correctness judge --support --judge-dry-run
    python scripts/score_generation.py --corpus unpc --correctness judge --support

Correctness sources:
  exact_match   normalized answer tokens identical (free)
  judge         `generation.judge` decides. It is the same model as the question verifier (qwen3.8-27b);
                the human audit of judge decisions is the check on it. Run --judge-dry-run first for the
                number of uncached calls and an approximate token count.

--support also asks the judge whether a non-abstaining answer is supported by its passages (hallucination).

Reads data/generation/<corpus>/<answerer model>/<condition>/<lang>.jsonl. Reports, per answerer, condition
and language: accuracy, retrieval rate, the four attribution cells, abstention and hallucination rates.
It also reports:
- retrieval cost (oracle - rag) and generation cost (1 - oracle) per answerer and language
- the paired EN - AR accuracy difference with a 95% bootstrap CI
- contamination: for every condition and language answered by two models, each model's accuracy on
  the shared questions, the paired accuracy difference and the agreement rate (both right or both
  wrong)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from rageval.eval.answers import answer_scores
from rageval.eval.attribution import attribution_table, decomposition
from rageval.eval.corpora import QUESTION_FILES
from rageval.eval.stats import paired_bootstrap_ci
from rageval.io import load_config, load_dotenv, read_jsonl, write_jsonl
from rageval.llm.client import ChatClient, DailyLimitReached, DiskCache, cache_key
from rageval.questions.builder import LANG_NAMES, Prompts, format_passages
from rageval.questions.checks import is_true, parse_json_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), required=True)
    parser.add_argument("--correctness", choices=["exact_match", "judge"], default="exact_match")
    parser.add_argument("--support", action="store_true")
    parser.add_argument("--judge-dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    load_dotenv()
    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data"])
    questions = {q["qid"]: q for q in read_jsonl(data / "questions" / QUESTION_FILES[args.corpus])}
    prompts = Prompts(cfg["paths"]["prompts"])
    files = sorted(p for p in (data / "generation" / args.corpus).glob("*/*/*.jsonl") if not p.name.endswith(".scored.jsonl"))
    if not files:
        raise SystemExit("no generation outputs; run scripts/run_generation.py first")

    judge = None
    if args.correctness == "judge" or args.support:
        judge_cfg = (cfg.get("generation") or {}).get("judge")
        if not judge_cfg:
            raise SystemExit("set generation.judge in the config before judging")
        judge = ChatClient.from_config(judge_cfg, DiskCache(data / "llm_cache"), data / "llm_usage.jsonl")

    loaded = {p: read_jsonl(p) for p in files}
    contexts = {
        corpus: {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / corpus / "chunks.jsonl")}
        for corpus in {r["context_corpus"] for rows in loaded.values() for r in rows}
    }

    def judge_prompts(row: dict) -> tuple[str | None, str | None]:
        q, lang = questions[row["qid"]], row["lang"]
        correctness = support = None
        if args.correctness == "judge":
            correctness = prompts.render("judge_correctness", lang_name=LANG_NAMES[lang], question=q["question"][lang], reference=q["answer"][lang], candidate=row["answer"])
        if args.support and row["context_ids"] and not row["abstained"]:
            chunks = contexts[row["context_corpus"]]
            support = prompts.render("judge_support", question=q["question"][lang], answer=row["answer"], passages=format_passages([chunks[c][lang] for c in row["context_ids"]]))
        return correctness, support

    if args.judge_dry_run:
        if judge is None:
            raise SystemExit("--judge-dry-run needs --correctness judge and/or --support")
        # Deduplicate by cache key: two models that produce the same answer render the same judge prompt,
        # and the second one is a cache hit, not a call. Counting prompts instead of distinct keys
        # overestimated a contamination-subset run by 59 calls out of 240.
        pending: dict[str, str] = {}
        repeats = 0
        for rows in loaded.values():
            for r in rows:
                for prompt in judge_prompts(r):
                    if not prompt:
                        continue
                    key = cache_key(judge.base_url, judge.model, [{"role": "user", "content": prompt}], judge.params)
                    if judge.cache.get(key) is not None:
                        continue
                    if key in pending:
                        repeats += 1
                    else:
                        pending[key] = prompt
        chars = sum(len(p) for p in pending.values())
        print(json.dumps({"judge": judge.model, "uncached_judge_calls": len(pending), "repeated_prompts_not_counted": repeats, "prompt_characters": chars, "approx_prompt_tokens_at_3.5_chars": round(chars / 3.5)}, indent=2))
        return

    report: dict = {}
    scored: dict[tuple[str, str, str], dict[str, dict]] = {}
    try:
        for path, rows in loaded.items():
            model, condition, lang = path.parent.parent.name, path.parent.name, path.stem
            for r in rows:
                reference = questions[r["qid"]]["answer"][lang]
                r.update(answer_scores(r["answer"], reference, lang))
                correctness_prompt, support_prompt = judge_prompts(r)
                if correctness_prompt:
                    verdict = parse_json_object(judge.chat([{"role": "user", "content": correctness_prompt}]).text) or {}
                    r["correct"] = is_true(verdict.get("correct"))
                else:
                    r["correct"] = bool(r["exact_match"])
                if r["abstained"]:
                    r["correct"] = False
                if support_prompt:
                    verdict = parse_json_object(judge.chat([{"role": "user", "content": support_prompt}]).text) or {}
                    r["supported"] = is_true(verdict.get("supported"))
                else:
                    r.setdefault("supported", None)
            write_jsonl(path.with_suffix(".scored.jsonl"), rows)
            scored[(model, condition, lang)] = {r["qid"]: r for r in rows}
            report[f"{model}/{condition}/{lang}"] = attribution_table(rows)
    except DailyLimitReached as limit:
        print(f"stopped at daily limit: {limit}")
        sys.exit(3)

    for (model, condition, lang), rows in sorted(scored.items()):
        oracle = scored.get((model, "oracle", lang))
        if condition.startswith("rag_") and oracle:
            shared = sorted(set(rows) & set(oracle))
            report[f"{model}/{condition}/{lang}/decomposition"] = decomposition(
                sum(oracle[q]["correct"] for q in shared) / len(shared), sum(rows[q]["correct"] for q in shared) / len(shared)
            )
        if lang == "en" and (model, condition, "ar") in scored:
            ar = scored[(model, condition, "ar")]
            shared = sorted(set(rows) & set(ar))
            report[f"{model}/{condition}/en_minus_ar_accuracy"] = paired_bootstrap_ci(
                [float(rows[q]["correct"]) for q in shared], [float(ar[q]["correct"]) for q in shared], seed=args.seed
            )

    models = sorted({m for m, _, _ in scored})
    for a, b in combinations(models, 2):
        for (model, condition, lang), rows_a in sorted(scored.items()):
            if model != a or (b, condition, lang) not in scored:
                continue
            rows_b = scored[(b, condition, lang)]
            shared = sorted(set(rows_a) & set(rows_b))
            if not shared:
                continue
            ci = paired_bootstrap_ci([float(rows_a[q]["correct"]) for q in shared], [float(rows_b[q]["correct"]) for q in shared], seed=args.seed)
            report[f"contamination/{condition}/{lang}/{a}_vs_{b}"] = {
                "n": len(shared),
                f"accuracy_{a}": sum(rows_a[q]["correct"] for q in shared) / len(shared),
                f"accuracy_{b}": sum(rows_b[q]["correct"] for q in shared) / len(shared),
                "accuracy_difference": ci,
                "agreement_rate": sum(rows_a[q]["correct"] == rows_b[q]["correct"] for q in shared) / len(shared),
            }

    out = data / "results" / args.corpus
    out.mkdir(parents=True, exist_ok=True)
    (out / f"generation_{args.correctness}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
