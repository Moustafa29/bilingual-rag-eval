"""Score generated answers and report four-cell attribution per condition and language.

    python scripts/score_generation.py --corpus unpc --correctness exact_match
    python scripts/score_generation.py --corpus unpc --correctness judge --judge-dry-run
    python scripts/score_generation.py --corpus unpc --correctness judge --support

Correctness sources:
  exact_match   normalized answer tokens identical (free)
  judge         `generation.judge` from the config decides (paid; run --judge-dry-run first for the
                number of uncached calls and an approximate token count)

--support also asks the judge whether a non-abstaining answer is supported by its passages, which gives
the hallucination rate.

Reports, per condition and language: accuracy, retrieval rate, the four attribution cells, abstention
and hallucination rates. When oracle and rag conditions are present: retrieval cost (oracle - rag) and
generation cost (1 - oracle). When a condition was run in both languages: the paired EN - AR accuracy
difference with a 95% bootstrap CI. Scored rows go to data/generation/<corpus>/<condition>/<lang>.scored.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    files = sorted(p for p in (data / "generation" / args.corpus).glob("*/*.jsonl") if not p.name.endswith(".scored.jsonl"))
    if not files:
        raise SystemExit("no generation outputs; run scripts/run_generation.py first")

    judge = None
    if args.correctness == "judge" or args.support:
        judge_cfg = (cfg.get("generation") or {}).get("judge")
        if not judge_cfg:
            raise SystemExit("set generation.judge in the config before judging")
        judge = ChatClient.from_config(judge_cfg, DiskCache(data / "llm_cache"), data / "llm_usage.jsonl")

    def judge_prompts(row: dict) -> list[str]:
        q, lang = questions[row["qid"]], row["lang"]
        out = []
        if args.correctness == "judge":
            out.append(prompts.render("judge_correctness", lang_name=LANG_NAMES[lang], question=q["question"][lang], reference=q["answer"][lang], candidate=row["answer"]))
        if args.support and row["context_ids"] and not row["abstained"]:
            chunks = contexts[row["context_corpus"]]
            out.append(prompts.render("judge_support", question=q["question"][lang], answer=row["answer"], passages=format_passages([chunks[c][lang] for c in row["context_ids"]])))
        return out

    contexts: dict[str, dict] = {}
    loaded = {}
    for path in files:
        rows = read_jsonl(path)
        loaded[path] = rows
        for r in rows:
            if r["context_corpus"] not in contexts:
                contexts[r["context_corpus"]] = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / r["context_corpus"] / "chunks.jsonl")}

    if args.judge_dry_run:
        if judge is None:
            raise SystemExit("--judge-dry-run needs --correctness judge and/or --support")
        pending = [p for rows in loaded.values() for r in rows for p in judge_prompts(r) if judge.cache.get(cache_key(judge.base_url, judge.model, [{"role": "user", "content": p}], judge.params)) is None]
        chars = sum(len(p) for p in pending)
        print(json.dumps({"uncached_judge_calls": len(pending), "prompt_characters": chars, "approx_prompt_tokens_at_3.5_chars": round(chars / 3.5)}, indent=2))
        return

    report: dict = {}
    by_condition: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    try:
        for path, rows in loaded.items():
            condition, lang = path.parent.name, path.stem
            for r in rows:
                q = r_q = questions[r["qid"]]
                r.update(answer_scores(r["answer"], r_q["answer"][lang], lang))
                prompts_for_row = judge_prompts(r)
                if args.correctness == "judge":
                    verdict = parse_json_object(judge.chat([{"role": "user", "content": prompts_for_row.pop(0)}]).text) or {}
                    r["correct"] = is_true(verdict.get("correct"))
                else:
                    r["correct"] = bool(r["exact_match"])
                if r["abstained"]:
                    r["correct"] = False
                if prompts_for_row:
                    verdict = parse_json_object(judge.chat([{"role": "user", "content": prompts_for_row[0]}]).text) or {}
                    r["supported"] = is_true(verdict.get("supported"))
                else:
                    r.setdefault("supported", None)
            write_jsonl(path.with_suffix(".scored.jsonl"), rows)
            by_condition[condition][lang] = rows
            report[f"{condition}/{lang}"] = attribution_table(rows)
    except DailyLimitReached as limit:
        print(f"stopped at daily limit: {limit}")
        sys.exit(3)

    for condition, langs in sorted(by_condition.items()):
        if "en" in langs and "ar" in langs:
            en = {r["qid"]: r["correct"] for r in langs["en"]}
            ar = {r["qid"]: r["correct"] for r in langs["ar"]}
            shared = sorted(set(en) & set(ar))
            report[f"{condition}/en_minus_ar_accuracy"] = paired_bootstrap_ci([float(en[q]) for q in shared], [float(ar[q]) for q in shared], seed=args.seed)
    for lang in ("en", "ar"):
        oracle = report.get(f"oracle/{lang}")
        for condition in by_condition:
            if condition.startswith("rag_") and oracle and f"{condition}/{lang}" in report:
                report[f"{condition}/{lang}/decomposition"] = decomposition(oracle["accuracy"], report[f"{condition}/{lang}"]["accuracy"])

    out = data / "results" / args.corpus
    out.mkdir(parents=True, exist_ok=True)
    (out / f"generation_{args.correctness}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
