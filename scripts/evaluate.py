"""Score every run of a corpus and print the retrieval table with paired comparisons.

    python scripts/evaluate.py --corpus xquad
    python scripts/evaluate.py --corpus unpc --subset numeric --baseline-corpus unpc_uncorrected

For each config and language pair: recall@1/5/10/20, all_recall@5 (differs from recall@5 only for
multi-group questions), MRR@10 and nDCG@10.

Paired comparisons (95% bootstrap CI over questions, exact McNemar on hit@5):
- EN - AR, for every config run in both en-en and ar-ar.
- With --baseline-corpus: this corpus - baseline, per config and language pair, on the same
  questions. This measures what correcting the Arabic digit-group reversal is worth. en-en rows must
  show exactly zero difference, because the correction does not touch English text.

--subset numeric keeps questions whose English question or answer contains a number with thousands
separators, the numbers the reversal affects. Results go to data/results/<corpus>/retrieval.json,
or retrieval_<subset>.json for a subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rageval.corpus.numbers import is_numeric_question
from rageval.eval.corpora import QUESTION_FILES
from rageval.eval.metrics import question_metrics
from rageval.eval.stats import mcnemar_exact, paired_bootstrap_ci
from rageval.io import read_jsonl

COLUMNS = ["recall@1", "recall@5", "recall@10", "recall@20", "all_recall@5", "mrr@10", "ndcg@10"]
PAIRED = ["recall@5", "mrr@10", "ndcg@10"]


def score_runs(data: Path, corpus: str, questions: dict[str, dict]) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    scored = {}
    for run_file in sorted((data / "runs" / corpus).glob("*/*.jsonl")):
        rows = {}
        for record in read_jsonl(run_file):
            if record["qid"] in questions:
                ranking = [c for c, _ in record["ranking"]]
                rows[record["qid"]] = question_metrics(ranking, questions[record["qid"]]["relevant_groups"])
        missing = set(questions) - set(rows)
        if missing:
            raise SystemExit(f"{run_file}: {len(missing)} questions have no ranking")
        scored[(run_file.parent.name, run_file.stem)] = rows
    return scored


def paired(a: dict, b: dict, qids: list[str], seed: int) -> dict:
    result = {m: paired_bootstrap_ci([a[q][m] for q in qids], [b[q][m] for q in qids], seed=seed) for m in PAIRED}
    result["mcnemar_hit@5"] = mcnemar_exact([a[q]["recall@5"] == 1.0 for q in qids], [b[q]["recall@5"] == 1.0 for q in qids])
    return result


def print_paired(title: str, first: str, second: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}\n")
    print("| config | " + " | ".join(f"Δ {m} [CI]" for m in PAIRED) + f" | hit@5 {first}-only / {second}-only | McNemar p |")
    print("|---|" + "---|" * (len(PAIRED) + 2))
    for label, r in rows:
        cells = [f"{r[m]['mean_diff']:+.3f} [{r[m]['ci_low']:+.3f}, {r[m]['ci_high']:+.3f}]" for m in PAIRED]
        test = r["mcnemar_hit@5"]
        print(f"| {label} | " + " | ".join(cells) + f" | {test['a_only']} / {test['b_only']} | {test['p_value']:.3g} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--subset", choices=["all", "numeric"], default="all")
    parser.add_argument("--baseline-corpus", choices=sorted(QUESTION_FILES))
    args = parser.parse_args()
    data = Path(args.data)

    questions = {q["qid"]: q for q in read_jsonl(data / "questions" / QUESTION_FILES[args.corpus])}
    if args.subset == "numeric":
        questions = {qid: q for qid, q in questions.items() if is_numeric_question(q)}
        if not questions:
            raise SystemExit("no questions in the numeric subset")
    per_question = score_runs(data, args.corpus, questions)
    qids = sorted(questions)

    table = {}
    print(f"\n{args.corpus} ({args.subset}): {len(qids)} questions\n")
    print("| config | query-doc | " + " | ".join(COLUMNS) + " |")
    print("|---|---|" + "---|" * len(COLUMNS))
    for (config, pair), rows in sorted(per_question.items()):
        means = {m: float(np.mean([rows[q][m] for q in qids])) for m in COLUMNS}
        table[f"{config}/{pair}"] = means
        print(f"| {config} | {pair} | " + " | ".join(f"{means[m]:.3f}" for m in COLUMNS) + " |")

    en_minus_ar = [
        (config, paired(per_question[(config, "en-en")], per_question[(config, "ar-ar")], qids, args.seed))
        for config in sorted({c for c, _ in per_question})
        if (config, "en-en") in per_question and (config, "ar-ar") in per_question
    ]
    print_paired("Paired EN - AR (same questions; 95% bootstrap CI; exact McNemar on hit@5)", "EN", "AR", en_minus_ar)
    output = {"corpus": args.corpus, "subset": args.subset, "n_questions": len(qids), "table": table, "en_minus_ar": dict(en_minus_ar)}

    if args.baseline_corpus:
        baseline = score_runs(data, args.baseline_corpus, questions)
        rows = [
            (f"{config} {pair}", paired(per_question[(config, pair)], baseline[(config, pair)], qids, args.seed))
            for config, pair in sorted(set(per_question) & set(baseline))
        ]
        print_paired(f"Paired {args.corpus} - {args.baseline_corpus} (same questions)", args.corpus, args.baseline_corpus, rows)
        output[f"minus_{args.baseline_corpus}"] = dict(rows)

    out = data / "results" / args.corpus
    out.mkdir(parents=True, exist_ok=True)
    name = "retrieval.json" if args.subset == "all" else f"retrieval_{args.subset}.json"
    (out / name).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
