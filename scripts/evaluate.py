"""Score every run of a corpus and print the retrieval table with paired EN-AR comparisons.

    python scripts/evaluate.py --corpus xquad

For each config and language pair: recall@1/5/10/20, all_recall@5 (differs from recall@5 only for
multi-group questions), MRR@10 and nDCG@10. For each config run in both en-en and ar-ar: the
paired difference EN - AR in recall@5, MRR@10 and nDCG@10 with a 95% bootstrap CI over questions,
and an exact McNemar test on hit@5. Writes data/results/<corpus>/retrieval.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rageval.eval.metrics import question_metrics
from rageval.eval.stats import mcnemar_exact, paired_bootstrap_ci
from rageval.io import read_jsonl

COLUMNS = ["recall@1", "recall@5", "recall@10", "recall@20", "all_recall@5", "mrr@10", "ndcg@10"]
PAIRED = ["recall@5", "mrr@10", "ndcg@10"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["xquad", "unpc"], required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    data = Path(args.data)
    questions_file = {"xquad": "xquad.jsonl", "unpc": "unpc_pilot.jsonl"}[args.corpus]
    questions = {q["qid"]: q for q in read_jsonl(data / "questions" / questions_file)}

    per_question: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for run_file in sorted((data / "runs" / args.corpus).glob("*/*.jsonl")):
        config, pair = run_file.parent.name, run_file.stem
        rows = {}
        for record in read_jsonl(run_file):
            ranking = [c for c, _ in record["ranking"]]
            rows[record["qid"]] = question_metrics(ranking, questions[record["qid"]]["relevant_groups"])
        missing = set(questions) - set(rows)
        if missing:
            raise SystemExit(f"{run_file}: {len(missing)} questions have no ranking")
        per_question[(config, pair)] = rows

    qids = sorted(questions)
    table = {}
    print(f"\n{args.corpus}: {len(qids)} questions\n")
    print("| config | query-doc | " + " | ".join(COLUMNS) + " |")
    print("|---|---|" + "---|" * len(COLUMNS))
    for (config, pair), rows in sorted(per_question.items()):
        means = {m: float(np.mean([rows[q][m] for q in qids])) for m in COLUMNS}
        table[f"{config}/{pair}"] = means
        print(f"| {config} | {pair} | " + " | ".join(f"{means[m]:.3f}" for m in COLUMNS) + " |")

    comparisons = {}
    configs = sorted({c for c, _ in per_question})
    print("\nPaired EN - AR (same questions; 95% bootstrap CI; exact McNemar on hit@5)\n")
    print("| config | " + " | ".join(f"Δ {m} [CI]" for m in PAIRED) + " | hit@5 EN-only / AR-only | McNemar p |")
    print("|---|" + "---|" * (len(PAIRED) + 2))
    for config in configs:
        if (config, "en-en") not in per_question or (config, "ar-ar") not in per_question:
            continue
        en, ar = per_question[(config, "en-en")], per_question[(config, "ar-ar")]
        cells, result = [], {}
        for m in PAIRED:
            ci = paired_bootstrap_ci([en[q][m] for q in qids], [ar[q][m] for q in qids], seed=args.seed)
            result[m] = ci
            cells.append(f"{ci['mean_diff']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
        test = mcnemar_exact([en[q]["recall@5"] == 1.0 for q in qids], [ar[q]["recall@5"] == 1.0 for q in qids])
        result["mcnemar_hit@5"] = test
        comparisons[config] = result
        print(f"| {config} | " + " | ".join(cells) + f" | {test['a_only']} / {test['b_only']} | {test['p_value']:.3g} |")

    out = data / "results" / args.corpus
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps({"n_questions": len(qids), "table": table, "en_minus_ar": comparisons}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
