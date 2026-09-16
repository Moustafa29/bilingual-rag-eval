"""Two measurements on answers that already exist. Reads the LLM cache and the usage ledger; no calls.

    python scripts/analyze_scoring.py --corpus unpc

1. Exact match against the judge, per model and language: how often the judge accepts an answer that
   exact match rejects, what has to change for exact match to accept it, and whether the penalty is
   larger in Arabic (paired exact McNemar on the same questions).
2. Tokens per character of Arabic and English text alone. A prompt is an English template plus fields in
   the answer language, so a whole-prompt ratio mixes the two. Each prompt is split into constant
   template characters and language-specific field characters, and

       real prompt tokens = intercept + slope x field characters

   is fitted per language by least squares over the ledger. The slope is tokens per character of that
   language's text; the intercept estimates the shared template and should agree between the two fits,
   which is the check that the split measures what it claims.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rageval.eval.answers import exact_match
from rageval.eval.stats import mcnemar_exact
from rageval.io import load_config, read_jsonl
from rageval.llm.client import DiskCache, cache_key
from rageval.questions.builder import LANG_NAMES, Prompts, format_passages
from rageval.questions.checks import is_true, parse_json_object

QUESTION_FILES = {"unpc": "unpc_pilot.jsonl", "xquad": "xquad_pilot.jsonl"}
# Thousands separators, between two digits only: comma (English convention), space, non-breaking space,
# narrow no-break space, Arabic thousands separator (UN Arabic writes thousands with a space).
GROUP_SEPARATOR = re.compile(r"(?<=\d)[,\s  ٬ ](?=\d)")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def difference_kind(answer: str, reference: str, lang: str) -> str:
    """What would have to change for exact match to accept this answer."""
    stripped = GROUP_SEPARATOR.sub("", answer), GROUP_SEPARATOR.sub("", reference)
    if exact_match(stripped[0], stripped[1], lang):
        return "digit grouping alone"
    in_answer, in_reference = set(NUMBER.findall(stripped[0])), set(NUMBER.findall(stripped[1]))
    if in_answer and in_answer == in_reference:
        return "same numbers, wording around them differs"
    if in_answer and in_reference:
        return "numbers differ"
    return "wording differs, no numbers"


def fit_tokens_per_char(rows: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    field_chars = np.array([r[0] for r in rows], float)
    tokens = np.array([r[2] for r in rows], float)
    design = np.vstack([field_chars, np.ones_like(field_chars)]).T
    slope, intercept = np.linalg.lstsq(design, tokens, rcond=None)[0]
    predicted = design @ [slope, intercept]
    r2 = 1 - ((tokens - predicted) ** 2).sum() / ((tokens - tokens.mean()) ** 2).sum()
    return float(slope), float(intercept), float(r2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), default="unpc")
    parser.add_argument("--condition", default="oracle", help="which answers to read (default: oracle)")
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data"])
    questions = {q["qid"]: q for q in read_jsonl(data / "questions" / QUESTION_FILES[args.corpus])}
    prompts = Prompts(cfg["paths"]["prompts"])
    cache = DiskCache(data / "llm_cache")
    judge_cfg = cfg["generation"]["judge"]
    answerers = {m["model"]: m for m in (cfg["generation"]["answerer"], cfg["generation"]["contamination_answerer"])}

    ledger = {}
    for line in (data / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        ledger[entry["key"]] = entry  # one key is one distinct prompt; repeats were cache hits, not calls

    def judge_prompt(row: dict, lang: str) -> str:
        q = questions[row["qid"]]
        return prompts.render("judge_correctness", lang_name=LANG_NAMES[lang], question=q["question"][lang],
                              reference=q["answer"][lang], candidate=row["answer"])

    def judged_correct(row: dict, lang: str) -> bool | None:
        key = cache_key(judge_cfg["base_url"], judge_cfg["model"],
                        [{"role": "user", "content": judge_prompt(row, lang)}], judge_cfg["params"])
        record = cache.get(key)
        if record is None:
            return None
        return is_true((parse_json_object(record["response"]["text"]) or {}).get("correct"))

    files = sorted(p for p in (data / "generation" / args.corpus).glob(f"*/{args.condition}/*.jsonl")
                   if not p.name.endswith(".scored.jsonl"))
    if not files:
        raise SystemExit(f"no {args.condition} answers under data/generation/{args.corpus}")

    print(f"## Exact match against the judge ({args.condition} answers)\n")
    penalised: dict[tuple[str, str], dict[str, bool]] = {}
    for path in files:
        model, lang = path.parent.parent.name, path.stem
        rows = read_jsonl(path)
        kinds, per_qid, examples = Counter(), {}, []
        judge_only = em_only = both = neither = uncached = 0
        for row in rows:
            reference = questions[row["qid"]]["answer"][lang]
            em = exact_match(row["answer"], reference, lang) and not row["abstained"]
            judged = judged_correct(row, lang)
            if judged is None:
                uncached += 1
                continue
            judged = judged and not row["abstained"]
            per_qid[row["qid"]] = bool(judged and not em)
            if judged and not em:
                judge_only += 1
                kinds[difference_kind(row["answer"], reference, lang)] += 1
                examples.append((row["qid"], reference, row["answer"]))
            elif em and not judged:
                em_only += 1
            elif em:
                both += 1
            else:
                neither += 1
        penalised[(model, lang)] = per_qid
        print(f"### {model} / {lang} (n={len(rows)}, judge verdicts not cached: {uncached})")
        print(f"- both correct {both} | both wrong {neither} | judge correct, exact match wrong {judge_only} "
              f"| exact match correct, judge wrong {em_only}")
        for kind, n in kinds.most_common():
            print(f"  - {kind}: {n}")
        for qid, reference, answer in examples[: args.examples]:
            print(f"  - {qid}: reference {reference!r} vs answer {answer!r}")
        print()

    print("## Is the exact-match penalty larger in Arabic (paired, exact McNemar)\n")
    for model in sorted({m for m, _ in penalised}):
        ar, en = penalised.get((model, "ar")), penalised.get((model, "en"))
        if not ar or not en:
            continue
        qids = sorted(set(ar) & set(en))
        a, b = [ar[q] for q in qids], [en[q] for q in qids]
        test = mcnemar_exact(a, b)
        print(f"- {model} (n={len(qids)}): Arabic only {test['a_only']}, English only {test['b_only']}, "
              f"both {sum(1 for x, y in zip(a, b) if x and y)}, p = {test['p_value']:.3f}")

    print("\n## Tokens per character by language, fitted from the ledger\n")
    chunks = {c["chunk_id"]: c for c in read_jsonl(data / "corpus" / args.corpus / "chunks.jsonl")}
    samples: dict[str, dict[str, list[tuple[int, int, int]]]] = {
        "judge prompts": {"en": [], "ar": []},
        "answering prompts": {"en": [], "ar": []},
    }
    for path in files:
        lang = path.stem
        for row in read_jsonl(path):
            q = questions[row["qid"]]
            prompt = judge_prompt(row, lang)
            fields = q["question"][lang] + q["answer"][lang] + row["answer"]
            entry = ledger.get(cache_key(judge_cfg["base_url"], judge_cfg["model"],
                                         [{"role": "user", "content": prompt}], judge_cfg["params"]))
            if entry:
                samples["judge prompts"][lang].append((len(fields), len(prompt) - len(fields), entry["prompt_tokens"]))
            answerer = answerers.get(row["model"])
            if not row["context_ids"] or answerer is None:
                continue
            passages = format_passages([chunks[c][lang] for c in row["context_ids"]])
            answer_prompt = prompts.render("answer_rag", lang_name=LANG_NAMES[lang],
                                           question=q["question"][lang], passages=passages)
            entry = ledger.get(cache_key(answerer["base_url"], answerer["model"],
                                         [{"role": "user", "content": answer_prompt}], answerer["params"]))
            if entry:
                fields = q["question"][lang] + passages
                samples["answering prompts"][lang].append(
                    (len(fields), len(answer_prompt) - len(fields), entry["prompt_tokens"]))

    for kind, by_lang in samples.items():
        print(f"### {kind}")
        slopes = {}
        for lang, rows in by_lang.items():
            if len(rows) < 5:
                print(f"- {lang}: {len(rows)} matched calls, too few to fit")
                continue
            slope, intercept, r2 = fit_tokens_per_char(rows)
            slopes[lang] = slope
            mean_fields = sum(r[0] for r in rows) / len(rows)
            print(f"- {lang}: {len(rows)} calls, {mean_fields:.0f} field characters on average | "
                  f"{slope:.4f} tokens per character ({1 / slope:.2f} characters per token) | "
                  f"template intercept {intercept:.0f} tokens | R^2 {r2:.2f}")
        if len(slopes) == 2:
            print(f"- Arabic / English tokens per character: {slopes['ar'] / slopes['en']:.2f}")
        print()


if __name__ == "__main__":
    main()
