"""Every figure in README.md is checked against the committed results files.

The README is the one document that restates numbers produced elsewhere, so it is the one that can go
stale silently: a re-run that shifts recall@5 by a thousandth leaves the prose saying the old value. Each
entry below names a results file, the path to the value inside it, and the string that must appear in the
README. The test fails if the file's value formats differently, or if that string is missing from the
README.

Figures whose source is a documented finding rather than a results file (hand-inspected rejections,
multi-hop funnels) are not covered here; they live in docs/corpus.md with their evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
RERANK = "rerank_hybrid_bm25-light__bge-m3"

# (label, file, path inside the JSON, format, the text that must appear in README.md)
FIGURES = [
    # Retrieval: the UN corpus
    ("rerank recall@5 EN", "unpc/retrieval.json", ("table", f"{RERANK}/en-en", "recall@5"), "{:.3f}", "0.938"),
    ("rerank recall@5 AR", "unpc/retrieval.json", ("table", f"{RERANK}/ar-ar", "recall@5"), "{:.3f}", "0.859"),
    ("rerank EN-AR", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "mean_diff"), "{:.3f}", "+0.078"),
    ("rerank CI low", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "ci_low"), "{:.3f}", "+0.016"),
    ("rerank CI high", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "ci_high"), "{:.3f}", "+0.141"),
    ("rerank McNemar p", "unpc/retrieval.json", ("en_minus_ar", RERANK, "mcnemar_hit@5", "p_value"), "{:.3f}", "0.031"),
    ("bm25-raw EN-AR", "unpc/retrieval.json", ("en_minus_ar", "bm25-raw", "recall@5", "mean_diff"), "{:.3f}", "+0.242"),
    ("bm25-light EN-AR", "unpc/retrieval.json", ("en_minus_ar", "bm25-light", "recall@5", "mean_diff"), "{:.3f}", "+0.094"),
    ("bm25-light p", "unpc/retrieval.json", ("en_minus_ar", "bm25-light", "mcnemar_hit@5", "p_value"), "{:.3f}", "0.088"),
    ("bm25-raw EN recall@5", "unpc/retrieval.json", ("table", "bm25-raw/en-en", "recall@5"), "{:.3f}", "0.805"),
    ("bm25-light EN recall@5", "unpc/retrieval.json", ("table", "bm25-light/en-en", "recall@5"), "{:.3f}", "0.742"),
    ("bm25-raw AR recall@5", "unpc/retrieval.json", ("table", "bm25-raw/ar-ar", "recall@5"), "{:.3f}", "0.562"),
    ("bm25-light AR recall@5", "unpc/retrieval.json", ("table", "bm25-light/ar-ar", "recall@5"), "{:.3f}", "0.648"),
    ("e5 en-ar recall@1", "unpc/retrieval.json", ("table", "e5-base/en-ar", "recall@1"), "{:.3f}", "0.094"),
    ("e5 ar-ar recall@1", "unpc/retrieval.json", ("table", "e5-base/ar-ar", "recall@1"), "{:.3f}", "0.398"),
    ("e5 ar-en recall@1", "unpc/retrieval.json", ("table", "e5-base/ar-en", "recall@1"), "{:.3f}", "0.328"),
    ("bge-m3 en-ar recall@1", "unpc/retrieval.json", ("table", "bge-m3/en-ar", "recall@1"), "{:.3f}", "0.367"),
    # Retrieval: XQuAD, the ceiling contrast
    ("XQuAD rerank EN", "xquad/retrieval.json", ("table", f"{RERANK}/en-en", "recall@5"), "{:.3f}", "1.000"),
    ("XQuAD rerank AR", "xquad/retrieval.json", ("table", f"{RERANK}/ar-ar", "recall@5"), "{:.3f}", "1.000"),
    ("XQuAD e5 en-ar", "xquad/retrieval.json", ("table", "e5-base/en-ar", "recall@1"), "{:.3f}", "0.823"),
    ("XQuAD bge-m3 en-ar", "xquad/retrieval.json", ("table", "bge-m3/en-ar", "recall@1"), "{:.3f}", "0.854"),
    ("XQuAD bm25-raw EN-AR", "xquad/retrieval.json", ("en_minus_ar", "bm25-raw", "recall@5", "mean_diff"), "{:.3f}", "+0.047"),
    ("XQuAD bm25-light EN-AR", "xquad/retrieval.json", ("en_minus_ar", "bm25-light", "recall@5", "mean_diff"), "{:.3f}", "+0.019"),
    # The numeric subset and the digit-group correction
    ("numeric rerank EN-AR", "unpc/retrieval_numeric.json", ("en_minus_ar", RERANK, "recall@5", "mean_diff"), "{:.3f}", "+0.023"),
    ("correction, e5 recall@5", "unpc/retrieval_numeric.json", ("minus_unpc_uncorrected", "e5-base ar-ar", "recall@5", "mean_diff"), "{:.3f}", "+0.023"),
    ("correction, bge-m3 MRR", "unpc/retrieval_numeric.json", ("minus_unpc_uncorrected", "bge-m3 ar-ar", "mrr@10", "mean_diff"), "{:.3f}", "+0.010"),
    # Generation controls
    ("oracle judge EN", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/en", "accuracy"), "{:.3f}", "0.984"),
    ("oracle judge AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/ar", "accuracy"), "{:.3f}", "0.953"),
    ("oracle judge EN-AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/en_minus_ar_accuracy", "mean_diff"), "{:.3f}", "+0.031"),
    ("closed-book judge EN", "unpc/generation_judge.json", ("openai_gpt-oss-20b/closed_book/en", "accuracy"), "{:.3f}", "0.031"),
    ("closed-book judge AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/closed_book/ar", "accuracy"), "{:.3f}", "0.031"),
    ("oracle exact match EN", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/en", "accuracy"), "{:.3f}", "0.711"),
    ("oracle exact match AR", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/ar", "accuracy"), "{:.3f}", "0.539"),
    ("oracle exact-match EN-AR", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/en_minus_ar_accuracy", "mean_diff"), "{:.3f}", "+0.172"),
]

# Corpus counts, written with thousands separators in the README.
CORPUS_FIGURES = [
    ("chunks", ("chunks",), "33,476"),
    ("document pairs", ("pairs",), "114,047"),
    ("eligible pairs", ("eligible_pairs",), "35,979"),
    ("numbers corrected", ("numbers_corrected",), "3,600"),
    ("chunks with corrections", ("chunks_with_corrections",), "1,604"),
    ("Arabic-only truncated", ("truncated_e5", "ar_only"), "40"),
]


def dig(data: dict, path: tuple[str, ...]):
    for key in path:
        data = data[key]
    return data


def load(relative: str) -> dict:
    path = RESULTS / relative
    if not path.exists():
        pytest.skip(f"{path} not committed; results are needed to check the README")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("label,source,path,fmt,expected", FIGURES, ids=[f[0] for f in FIGURES])
def test_readme_figure_matches_results_file(label, source, path, fmt, expected, readme):
    value = dig(load(source), path)
    formatted = fmt.format(value)
    signed = f"+{formatted}" if expected.startswith("+") and not formatted.startswith("-") else formatted
    assert signed == expected, f"{label}: {source} holds {signed}, README says {expected}"
    assert expected in readme, f"{label}: {expected} is not in README.md"


@pytest.mark.parametrize("label,path,expected", CORPUS_FIGURES, ids=[f[0] for f in CORPUS_FIGURES])
def test_readme_corpus_count_matches_stats(label, path, expected, readme):
    stats_path = ROOT / "data" / "corpus" / "unpc" / "stats.json"
    if not stats_path.exists():
        pytest.skip("data/corpus/unpc/stats.json not present; run scripts/build_unpc.py")
    value = dig(json.loads(stats_path.read_text(encoding="utf-8")), path)
    assert f"{value:,}" == expected, f"{label}: stats.json holds {value:,}, README says {expected}"
    assert expected in readme, f"{label}: {expected} is not in README.md"
