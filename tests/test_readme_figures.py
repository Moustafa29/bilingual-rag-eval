"""Every figure quoted in the write-up is checked against the committed results files.

README.md and docs/limitations.md restate numbers produced elsewhere, so they are the documents that can
go stale silently: a re-run that shifts recall@5 by a thousandth leaves the prose saying the old value.
Each entry below names a results file, the path to the value inside it, the string that must appear, and
which document it must appear in. The test fails if the file's value formats differently, or if the
string is missing from that document.

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

# (label, file, path inside the JSON, format, the text that must appear, the document it appears in)
FIGURES = [
    # Retrieval: the UN corpus
    ("rerank recall@5 EN", "unpc/retrieval.json", ("table", f"{RERANK}/en-en", "recall@5"), "{:.3f}", "0.938", "README.md"),
    ("rerank recall@5 AR", "unpc/retrieval.json", ("table", f"{RERANK}/ar-ar", "recall@5"), "{:.3f}", "0.859", "README.md"),
    ("rerank EN-AR", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "mean_diff"), "{:.3f}", "+0.078", "README.md"),
    ("rerank CI low", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "ci_low"), "{:.3f}", "+0.016", "README.md"),
    ("rerank CI high", "unpc/retrieval.json", ("en_minus_ar", RERANK, "recall@5", "ci_high"), "{:.3f}", "+0.141", "README.md"),
    ("rerank McNemar p", "unpc/retrieval.json", ("en_minus_ar", RERANK, "mcnemar_hit@5", "p_value"), "{:.3f}", "0.031", "README.md"),
    ("bm25-raw EN-AR", "unpc/retrieval.json", ("en_minus_ar", "bm25-raw", "recall@5", "mean_diff"), "{:.3f}", "+0.242", "README.md"),
    ("bm25-light EN-AR", "unpc/retrieval.json", ("en_minus_ar", "bm25-light", "recall@5", "mean_diff"), "{:.3f}", "+0.094", "README.md"),
    ("bm25-light p", "unpc/retrieval.json", ("en_minus_ar", "bm25-light", "mcnemar_hit@5", "p_value"), "{:.3f}", "0.088", "README.md"),
    ("bm25-raw EN recall@5", "unpc/retrieval.json", ("table", "bm25-raw/en-en", "recall@5"), "{:.3f}", "0.805", "README.md"),
    ("bm25-light EN recall@5", "unpc/retrieval.json", ("table", "bm25-light/en-en", "recall@5"), "{:.3f}", "0.742", "README.md"),
    ("bm25-raw AR recall@5", "unpc/retrieval.json", ("table", "bm25-raw/ar-ar", "recall@5"), "{:.3f}", "0.562", "README.md"),
    ("bm25-light AR recall@5", "unpc/retrieval.json", ("table", "bm25-light/ar-ar", "recall@5"), "{:.3f}", "0.648", "README.md"),
    ("e5 en-ar recall@1", "unpc/retrieval.json", ("table", "e5-base/en-ar", "recall@1"), "{:.3f}", "0.094", "README.md"),
    ("e5 ar-ar recall@1", "unpc/retrieval.json", ("table", "e5-base/ar-ar", "recall@1"), "{:.3f}", "0.398", "README.md"),
    ("e5 ar-en recall@1", "unpc/retrieval.json", ("table", "e5-base/ar-en", "recall@1"), "{:.3f}", "0.328", "README.md"),
    ("bge-m3 en-ar recall@1", "unpc/retrieval.json", ("table", "bge-m3/en-ar", "recall@1"), "{:.3f}", "0.367", "README.md"),
    # Retrieval: XQuAD, the ceiling contrast
    ("XQuAD rerank EN", "xquad/retrieval.json", ("table", f"{RERANK}/en-en", "recall@5"), "{:.3f}", "1.000", "README.md"),
    ("XQuAD rerank AR", "xquad/retrieval.json", ("table", f"{RERANK}/ar-ar", "recall@5"), "{:.3f}", "1.000", "README.md"),
    ("XQuAD e5 en-ar", "xquad/retrieval.json", ("table", "e5-base/en-ar", "recall@1"), "{:.3f}", "0.823", "README.md"),
    ("XQuAD bge-m3 en-ar", "xquad/retrieval.json", ("table", "bge-m3/en-ar", "recall@1"), "{:.3f}", "0.854", "README.md"),
    ("XQuAD bm25-raw EN-AR", "xquad/retrieval.json", ("en_minus_ar", "bm25-raw", "recall@5", "mean_diff"), "{:.3f}", "+0.047", "README.md"),
    ("XQuAD bm25-light EN-AR", "xquad/retrieval.json", ("en_minus_ar", "bm25-light", "recall@5", "mean_diff"), "{:.3f}", "+0.019", "README.md"),
    # The numeric subset and the digit-group correction
    ("numeric rerank EN-AR", "unpc/retrieval_numeric.json", ("en_minus_ar", RERANK, "recall@5", "mean_diff"), "{:.3f}", "+0.023", "README.md"),
    ("correction, e5 recall@5", "unpc/retrieval_numeric.json", ("minus_unpc_uncorrected", "e5-base ar-ar", "recall@5", "mean_diff"), "{:.3f}", "+0.023", "README.md"),
    ("correction, bge-m3 MRR", "unpc/retrieval_numeric.json", ("minus_unpc_uncorrected", "bge-m3 ar-ar", "mrr@10", "mean_diff"), "{:.3f}", "+0.010", "README.md"),
    # Generation controls
    ("oracle judge EN", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/en", "accuracy"), "{:.3f}", "0.984", "README.md"),
    ("oracle judge AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/ar", "accuracy"), "{:.3f}", "0.953", "README.md"),
    ("oracle judge EN-AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/oracle/en_minus_ar_accuracy", "mean_diff"), "{:.3f}", "+0.031", "README.md"),
    ("closed-book judge EN", "unpc/generation_judge.json", ("openai_gpt-oss-20b/closed_book/en", "accuracy"), "{:.3f}", "0.031", "README.md"),
    ("closed-book judge AR", "unpc/generation_judge.json", ("openai_gpt-oss-20b/closed_book/ar", "accuracy"), "{:.3f}", "0.031", "README.md"),
    ("oracle exact match EN", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/en", "accuracy"), "{:.3f}", "0.711", "docs/limitations.md"),
    ("oracle exact match AR", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/ar", "accuracy"), "{:.3f}", "0.539", "docs/limitations.md"),
    ("oracle exact-match EN-AR", "unpc/generation_exact_match.json", ("openai_gpt-oss-20b/oracle/en_minus_ar_accuracy", "mean_diff"), "{:.3f}", "+0.172", "README.md"),
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
def documents() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "docs/limitations.md")}


@pytest.mark.parametrize("label,source,path,fmt,expected,document", FIGURES, ids=[f[0] for f in FIGURES])
def test_quoted_figure_matches_results_file(label, source, path, fmt, expected, document, documents):
    value = dig(load(source), path)
    formatted = fmt.format(value)
    signed = f"+{formatted}" if expected.startswith("+") and not formatted.startswith("-") else formatted
    assert signed == expected, f"{label}: {source} holds {signed}, {document} says {expected}"
    assert expected in documents[document], f"{label}: {expected} is not in {document}"


@pytest.mark.parametrize("label,path,expected", CORPUS_FIGURES, ids=[f[0] for f in CORPUS_FIGURES])
def test_readme_corpus_count_matches_stats(label, path, expected, documents):
    stats_path = ROOT / "data" / "corpus" / "unpc" / "stats.json"
    if not stats_path.exists():
        pytest.skip("data/corpus/unpc/stats.json not present; run scripts/build_unpc.py")
    value = dig(json.loads(stats_path.read_text(encoding="utf-8")), path)
    assert f"{value:,}" == expected, f"{label}: stats.json holds {value:,}, README says {expected}"
    assert expected in documents["README.md"], f"{label}: {expected} is not in README.md"
