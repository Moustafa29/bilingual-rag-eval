# Retrieval results (Phase 3)

Every number comes from `python scripts/evaluate.py --corpus <corpus>`, which writes
`data/results/<corpus>/retrieval.json`. Empty cells have not been run yet: dense models and
the reranker need the Colab T4 (`docs/colab.md`).

## Headline so far: light stemming halves BM25's Arabic gap

On XQuAD's 1,190 paired questions, BM25 with Arabic light stemming cuts the English − Arabic
MRR@10 gap from **+0.082** (raw tokens) to **+0.041**. Normalization alone does almost nothing
(+0.081).

| BM25 analyzer | Arabic recall@1 | Δ MRR@10, EN − AR [95% CI] |
|---|---|---|
| raw (whitespace, lowercase) | 0.816 | +0.082 [+0.065, +0.099] |
| norm (+ orthographic normalization) | 0.819 | +0.081 [+0.064, +0.098] |
| light (+ Lucene-style light stemming) | 0.876 | +0.041 [+0.028, +0.055] |

- **What it means:** about half of naive BM25's Arabic penalty comes from the tokenizer (clitics
  like و، ب، ال left attached), not from the language. An Arabic lexical baseline without an
  analyzer overstates the gap by about 2×.
- **What remains:** the gap after stemming is still there (CI excludes zero). Light stemming
  cannot relate broken plurals, one candidate for the rest.
- **Limit of this table:** XQuAD is near its ceiling. English BM25 reaches 0.98 recall@5 over
  only 240 passages, so this corpus can rank analyzers but cannot show how *large* the Arabic
  gap is. That needs the 33,476-chunk UN corpus.

## XQuAD (240 passages, 1,190 questions, human-translated)

Retrieval is over all 240 passages; each question's gold passage is its own paragraph.

**Ceiling effect.** With 240 passages and one gold each, English BM25 already reaches 0.98
recall@5. This corpus can separate configurations near the top, but it can't show a large gap.
The UN corpus (33,476 chunks) is where gaps have room to appear.

### Same-language retrieval

| Config | Query–doc | recall@1 | recall@5 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| bm25-raw | en-en | 0.919 | 0.984 | 0.992 | 0.993 | 0.949 | 0.960 |
| bm25-raw | ar-ar | 0.816 | 0.937 | 0.954 | 0.963 | 0.867 | 0.889 |
| bm25-norm | en-en | 0.919 | 0.984 | 0.992 | 0.993 | 0.949 | 0.960 |
| bm25-norm | ar-ar | 0.819 | 0.936 | 0.953 | 0.963 | 0.869 | 0.890 |
| bm25-light | en-en | 0.932 | 0.987 | 0.994 | 0.996 | 0.958 | 0.967 |
| bm25-light | ar-ar | 0.876 | 0.967 | 0.978 | 0.984 | 0.916 | 0.932 |
| e5-base | en-en | | | | | | |
| e5-base | ar-ar | | | | | | |
| bge-m3 | en-en | | | | | | |
| bge-m3 | ar-ar | | | | | | |
| hybrid (bm25-light + e5-base) | en-en | | | | | | |
| hybrid (bm25-light + e5-base) | ar-ar | | | | | | |
| hybrid (bm25-light + bge-m3) | en-en | | | | | | |
| hybrid (bm25-light + bge-m3) | ar-ar | | | | | | |
| hybrid + reranker | en-en | | | | | | |
| hybrid + reranker | ar-ar | | | | | | |

### Paired EN − AR

Same 1,190 questions in both languages. CIs are 95% paired bootstrap intervals over questions
(10,000 resamples). McNemar tests hit@5.

| Config | Δ recall@5 [95% CI] | Δ MRR@10 [95% CI] | Δ nDCG@10 [95% CI] | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|---|
| bm25-raw | +0.047 [+0.033, +0.061] | +0.082 [+0.065, +0.099] | +0.071 [+0.057, +0.086] | 68 / 12 | 1.2e-10 |
| bm25-norm | +0.048 [+0.034, +0.062] | +0.081 [+0.064, +0.098] | +0.070 [+0.056, +0.085] | 69 / 12 | 7.0e-11 |
| bm25-light | +0.019 [+0.009, +0.029] | +0.041 [+0.028, +0.055] | +0.035 [+0.024, +0.047] | 31 / 8 | 2.9e-4 |
| e5-base | | | | | |
| bge-m3 | | | | | |
| hybrid + reranker | | | | | |

### Cross-lingual (dense only)

| Model | en-en | ar-en (AR query, EN docs) | en-ar (EN query, AR docs) | ar-ar |
|---|---|---|---|---|
| e5-base recall@5 | | | | |
| bge-m3 recall@5 | | | | |

## UN corpus pilot

Not run yet: the question set does not exist until Groq generation finishes.
