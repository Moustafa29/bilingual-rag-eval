# Retrieval results (Phase 3)

Every number comes from `python scripts/evaluate.py --corpus <corpus>`, which writes
`data/results/<corpus>/retrieval.json`. XQuAD's dense, hybrid and reranker rows were run on a Colab T4
(`docs/colab.md`); their results file is committed at `7486ae2`.

## XQuAD: what closes the Arabic gap

Same 1,190 questions in English and Arabic, retrieved from 240 passages (professional translation).
Every English − Arabic comparison is paired by question. Intervals are 95% paired bootstrap
intervals (10,000 resamples), and McNemar's test is run on hit@5.

**Ceiling caveat, for everything in this section:**
- **The corpus is small.** Each question has one gold paragraph among only 240.
- **Near the top already:** English BM25 reaches 0.984 recall@5, and every dense configuration
  reaches at least 0.990 in English.
- **What that means:** XQuAD can rank configurations near the top of the scale. It cannot show how
  large an Arabic gap would be on a realistic corpus. That needs the UN corpus.

### 1. The recall@5 gap narrows tier by tier, to exactly zero after reranking

| Tier | Config | Δ recall@5, EN − AR [95% CI] | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|
| Lexical | bm25-raw | +0.047 [+0.033, +0.061] | 68 / 12 | 1.2e-10 |
| Lexical | bm25-norm | +0.048 [+0.034, +0.062] | 69 / 12 | 7.0e-11 |
| Lexical, stemmed | bm25-light | +0.019 [+0.009, +0.029] | 31 / 8 | 2.9e-4 |
| Dense | e5-base | +0.012 [+0.005, +0.019] | 16 / 2 | 0.0013 |
| Dense | bge-m3 | +0.010 [+0.003, +0.018] | 16 / 4 | 0.012 |
| Hybrid (RRF) | bm25-light + e5-base | +0.008 [+0.003, +0.015] | 12 / 2 | 0.013 |
| Hybrid (RRF) | bm25-light + bge-m3 | +0.011 [+0.003, +0.018] | 17 / 4 | 0.0072 |
| Reranked | hybrid bm25-light + bge-m3, bge-reranker-v2-m3 over the top 50 | **+0.000 [+0.000, +0.000]** | **0 / 0** | **1.0** |

- **Tier by tier, the gap closes:** lexical +0.047–0.048 → stemmed +0.019 → dense and hybrid
  +0.008–0.012 → reranked +0.000. After reranking, no question is found in the top 5 in one
  language but not the other.
- **Not strictly monotonic config by config:**
  - Normalization is 0.001 above raw tokens.
  - The bge-m3 hybrid (+0.011) is above bge-m3 alone (+0.010), and the reranker is built on that
    hybrid.
- **Tokenizer vs language:** within the lexical tier, light stemming halves BM25's gap. MRR@10
  falls from +0.082 to +0.041, and recall@5 from +0.047 to +0.019. About half of naive BM25's Arabic
  penalty is the tokenizer (clitics like و، ب، ال left attached), not the language. Normalization
  alone changes nothing.
- **Ceiling:** after reranking, recall@5 is **1.000 in both languages**. Every one of the 1,190 gold
  passages is in the top 5, in English and in Arabic. On XQuAD the reranked gap cannot be anything
  but zero. This shows the reranker saturates XQuAD, not that it removes the Arabic penalty in
  general.

### 2. Finding the passage is solved here; ranking it first is not

| After reranking | English | Arabic | Δ EN − AR [95% CI] |
|---|---|---|---|
| recall@5 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| recall@1 | 0.985 | 0.971 | +0.013 (no interval computed) |
| MRR@10 | 0.992 | 0.984 | **+0.008 [+0.003, +0.012]** |
| nDCG@10 | 0.994 | 0.988 | +0.006 [+0.002, +0.009] |

- **Two different claims:**
  - recall@5 says the gold passage is among the five a generator would receive.
  - MRR@10 and nDCG@10 say *where* it is. Both still favour English, with intervals excluding zero.
- **Not closed:** reranking shrinks the MRR gap about fourfold (from +0.031 for the hybrid it
  reranks) but does not close it.
- **Whether it matters for answers** is measured in Phase 4, which records where the gold passage
  sits in the context of every question.
- **Ceiling:** English MRR@10 is already 0.992, so the residual gap is measurable on XQuAD, but its
  size on a harder corpus is unknown.

### 3. e5-base is asymmetric across languages; bge-m3 is not

Rows are query language – passage language: `ar-en` means Arabic questions over English passages.

| Model | Metric | en-en | ar-en | en-ar | ar-ar |
|---|---|---|---|---|---|
| e5-base | recall@1 | 0.955 | **0.766** | **0.823** | 0.889 |
| e5-base | recall@5 | 0.996 | 0.941 | 0.971 | 0.984 |
| e5-base | MRR@10 | 0.974 | 0.841 | 0.888 | 0.931 |
| bge-m3 | recall@1 | 0.940 | **0.849** | **0.854** | 0.890 |
| bge-m3 | recall@5 | 0.991 | 0.970 | 0.968 | 0.981 |
| bge-m3 | MRR@10 | 0.963 | 0.903 | 0.904 | 0.930 |

- **e5-base is lopsided.** Starting from English questions over English passages (0.955 recall@1):
  - Arabic questions over English passages: 0.766 (−0.189).
  - English questions over Arabic passages: 0.823 (−0.132).
  - So its cross-lingual retrieval is weaker when the question is in Arabic than when the passages
    are.
- **bge-m3 loses about the same either way:** 0.940 → 0.849 and 0.854, a difference of 0.005.
- **This is a difference between the models,** consistent at recall@1, recall@5 and MRR@10 for e5,
  and absent at all three for bge-m3.

**Limits:**
- **No confidence interval.** `evaluate.py` computes paired intervals only for en-en against ar-ar.
  A paired test for ar-en against en-ar needs the Colab run files, which were not brought back.
- **Two effects mixed.** Cross-lingual retrieval combines how well each language is encoded with how
  well the two languages are aligned in the embedding space. The 2×2 separates those only partly.
- **Ceiling:** at recall@5 every cell is above 0.94, so the asymmetry is clearest at recall@1.

### Full table: same-language retrieval

`all_recall@5` is omitted: every XQuAD question has one relevance group, so it equals recall@5.

| Config | Query–passage | recall@1 | recall@5 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| bm25-raw | en-en | 0.919 | 0.984 | 0.992 | 0.993 | 0.949 | 0.960 |
| bm25-raw | ar-ar | 0.816 | 0.937 | 0.954 | 0.963 | 0.867 | 0.889 |
| bm25-norm | en-en | 0.919 | 0.984 | 0.992 | 0.993 | 0.949 | 0.960 |
| bm25-norm | ar-ar | 0.819 | 0.936 | 0.953 | 0.963 | 0.869 | 0.890 |
| bm25-light | en-en | 0.932 | 0.987 | 0.994 | 0.996 | 0.958 | 0.967 |
| bm25-light | ar-ar | 0.876 | 0.967 | 0.978 | 0.984 | 0.916 | 0.932 |
| e5-base | en-en | 0.955 | 0.996 | 0.999 | 1.000 | 0.974 | 0.980 |
| e5-base | ar-ar | 0.889 | 0.984 | 0.991 | 0.996 | 0.931 | 0.946 |
| bge-m3 | en-en | 0.940 | 0.991 | 0.997 | 0.999 | 0.963 | 0.972 |
| bge-m3 | ar-ar | 0.890 | 0.981 | 0.988 | 0.993 | 0.930 | 0.945 |
| hybrid bm25-light + e5-base | en-en | 0.966 | 0.997 | 0.997 | 0.997 | 0.979 | 0.984 |
| hybrid bm25-light + e5-base | ar-ar | 0.903 | 0.988 | 0.990 | 0.992 | 0.940 | 0.953 |
| hybrid bm25-light + bge-m3 | en-en | 0.957 | 0.994 | 0.997 | 0.997 | 0.974 | 0.980 |
| hybrid bm25-light + bge-m3 | ar-ar | 0.911 | 0.983 | 0.988 | 0.992 | 0.943 | 0.955 |
| hybrid bm25-light + bge-m3, reranked | en-en | 0.985 | 1.000 | 1.000 | 1.000 | 0.992 | 0.994 |
| hybrid bm25-light + bge-m3, reranked | ar-ar | 0.971 | 1.000 | 1.000 | 1.000 | 0.984 | 0.988 |

### Full table: paired EN − AR

| Config | Δ recall@5 [95% CI] | Δ MRR@10 [95% CI] | Δ nDCG@10 [95% CI] | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|---|
| bm25-raw | +0.047 [+0.033, +0.061] | +0.082 [+0.065, +0.099] | +0.071 [+0.057, +0.086] | 68 / 12 | 1.2e-10 |
| bm25-norm | +0.048 [+0.034, +0.062] | +0.081 [+0.064, +0.098] | +0.070 [+0.056, +0.085] | 69 / 12 | 7.0e-11 |
| bm25-light | +0.019 [+0.009, +0.029] | +0.041 [+0.028, +0.055] | +0.035 [+0.024, +0.047] | 31 / 8 | 2.9e-4 |
| e5-base | +0.012 [+0.005, +0.019] | +0.043 [+0.032, +0.054] | +0.034 [+0.026, +0.043] | 16 / 2 | 0.0013 |
| bge-m3 | +0.010 [+0.003, +0.018] | +0.033 [+0.023, +0.043] | +0.027 [+0.019, +0.035] | 16 / 4 | 0.012 |
| hybrid bm25-light + e5-base | +0.008 [+0.003, +0.015] | +0.039 [+0.029, +0.050] | +0.031 [+0.023, +0.040] | 12 / 2 | 0.013 |
| hybrid bm25-light + bge-m3 | +0.011 [+0.003, +0.018] | +0.031 [+0.021, +0.041] | +0.025 [+0.017, +0.034] | 17 / 4 | 0.0072 |
| hybrid bm25-light + bge-m3, reranked | +0.000 [+0.000, +0.000] | +0.008 [+0.003, +0.012] | +0.006 [+0.002, +0.009] | 0 / 0 | 1.0 |

### Reproducibility check

The BM25 rows were computed twice: on the laptop (`4fc2f9d`) and on Colab (`7486ae2`). Every BM25
metric, interval and McNemar count is identical between the two, compared as JSON text. The bootstrap
seed is fixed, so identical rankings must give identical intervals.
- **Scope:** the comparison covers the scores; the Colab ranking files were not brought back.
- **Source of the other rows:** the dense, hybrid and reranker rows exist only from the Colab run.

## UN corpus pilot

The question set is incomplete: single-hop and numeric questions only, with no multi-hop questions
(`docs/corpus.md` §3). The tables below cover only what that set supports.

**Question set used here:** re-verified with `qwen/qwen3.8-27b` after the original verifier was
withdrawn (`docs/corpus.md` §3). It has 82 single-hop questions (of a target of 100) and 40
numeric questions. Single-hop generation continues, so the numeric subset below can still grow.

### What correcting the Arabic digit groups is worth: BM25

`unpc` has the reversed Arabic digit groups corrected; `unpc_uncorrected` is the text as
distributed (`docs/corpus.md`, second result). Measured on the 43 questions whose English question
or answer contains a grouped number:

| Config | Δ recall@5, corrected − uncorrected, ar-ar | Same, en-en (sanity check) |
|---|---|---|
| bm25-raw | +0.000 [+0.000, +0.000] | +0.000 |
| bm25-norm | +0.000 [+0.000, +0.000] | +0.000 |
| bm25-light | +0.000 [+0.000, +0.000] | +0.000 |

**The correction is worth exactly nothing to BM25, and that is structural, not a lack of
statistical power.**
- **Why:** BM25 scores an unordered bag of tokens. `000 50` and `50 000` split into the same two
  tokens.
- **Token check:** in 1,570 of the 1,604 corrected chunks (1,575 with light stemming), the
  corrected and uncorrected Arabic have identical token multisets.
- **Ranking check:** when first measured on the original question set, every top-100 ranking was
  identical across the two corpora for all 53 questions, all three analyzers and both languages. On
  the re-verified set, every paired difference for the 43 numeric questions is exactly zero, with
  0 / 0 one-sided hits.
- **The 34 exceptions** are a separate, minor text defect: a number glued to the preceding Arabic
  word without a space (`و440`, `قدرها311`), where reversing the groups changes which digits are
  attached.

**Where the correction can matter:**
- **Dense retrieval.** Embedding models read token order, so `000 50` and `50 000` produce
  different vectors. Not yet run; needs the Colab T4 (`docs/colab.md`).
- **Phase 4 answer scoring.** An exact-match check of the answer "50,000" against a passage span
  `000 50` fails.

### BM25 on the numeric questions (n = 43)

| Config | Query–doc | recall@1 | recall@5 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| bm25-raw | en-en | 0.651 | 0.860 | 0.907 | 0.930 | 0.744 | 0.784 |
| bm25-raw | ar-ar | 0.465 | 0.605 | 0.628 | 0.674 | 0.516 | 0.543 |
| bm25-norm | en-en | 0.651 | 0.860 | 0.907 | 0.930 | 0.744 | 0.784 |
| bm25-norm | ar-ar | 0.488 | 0.628 | 0.651 | 0.698 | 0.541 | 0.568 |
| bm25-light | en-en | 0.605 | 0.814 | 0.837 | 0.860 | 0.684 | 0.722 |
| bm25-light | ar-ar | 0.512 | 0.628 | 0.744 | 0.814 | 0.569 | 0.609 |

| Config | Δ recall@5, EN − AR [95% CI] | Δ MRR@10 [95% CI] | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|
| bm25-raw | +0.256 [+0.093, +0.419] | +0.229 [+0.054, +0.402] | 14 / 3 | 0.013 |
| bm25-norm | +0.233 [+0.070, +0.395] | +0.203 [+0.026, +0.380] | 13 / 3 | 0.021 |
| bm25-light | +0.186 [+0.000, +0.349] | +0.116 [−0.065, +0.296] | 12 / 4 | 0.077 |

**Limits of this table:**
- **Sample.** 43 questions, all drawn from chunks containing large numbers, so it is not a sample of
  the corpus. The intervals are wide.
- **Unstable at this size.** On the earlier 41-question set, the stemmed recall@5 gap was +0.220
  with McNemar p = 0.035. Two added questions moved it to +0.186 with p = 0.077. A result that shifts
  that much when two questions are added should not be read beyond its direction.
- **Consistent with XQuAD in direction.** Light stemming narrows the Arabic gap here too: the MRR
  gap falls from +0.229 to +0.116, and its interval includes zero.
- **English stemming hurts.** Light stemming lowers English recall@5 on these questions (0.860 →
  0.814), which XQuAD did not show.
- **Not a headline.** The corpus-wide gap needs the full single-hop set.
