# Retrieval results (Phase 3)

Every number comes from `python scripts/evaluate.py --corpus <corpus>`, which writes
`data/results/<corpus>/retrieval.json`. XQuAD's dense, hybrid and reranker rows were run on a Colab T4
(`docs/colab.md`); their results file is committed at `7486ae2`. The UN corpus dense, hybrid and
reranker rows were run in one Kaggle GPU session (`docs/kaggle.md`).

## The headline: one reranker, two corpora, opposite conclusions

The same configuration — RRF over stemmed BM25 and `bge-m3`, reranked by `bge-reranker-v2-m3` over the
top 50 — is the best retriever in this project. Whether it removes the Arabic penalty depends entirely
on which corpus it is measured on.

| Corpus | Candidates per question | recall@5 EN | recall@5 AR | Δ recall@5, EN − AR [95% CI] | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|---|---|
| XQuAD | 240 passages | **1.000** | **1.000** | **+0.000 [+0.000, +0.000]** | 0 / 0 | 1.0 |
| UN corpus | 33,476 chunks | 0.938 | 0.859 | **+0.078 [+0.016, +0.141]** | 14 / 4 | 0.031 |

**Both numbers are correct, and only one of them is about the reranker.** On XQuAD the reranked
configuration reaches recall@5 = 1.000 in *both* languages. Every gold passage is in the top 5, so the
difference between the languages cannot be anything other than zero, whatever the reranker does to
Arabic. The measurement has no room left to show a gap. On the UN corpus, with 140 times as many
candidates and no ceiling, the same configuration leaves an Arabic penalty that a paired test rejects
zero for.

**What this means for the thesis.** The retrieval ceiling is lower in Arabic, and a reranker does not
remove that; XQuAD only makes it look removed. The gap is **+0.078 recall@5 after the strongest
retrieval configuration available here**, which is the ceiling generation inherits in Arabic.

**What it means for reading benchmarks.** A saturated benchmark reports the ceiling of its own
measurement, not the property being measured. Every paper that closes a gap on a small-corpus benchmark
is exposed to this, and the check costs nothing: report the absolute score alongside the difference. A
difference of zero next to a score of 1.000 is a ceiling, not a finding.

The ceiling caveat was written into the XQuAD section before the UN corpus was run, not added after it
disagreed.

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

**Question set used here:** frozen at 88 single-hop + 40 numeric = 128 questions, re-verified with
`qwen/qwen3.8-27b` after the original verifier was withdrawn (`docs/corpus.md` §3). The numeric subset
used for the corrected-vs-uncorrected comparison is the 43 questions whose English question or answer
contains a grouped number.

### Full table: same-language retrieval (n = 128)

| Config | Query–doc | recall@1 | recall@5 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| bm25-raw | en-en | 0.539 | 0.805 | 0.859 | 0.891 | 0.654 | 0.704 |
| bm25-raw | ar-ar | 0.391 | 0.562 | 0.602 | 0.664 | 0.458 | 0.493 |
| bm25-norm | en-en | 0.539 | 0.805 | 0.859 | 0.891 | 0.654 | 0.704 |
| bm25-norm | ar-ar | 0.438 | 0.586 | 0.617 | 0.680 | 0.495 | 0.525 |
| bm25-light | en-en | 0.539 | 0.742 | 0.789 | 0.852 | 0.626 | 0.665 |
| bm25-light | ar-ar | 0.461 | 0.648 | 0.695 | 0.742 | 0.533 | 0.572 |
| e5-base | en-en | 0.477 | 0.695 | 0.758 | 0.812 | 0.571 | 0.617 |
| e5-base | ar-ar | 0.398 | 0.562 | 0.656 | 0.695 | 0.474 | 0.517 |
| bge-m3 | en-en | 0.445 | 0.703 | 0.781 | 0.852 | 0.555 | 0.610 |
| bge-m3 | ar-ar | 0.398 | 0.617 | 0.695 | 0.766 | 0.483 | 0.533 |
| hybrid bm25-light + e5-base | en-en | 0.586 | 0.828 | 0.883 | 0.922 | 0.673 | 0.723 |
| hybrid bm25-light + e5-base | ar-ar | 0.469 | 0.656 | 0.766 | 0.805 | 0.555 | 0.605 |
| hybrid bm25-light + bge-m3 | en-en | 0.570 | 0.789 | 0.883 | 0.914 | 0.663 | 0.716 |
| hybrid bm25-light + bge-m3 | ar-ar | 0.461 | 0.648 | 0.727 | 0.812 | 0.549 | 0.592 |
| **reranked hybrid** | en-en | 0.828 | 0.938 | 0.945 | 0.953 | 0.871 | 0.890 |
| **reranked hybrid** | ar-ar | 0.750 | 0.859 | 0.898 | 0.914 | 0.795 | 0.820 |

The reranked row is `rerank:hybrid:bm25-light+bge-m3`: RRF over stemmed BM25 and bge-m3, then
`bge-reranker-v2-m3` over the top 50. **Nothing here is saturated.** The best English recall@5 is
0.938 and the best Arabic 0.859, against 1.000 in both languages on XQuAD, which is why this corpus
can still show a difference at all.

### Full table: paired EN − AR (n = 128)

| Config | Δ recall@5 [95% CI] | Δ MRR@10 [95% CI] | Δ nDCG@10 | hit@5 EN-only / AR-only | McNemar p |
|---|---|---|---|---|---|
| bm25-raw | +0.242 [+0.148, +0.336] | +0.195 [+0.105, +0.284] | +0.211 | 39 / 8 | 5.5e-06 |
| bm25-norm | +0.219 [+0.125, +0.312] | +0.159 [+0.065, +0.252] | +0.179 | 37 / 9 | 4.1e-05 |
| bm25-light | +0.094 [+0.000, +0.195] | +0.092 [+0.003, +0.181] | +0.093 | 27 / 15 | 0.0884 |
| e5-base | +0.133 [+0.062, +0.211] | +0.097 [+0.036, +0.161] | +0.099 | 21 / 4 | 0.0009 |
| bge-m3 | +0.086 [+0.016, +0.156] | +0.073 [+0.020, +0.126] | +0.077 | 17 / 6 | 0.0347 |
| hybrid bm25-light + e5-base | +0.172 [+0.086, +0.258] | +0.118 [+0.048, +0.190] | +0.118 | 28 / 6 | 0.0002 |
| hybrid bm25-light + bge-m3 | +0.141 [+0.062, +0.219] | +0.115 [+0.050, +0.182] | +0.124 | 23 / 5 | 0.0009 |
| **reranked hybrid** | +0.078 [+0.016, +0.141] | +0.076 [+0.016, +0.139] | +0.069 | 14 / 4 | 0.0309 |

- **Every configuration leaves an Arabic penalty**, including the best one. The tier-by-tier
  narrowing XQuAD showed happens here too — raw BM25 +0.242, stemmed +0.094, dense and hybrid
  +0.086 to +0.172, reranked +0.078 — but it stops at +0.078, not at zero.
- **The reranker is what makes the corpus tractable, and it does not equalise the languages.** It
  lifts Arabic recall@5 from 0.648 (its input hybrid) to 0.859 and recall@1 from 0.461 to 0.750, the
  largest gain of any tier. English gains too, from 0.789 to 0.938, but less (+0.149 against +0.211),
  which is why the gap narrows rather than holding.
- **Where the residual gap lives:** 14 questions are found in the top 5 in English but not Arabic,
  against 4 the other way.

### Cross-lingual retrieval (n = 128)

| Model | Query–doc | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| e5-base | en-ar | 0.094 | 0.219 | 0.273 | 0.147 |
| e5-base | ar-en | 0.328 | 0.516 | 0.555 | 0.403 |
| bge-m3 | en-ar | 0.367 | 0.562 | 0.586 | 0.447 |
| bge-m3 | ar-en | 0.406 | 0.555 | 0.617 | 0.465 |

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

**What "corrected" covers.** The correction handles whole digit groups. 56 decimal amounts
(`538.6 5` for 5,538.6) stay reversed in both corpus versions (`docs/corpus.md`, second result).
None of the 43 numeric questions has such an amount in its gold passage, so for these questions the
comparison is between fully corrected and uncorrected text.

### The correction is worth nothing to dense retrieval either: a null result

Dense retrieval was the case where the correction should have mattered. Embedding models read token
order, so `000 50` and `50 000` are different inputs and should produce different vectors. Measured on
the same 43 numeric questions, ar-ar, corrected − uncorrected:

| Config | Δ recall@5 [95% CI] | Δ MRR@10 [95% CI] | hit@5 corrected-only / uncorrected-only | McNemar p |
|---|---|---|---|---|
| bm25-raw | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 / 0 | 1.0 |
| bm25-norm | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 / 0 | 1.0 |
| bm25-light | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 / 0 | 1.0 |
| e5-base | +0.023 [+0.000, +0.070] | +0.001 [−0.034, +0.036] | 1 / 0 | 1.0 |
| bge-m3 | +0.000 [+0.000, +0.000] | +0.010 [−0.024, +0.045] | 0 / 0 | 1.0 |

**Null result. Across five retrieval configurations, correcting the reversed Arabic digit groups does
not measurably improve retrieval.** BM25's zero is structural and was predicted; the dense zero was
not. The expectation going in was that dense retrieval would be the place the correction paid off.

- **One question is not a finding.** e5-base's +0.023 is a single question moving from miss to hit
  (1 / 0 on McNemar, p = 1), and its MRR difference is +0.001. bge-m3 does not move a single question
  into or out of the top 5; only its ranking within the list shifts, by +0.010 MRR with an interval from
  −0.024 to +0.045.
- **The English sanity check holds everywhere:** every en-en difference is exactly zero on all five
  configurations, as it must be, since the correction never touches English text.
- **Where the correction does matter, and why it stays:** Phase 4 answer scoring. An exact-match check
  of "50,000" against a span reading `000 50` fails, and the Arabic gold answers themselves would carry
  garbled numbers (`docs/corpus.md` §3, finding 4, where digit grouping is the one Arabic-specific
  cause of exact-match rejection). The correction protects the answers, not the retrieval.
- **Worth measuring.** The reversal is a real, documented defect in the corpus text. Without this
  comparison the write-up would have said "the numbers were corrected", leaving the reader to assume
  retrieval improved.

### e5-base collapses cross-lingually on the UN corpus; bge-m3 does not

English question against Arabic passages (en-ar):

| Model | XQuAD recall@1 | UN recall@1 | UN recall@5 | UN MRR@10 |
|---|---|---|---|---|
| `multilingual-e5-base` | 0.823 | **0.094** | 0.219 | 0.147 |
| `bge-m3` | 0.854 | **0.367** | 0.562 | 0.447 |

The failure is directional. e5-base retrieves Arabic passages for Arabic queries at 0.398 recall@1 and
English passages for Arabic queries at 0.328, but Arabic passages for English queries at 0.094. Only the
en-ar direction collapses.

- **Part of the drop is the corpus,** which has 33,476 candidates against XQuAD's 240, and both models
  fall.
- **The model-specific part is the comparison on one corpus:** 0.094 against 0.367. Two models within
  three points of each other on XQuAD are 27 points apart on the UN corpus, one of them finding the
  right chunk first in fewer than one query in ten.
- **XQuAD saw a hint of it and understated it.** There, e5-base was the asymmetric model (ar-en 0.766
  vs en-ar 0.823) while bge-m3 was symmetric (0.849 / 0.854). A six-point asymmetry on the small corpus
  is a 73-point drop on the realistic one.
- **Consequence:** a cross-lingual retriever cannot be chosen on a saturated benchmark. On XQuAD these
  two models are interchangeable.

### Stemming: the XQuAD finding replicates, on a gap five times larger

Δ recall@5, EN − AR, same-language retrieval:

| Corpus | bm25-raw | bm25-light (Lucene-style Arabic stemmer) | Share of the gap removed |
|---|---|---|---|
| XQuAD | +0.047 | +0.019 | 60% |
| UN corpus | +0.242 | +0.094 | 61% |

The UN gap is five times the XQuAD gap, and light stemming removes the same share of it. **About 60% of
naive BM25's Arabic penalty is the tokenizer — clitics like و، ب، ال left attached to the word — not
the language.** That this holds at both scales is what makes it a property of the analyzer rather than
of one corpus.

- **The raw gap is unambiguous, the stemmed one is not.** On the UN corpus, bm25-raw is +0.242
  [+0.148, +0.336] with McNemar p = 5.5e-06; bm25-light is +0.094 [+0.000, +0.195] with p = 0.088. What
  survives stemming is no longer separable from zero at n = 128, which is a limit of this sample rather
  than evidence that stemming closes the gap.
- **Normalization alone is worth little,** here as on XQuAD: bm25-norm is +0.219 against raw's +0.242,
  and the whole of its effect is in Arabic (ar-ar recall@5 0.562 → 0.586, English unchanged).
- **The narrowing is two-sided, and the numeric subset's warning replicates.** Light stemming raises
  Arabic recall@5 from 0.562 to 0.648 (+0.086) and *lowers* English from 0.805 to 0.742 (−0.063). So
  58% of the gap it removes comes from Arabic improving and 42% from English getting worse. "English
  stemming hurts", first seen on the 43 numeric questions, holds on all 128. A stemmer chosen to close
  a language gap is partly closing it from the wrong end.

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

**Directional, not conclusive.** Read this table for the sign of the gaps, not their size.
- **Unstable at this size.** On the earlier 41-question set, the stemmed recall@5 gap was +0.220 with
  McNemar p = 0.035. Two added questions moved it to +0.186 with p = 0.077.
- **What a stable interval would need.**
  - Per question, the EN − AR difference has a standard deviation of 0.57–0.61, across the three
    analyzers and both metrics.
  - At n = 43 that makes a 95% interval about ±0.17–0.18 wide.
  - Holding that variability, ±0.10 needs roughly 130–145 numeric questions and ±0.05 roughly 500–570
    (normal approximation). The pilot will reach neither.
- **Sample.** All 43 questions are drawn from chunks containing large numbers, so this is not a
  sample of the corpus.
- **Consistent with XQuAD in direction.** Light stemming narrows the Arabic gap here too: the MRR
  gap falls from +0.229 to +0.116, and its interval includes zero.
- **English stemming hurts.** Light stemming lowers English recall@5 on these questions (0.860 →
  0.814), which XQuAD did not show.
- **Not a headline.** The corpus-wide gap needs the full single-hop set.
- **The reranked hybrid on this subset stays directional too.** Its EN − AR recall@5 difference is
  +0.023 [−0.093, +0.140], 4 English-only against 3 Arabic-only hits, McNemar p = 1. That is what 43
  questions support, and not evidence that the gap closes here.
  The corpus-wide reranked gap, on all 128 questions, is +0.078 [+0.016, +0.141], p = 0.031 (top of
  this document). Where the two disagree, the subset is the underpowered one.
