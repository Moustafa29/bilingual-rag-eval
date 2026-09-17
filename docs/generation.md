# Generation results (Phase 4)

Answering model `openai/gpt-oss-20b`, temperature 0, seed 20260915, k = 5 passages. Correctness is
judged by `qwen/qwen3.8-27b` (free tier, the same model as the question verifier — stated plainly; the
human audit of 100 judge decisions is the check on it). Exact match is reported beside every judge
number because the two disagree in a measurable, one-sided way (`docs/corpus.md` §3, finding 4).

All numbers come from `python scripts/score_generation.py --corpus unpc --correctness judge`, which
writes `data/results/unpc/generation_judge.json`.

## The two control conditions (n = 128 per language)

| Condition | What the model sees | Measures |
|---|---|---|
| Closed-book | the question only | what the model already knows, with no retrieval |
| Oracle | the gold passage among the 5 | the generation ceiling when retrieval is perfect |

| Condition | Language | Judge accuracy | Exact match | Abstention rate |
|---|---|---|---|---|
| Closed-book | English | 0.031 | 0.023 | 0.539 |
| Closed-book | Arabic | 0.031 | 0.008 | 0.570 |
| Oracle | English | **0.984** | 0.711 | 0.008 |
| Oracle | Arabic | **0.953** | 0.539 | 0.008 |

| Condition | EN − AR, judge [95% CI] | EN − AR, exact match [95% CI] |
|---|---|---|
| Closed-book | +0.000 [+0.000, +0.000] | +0.016 [+0.000, +0.039] |
| Oracle | +0.031 [+0.000, +0.070] | +0.172 [+0.063, +0.273] |

### What the controls establish

- **The model does not know these facts.** Closed-book accuracy is 0.031 in both languages: 4 questions
  of 128. It abstains on more than half. The corpus is UN documents from 2002–2013 — apportionment
  amounts, session numbers, committee names — and the answering model has no usable memory of them.
- **That makes the attribution clean.** Any question the RAG condition answers correctly was answered
  from the retrieved passages, not from memory, to within 3 percentage points. The four-cell attribution
  (retrieved/missed × correct/wrong) rests on this and it holds.
- **Generation is not the bottleneck when retrieval works.** Given the gold passage among five, the
  model answers 0.984 of English and 0.953 of Arabic questions correctly. The residual EN − AR gap in
  the oracle condition is +0.031 [+0.000, +0.070], which is the language penalty in *generation* with
  retrieval held perfect.
- **The Arabic penalty is therefore mostly upstream.** Retrieval's gap after the best configuration is
  +0.078 recall@5 (`docs/results.md`); generation's own gap with perfect retrieval is +0.031.
- **Exact match would tell a different story,** as it does everywhere in this project: it puts the
  oracle gap at +0.172, five times the judge's +0.031, because Arabic answers are rejected for phrasing
  and digit grouping. Judge correctness is the measure; exact match is reported, never used as the gate.

### Contamination check, applied as pre-registered

The same 60-question oracle subset answered by `qwen/qwen3.8-27b`, outside the gpt-oss family:

| Measure | Language | gpt-oss-20b | qwen3.8-27b | Paired difference [95% CI] |
|---|---|---|---|---|
| Judge | English | 0.983 | 0.983 | +0.000 [−0.050, +0.050] |
| Judge | Arabic | 0.967 | 0.967 | +0.000 [−0.050, +0.050] |
| Exact match | English | 0.700 | 0.817 | −0.117 [−0.200, −0.050] |
| Exact match | Arabic | 0.567 | 0.733 | −0.167 [−0.283, −0.050] |

**The gate did not trigger.** It asked whether `gpt-oss-20b` *exceeds* `qwen3.8-27b` by 10 points or
more with the interval excluding zero, on either measure, in either language. It is level on the judge
and behind on exact match, so the family effect is not driving the results and the main run proceeded.
Qwen leading on exact match by more than 10 points is not what the rule covers; the rule was written
one-sided before any contamination call and was not reinterpreted afterwards.

## Pre-registered prediction for the RAG condition

**Written before the RAG answers existed.** The git history of this file is the record.

If retrieval caps generation, the RAG condition's accuracy follows from three numbers already measured:
retrieval's recall@5 with the reranked hybrid, the oracle accuracy, and the closed-book accuracy.

    predicted accuracy = recall@5 × oracle accuracy + (1 − recall@5) × closed-book accuracy

| Language | recall@5 | Oracle | Closed-book | **Predicted RAG accuracy** |
|---|---|---|---|---|
| English | 0.938 | 0.984 | 0.031 | **0.925** |
| Arabic | 0.859 | 0.953 | 0.031 | **0.823** |

Predicted EN − AR gap: **+0.102**, which is larger than retrieval's +0.078 because the oracle gap
(+0.031) is added to it on the questions retrieval does find.

**What the model assumes,** each of which the measured result can contradict:
1. Whether retrieval finds the gold passage is independent of whether the answer would be right.
2. A question whose gold passage is missed is answered at the closed-book rate — that is, the other four
   passages contribute nothing.
3. k = 5 passages, so recall@5 is the relevant retrieval number.

**What a miss would mean:**
- **Measured above prediction:** distractor passages carry answerable information, or the judge accepts
  answers assembled from partial context. Assumption 2 is wrong.
- **Measured below prediction:** having the gold passage in context is not enough when four distractors
  are present, which the oracle condition cannot show because it contains the same five passages but
  ranked by a retriever that found the answer. Assumption 1 is wrong.
- **Arabic gap wider than +0.102:** retrieval and generation penalties compound rather than add.

## Status

The reranked-hybrid RAG condition is running: 256 answer calls, about 396,000 prompt tokens estimated
(the 3.5 characters-per-token rule runs about 8% high, so expect roughly 365,000). At 200,000 tokens per
day, and with the provider's counter running ahead of the local ledger (`docs/design.md` §10), it spans
more than one day and resumes at each limit.

Support judgements (hallucination rate) are judged on this condition only. The stemmed-BM25 RAG
condition is a bonus, run only if quota allows once this is complete.
