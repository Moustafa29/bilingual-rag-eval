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
  of 128. It abstains on more than half (0.539 English, 0.570 Arabic). The corpus is UN documents from
  2002–2013 — apportionment amounts, session numbers, committee names — and the answering model has no
  usable memory of them.
- **This was a real risk, and it is worth stating that it did not materialise.** UN documents are public,
  heavily mirrored, and plausibly in any web-scale training set, so a reader should ask whether the
  answering model is reciting them rather than reading the retrieved passages. If it were, the RAG
  condition would score well regardless of what retrieval returned, the "missed" cells of the attribution
  would fill with correct answers, and the whole retrieval-ceiling argument would collapse. The
  closed-book condition is the direct test of that, and it is what makes the argument safe rather than
  an assumption: **whatever this corpus contributed to pretraining, none of it is retrievable from the
  model by asking these questions.** Being in the training data and being answerable from memory are not
  the same thing; these questions ask for one specific figure or name out of a 33,476-chunk corpus of
  near-identical bureaucratic prose.
- **The bound this puts on the attribution:** a question the RAG condition answers correctly without the
  gold passage in context is either luck or memory, and memory is capped at 3.1%.
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

## The RAG condition: reranked hybrid, k = 5 (n = 128 per language)

Complete: 256 answers, all judged, support judged on this condition only.

| Language | Accuracy | Gold passage in context | Abstention | Unsupported answers (hallucination) |
|---|---|---|---|---|
| English | **0.867** | 0.938 | 0.016 | 0.016 (n = 126) |
| Arabic | **0.836** | 0.859 | 0.016 | **0.048** (n = 126) |

EN − AR accuracy: **+0.031 [−0.031, +0.094]** — the interval includes zero.

### The four cells

| Language | Retrieved & correct | Retrieved & wrong | Missed & correct | Missed & wrong |
|---|---|---|---|---|
| English | 111 | 9 | 0 | 8 |
| Arabic | 104 | 6 | 3 | 15 |

The retrieval rates (0.938 and 0.859) reproduce the retrieval run's recall@5 exactly, which is the check
that these contexts really are the reranked-hybrid run.

| Language | Accuracy when the gold passage is present | when it is missing | Oracle | Closed-book |
|---|---|---|---|---|
| English | 111/120 = 0.925 | 0/8 = 0.000 | 0.984 | 0.031 |
| Arabic | 104/110 = 0.945 | 3/18 = 0.167 | 0.953 | 0.031 |

### Cost decomposition

| Language | Retrieval cost (oracle − RAG) | Generation cost (1 − oracle) |
|---|---|---|
| English | 0.117 | 0.016 |
| Arabic | 0.117 | 0.047 |

**Retrieval costs both languages the same 0.117 in absolute accuracy.** What differs is the generation
cost, which is three times larger in Arabic (0.047 against 0.016) — and that is the same +0.031 oracle
gap seen from the other side.

## Prediction against outcome

The prediction was committed before any RAG answer existed (below, and in this file's git history).

| Language | Predicted | Measured | Miss |
|---|---|---|---|
| English | 0.925 | 0.867 | **−0.058** |
| Arabic | 0.823 | 0.836 | +0.012 |
| EN − AR gap | +0.102 | +0.031 | −0.071 |

**The prediction missed, in both directions, and the two misses have different causes — each one an
assumption written down beforehand.**

**English fell short: assumption 1 was wrong.** Whether retrieval succeeds is *not* independent of
whether the answer would be right. With the gold passage in context, the model scores 0.925 in the RAG
condition against 0.984 in the oracle condition — the same passage, the same k, six points apart. The
difference is what surrounds it: the oracle condition fills the context with passages drawn at random,
while the reranked hybrid fills it with the four chunks a cross-encoder ranked closest to the question.
Those are near-misses on the same topic, often the same committee and the same year, and they compete
with the gold passage in a way random distractors do not. **An oracle condition built from random
distractors overstates the generation ceiling.**

**Arabic overshot: assumption 2 was wrong.** A question whose gold passage is missed is not answered at
the closed-book rate. Arabic answered 3 of 18 such questions correctly (0.167) against a closed-book
rate of 0.031; English answered 0 of 8. The corpus contains near-duplicate chunks — the same figure
reported in a later session document, the same paragraph in an annex — so the fact is sometimes present
without the labelled gold chunk being there. The relevance groups already collapse near-duplicates for
*retrieval* scoring; this is the same phenomenon appearing in generation, where a passage outside the
group can still carry the fact.

**The net effect is that the end-to-end language gap is smaller than the retrieval gap**, +0.031 against
+0.078, and not distinguishable from zero at n = 128. The two errors point in opposite directions and
partly cancel: English loses accuracy on questions retrieval got right, Arabic gains accuracy on
questions retrieval got wrong.

**What this does not say.** It does not say the Arabic retrieval penalty is harmless. Retrieval still
costs Arabic 0.117 of absolute accuracy, and Arabic's 18 missed questions against English's 8 is the
reason its "missed" column is twice as large. It says that at k = 5 on this corpus, two effects offset
part of it, and that a ceiling computed as recall × oracle overstates what RAG achieves in English and
understates it in Arabic.

**Hallucination is the one place Arabic is clearly worse.** Answers judged unsupported by the passages
given: 0.048 in Arabic against 0.016 in English, both over 126 judged answers. Six unsupported Arabic
answers against two English. The numbers are small and no interval is reported for them here.

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
(the 3.5 characters-per-token rule runs about 8% high, so expect roughly 365,000). The free tier refills
at about 134 tokens per minute rather than resetting daily (`docs/design.md` §10), so the run stops at
the limit and resumes on a timer (`scripts/drain_quota.ps1`). The English half is answered; Arabic is
under way. Judging runs alongside, on answers that are already final.

**No RAG number is reported until all 256 answers exist and have been judged.** Answers already written
never change, so judging them early wastes nothing — but a partially answered condition has a different
retrieval-success rate from the finished one, and reporting it would be reporting a different experiment.
While the condition is incomplete the scorer writes to `--out data/results/unpc/partial_generation_judge.json`,
so the committed report holds only complete conditions.

Support judgements (hallucination rate) are judged on this condition only. The stemmed-BM25 RAG
condition is a bonus, run only if quota allows once this is complete.
