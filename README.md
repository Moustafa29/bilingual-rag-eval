# Bilingual RAG: measuring the retrieval ceiling in Arabic and English

_Structure only. Sections fill in as results are produced; every number will come from a committed
results file, with the command that produced it._

## The claim

On the same documents and the same questions, differing only in language: how much retrieval quality
is lost in Arabic, for which retriever types, and how much of the resulting answer-quality loss comes
from retrieval rather than generation.

## Headline results

**The Arabic penalty is in retrieval, not in generation.** On the same 128 questions over the same UN
documents, with the strongest retrieval configuration in this project (RRF over stemmed BM25 and bge-m3,
reranked by bge-reranker-v2-m3):

| Stage | English − Arabic | 95% CI |
|---|---|---|
| **Retrieval** (recall@5) | **+0.078** | [+0.016, +0.141], McNemar p = 0.031 |
| **Generation with retrieval held perfect** (oracle accuracy) | **+0.031** | [+0.000, +0.070] |

Give the model the right passage and it answers Arabic almost as well as English (0.953 against 0.984).
Make it find the passage first and the gap more than doubles. `docs/results.md`, `docs/generation.md`.

**The same reranker "closes" the gap on XQuAD and does not close it here**, because XQuAD reaches
recall@5 = 1.000 in both languages: at the ceiling the difference cannot be anything but zero. A zero
difference beside a perfect absolute score is a property of the benchmark, not of the retriever.

| Corpus | recall@5 EN | recall@5 AR | Δ EN − AR |
|---|---|---|---|
| XQuAD (240 passages) | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| UN corpus (33,476 chunks) | 0.938 | 0.859 | +0.078 [+0.016, +0.141] |

_Pending: the RAG condition's four-cell attribution and hallucination rate._

- What the Arabic digit-group correction is worth: nothing for BM25 (structural), nothing measurable
  for dense retrieval, and it matters for answer scoring (`docs/results.md`)

## Findings about Arabic that are not about retrieval quality

_To be written from `docs/corpus.md`:_
- Arabic-only truncation
- reversed digit groups
- prompt instructions followed less reliably in Arabic

## Evaluation fragility

_To be written from `docs/corpus.md` §3:_
- a threshold accepting a wrong answer
- a verifier swap and the shared "more than N" blind spot
- the three measured multi-hop attempts

## What doesn't work

_Written by the author._

## Data

- UN Parallel Corpus v1.0 subsample: 1,250 documents, 33,476 aligned chunks (`docs/corpus.md`)
- XQuAD English and Arabic: 240 passages, 1,190 questions

## Method

`docs/design.md` has the design and every change made while building; `docs/results.md` has the
retrieval tables. The neural retrieval steps need a GPU: `docs/kaggle.md` is the environment they run
in, `docs/colab.md` the fallback.

## Reproducing

_Commands for each phase, in order._

## Limitations

_Pending._

## License

Code: MIT. Data: UN Parallel Corpus v1.0 (acknowledge the United Nations; cite Ziemski,
Junczys-Dowmunt & Pouliquen 2016) and XQuAD (CC BY-SA 4.0).
