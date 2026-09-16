# Bilingual RAG: measuring the retrieval ceiling in Arabic and English

_Structure only. Sections fill in as results are produced; every number will come from a committed
results file, with the command that produced it._

## The claim

On the same documents and the same questions, differing only in language: how much retrieval quality
is lost in Arabic, for which retriever types, and how much of the resulting answer-quality loss comes
from retrieval rather than generation.

## Headline results

_Pending: the UN corpus question set, dense runs and generation._

- Retrieval: EN − AR gap by retriever tier (XQuAD; UN corpus)
- Generation: four-cell attribution and the retrieval/generation cost split, per language
- What the Arabic digit-group correction is worth: BM25, dense retrieval, answer scoring

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
