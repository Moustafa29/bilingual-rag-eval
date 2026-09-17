# Limitations

The README carries the five that most change how the results should be read. This is the full list.

**What none of them touches:** the retrieval/generation split is measured rather than inferred — the
closed-book control shows the answering model gets 3.1% of these questions right from memory, so the
oracle and RAG conditions are reading the passages. The XQuAD ceiling result is methodological and holds
for any saturated benchmark. The Arabic token cost, the truncation and the reversed digit groups are
measurements of the data itself, which no choice of model changes.

## Scope of the corpus

**One corpus, one domain, one register.** UN documents from 2002–2013: bureaucratic prose, numbered
paragraphs, apportionment tables, committee names. The Arabic is formal Modern Standard Arabic written by
professional UN translators. Nothing here transfers to dialect, to user-generated text, or to Arabic
written natively rather than translated. A retrieval gap measured on translationese is a gap on
translationese.

**The Arabic side is a translation of the English side.** That is what makes the comparison possible —
language is the only variable — and it also makes the task easier than reality, because translated Arabic
tracks English structure closely. The measured gap is plausibly a lower bound on what natively written
Arabic would show.

**The questions were written from the passages by an LLM.** They inherit passage vocabulary even with a
paraphrase instruction, which favours lexical retrieval; overlap-stratified reporting is in
`docs/design.md` §5.5. XQuAD, whose questions are human-written, is the control for this, and it points
the same way.

## Statistical power

**128 questions, so most intervals are wide.** The reranked gap, +0.078 [+0.016, +0.141], clears zero but
does not pin the size. Anything smaller than about 0.09 cannot be resolved at this n: the residual gap
after stemming, +0.094 [+0.000, +0.195], p = 0.088, is an example — not separable from zero, which is a
limit of the sample rather than evidence that stemming closes the gap.

**The 43-question numeric subset carries nothing on its own.** Its reranked row is +0.023 [−0.093,
+0.140], p = 1. Every numeric-subset result is labelled directional. To hold the observed per-question
variability at ±0.10 would need roughly 130–145 numeric questions, and ±0.05 roughly 500–570.

**The question set was frozen at 128 for quota reasons, not statistical ones.** The last 12 single-hop
questions would have narrowed intervals by about 4%, and the generator quota they needed was required for
answer generation.

## One of everything

One embedding pair (`multilingual-e5-base`, `bge-m3`), one reranker (`bge-reranker-v2-m3`), one answering
model (`gpt-oss-20b`), one judge, one value of k (5), one chunking scheme. No sweep over k, no second
reranker, no larger e5, no Arabic-specialised encoder.

**Where a single model behaves oddly, the cause is not diagnosed.** `e5-base` retrieves Arabic passages
for English queries at recall@1 0.094, against 0.398 for Arabic queries and 0.328 for English passages.
Query-prefix handling, normalization, and the model's cross-lingual alignment would each produce that
pattern, and nothing here distinguishes them. The finding is that the failure exists and is invisible on
XQuAD, not why it happens.

## Single-hop only

Three multi-hop constructions were measured and stopped, each against a rule fixed before its test
(`docs/corpus.md` §3). So "retrieval caps generation" is demonstrated for questions answerable from one
passage. Multi-passage questions are exactly where retrieval failures compound, where AllRecall@k would
separate hybrid retrieval from dense-only retrieval, and where a query-decomposition agent would earn its
place. None of that is measured here.

## The evaluation checks itself

**The questions are LLM-generated and LLM-verified.** The swap test that checked them — 1 flip in 68 —
detects disagreement between two models, not error shared by both. One shared error was found by hand:
both verifiers treat "more than 5,000" as a different fact from "5,000", against the prompt's explicit
instruction, in every case replayed. That bias shaped which numeric questions exist.

**The judge is the same model as the verifier** (`qwen/qwen3.8-27b` in both roles, both free tier). The
verifier decided which questions are valid and what their reference answers are; the judge decides whether
an answer matches that reference. A systematic error in one is the same error in the other, in the same
direction, and every internal consistency check still passes — including the swap test, which found
agreement partly *because* the bias is shared. Exact match is independent of the judge and disagrees with
it 65 times in 240, but it is broken in the opposite direction (it rejects correct answers and never
accepts wrong ones here), so the disagreement bounds nothing.

**Only an outside measurement closes this, and it has not been made.** The human audit of 100 questions
and 100 judge decisions is not started, so every accuracy number in this project is conditional on a judge
no person has checked — including oracle 0.984 / 0.953, which exact match would put at 0.711 / 0.539. The
$20 budget is held in reserve for a paid judge on a validation subset if the audit calls for one.

## Resources shaped the design

**Free-tier quota was a design constraint, not only a bill.** The corpus was cut from 53,587 to 33,476
chunks to fit a free GPU session; the question set was frozen at 128; contamination was measured on 60
questions rather than 128; the plan is one retrieval condition for generation rather than two; and the RAG
condition runs across several days in quota-limited slices. Arabic conditions cost about a quarter more
tokens than their English twins, so the more expensive language is also the more rationed one.

**Two reproducibility holes are not the author's to close.** The original verifier model
(`qwen/qwen3.6-27b`) was withdrawn mid-project with no deprecation notice, so those verdicts survive only
in the response cache and cannot be regenerated by anyone re-running the pipeline. And the provider's
daily token counter disagreed with the local usage ledger by up to 35% (`docs/design.md` §10), so
identical work takes an unpredictable number of days.

## Scale

**Exact search over 33,476 chunks.** No approximate index, so nothing here speaks to corpora of millions
of chunks, where ANN recall error, index parameters and their interaction with language would all matter.
Whether the Arabic gap grows, shrinks or stays flat with corpus size is untested.

## Not yet measured

- **Hallucination.** Support judgements are not started; they are judged on the RAG condition only. Any
  statement about grounding in Arabic is currently unmeasured.
- **The second retrieval condition** (RAG over stemmed BM25), which would show whether the generation gap
  tracks retrieval quality across retrievers rather than at one operating point.
- **The human audit**, above.
