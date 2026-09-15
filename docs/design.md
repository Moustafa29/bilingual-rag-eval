# Design: Bilingual RAG, Measuring the Retrieval Ceiling

Status: Phase 1 draft. There is no code yet, and no numbers in this document come
from a run.

---

## 0. Changes from the brief

These are the places where I think the brief should change. Section 13 has the
decisions that are yours to make.

1. **"Nobody publishes the Arabic retrieval cost" is not true. Change the claim, not
   the project.** Arabic retrieval is already benchmarked: MIRACL has Arabic with
   native-speaker relevance labels, MMTEB covers Arabic, ArabicMTEB (Swan, NAACL
   Findings 2025) focuses on Arabic, and Belebele is parallel across 122 languages.
   An interviewer who knows the field will raise these. What they do not give you is
   a **controlled** comparison. MIRACL's Arabic and English sets use different
   documents and different questions, so a gap between them mixes up language,
   topic and difficulty. None of them follows the gap through to generation either.
   The claim this project can defend:
   > On the same documents and the same questions, differing only in language, how
   > much retrieval quality is lost in Arabic, for which retriever types, and how
   > much of the resulting answer-quality loss is caused by retrieval rather than
   > generation.
2. **Paired Wikipedia articles are not a parallel corpus.** Arabic and English
   articles on the same topic are written separately, with different content and
   different lengths. A question whose answer is in the English article may have no
   answer in the Arabic one. That breaks the "language is the only variable"
   design, so I rejected this option.
3. **BM25 in Arabic without an analyzer is a straw man.** Whitespace tokens in Arabic
   carry attached clitics: و (and), ب (with), ل (for), ال (the). So `والمنظمة` never
   matches `منظمة`. Unanalyzed BM25 would show a large Arabic gap caused by the
   tokenizer, not by anything we want to measure. You fixed exactly this at Globant
   with Elasticsearch's Arabic analyzer. BM25 is run twice: with raw tokens and with
   normalization plus light stemming. The difference between the two is a finding in
   itself.
4. **Synthetic questions favour BM25.** When an LLM writes a question from a
   passage, it copies the passage's words. BEIR and later work document this. Section
   5.5 describes how we limit it and how we report it.
5. **"Retrieval caps generation" is only mostly true, and the project should say so.**
   The missed/correct cell you describe is exactly the case where it doesn't hold,
   because the model answers from memory. The ceiling applies to *grounded*
   answering. Phase 4 adds two control runs, closed-book and oracle-context, so the
   ceiling is measured rather than asserted (§8).
6. **Use an exact index, not ANN.** An approximate index can miss results, and those
   misses would be mixed into the retriever measurement. At this corpus size, exact
   search is fast enough (§6.3). Qdrant isn't needed.

---

## 1. Claims to test

| # | Claim | Evidence that would support it | Evidence that would refute it |
|---|---|---|---|
| C1 | Retrieval limits grounded answer accuracy | Accuracy when the gold passage is missed is near the closed-book rate; accuracy with oracle context is far above the RAG system's accuracy | RAG accuracy ≈ oracle accuracy while recall is low (the model doesn't need the passage) |
| C2 | The retrieval ceiling is lower in Arabic | Paired recall@k(EN) − recall@k(AR) > 0, with a 95% CI excluding 0, for dense retrievers | CI includes 0 once BM25 has a fair Arabic analyzer |
| C3 | Most of the Arabic answer-accuracy gap comes from retrieval | Accuracy gap *given the gold passage was retrieved* is much smaller than the overall accuracy gap | The gap persists with oracle context, meaning generation is the bottleneck |
| C4 | Retrieval strategies differ more on multi-hop questions | The spread across configurations in "all supporting passages in top-k" is larger for multi-hop than single-hop | Spreads are similar |

Any of these can come out false. The README reports whichever result we get.

---

## 2. Prior work and what's new here

| Resource | Arabic | Parallel across languages | Human questions | Full-corpus retrieval | Measures generation |
|---|---|---|---|---|---|
| MIRACL | yes | no | yes (native) | yes | no |
| MMTEB / ArabicMTEB | yes | mostly no | mixed | yes | no |
| Belebele | yes | yes (FLORES) | yes | no (488 passages, reading comprehension) | no |
| XQuAD | yes | yes (professional translation) | yes | tiny (240 paragraphs) | extractive QA only |
| MLQA | yes | partly (only the answer sentence is aligned; the surrounding paragraph differs) | yes | no | extractive QA only |
| **This project** | yes | **yes, document-aligned** | synthetic, audited | yes | **yes, with retrieval/generation attribution** |

What's new is the combination: a parallel design, multi-hop questions, and the 2×2
attribution between retrieval and generation. None of the pieces is new on its own.
The README should say that plainly.

---

## 3. Corpus

### 3.1 Options

| | UN Parallel Corpus v1.0 | TED2020 (OPUS) | Paired Wikipedia |
|---|---|---|---|
| Parallel? | Yes, at document and sentence level | Yes, at sentence level within each talk | **No** |
| Translation quality | Professional UN translators; Arabic is an official UN language | Volunteer community translators | n/a |
| Register | Formal institutional MSA | Transcribed speech | Encyclopedic |
| Good for factual questions? | Yes: dates, bodies, figures, decisions | Weak; talks are narrative | Yes |
| Natural multi-hop links | Yes: documents cite each other by symbol (e.g. `A/RES/68/1`) | No | Yes (hyperlinks) |
| Licence | Free use; must credit the UN and cite Ziemski et al. (2016) | **CC BY-NC-ND 4.0** | CC BY-SA |
| Size | 799,276 documents; fully aligned subcorpus of 86,307 documents | ~4,000 talks | n/a |

### 3.2 Choice: the UN Parallel Corpus v1.0, fully aligned subcorpus

- **Language is the only variable.** Every English sentence has a professionally
  translated Arabic sentence, and the documents are aligned.
- **Arabic is a first-class translation target.** UN Arabic is careful, formal MSA,
  not a volunteer subtitle.
- **The licence allows what we need.** Terms from the UN DGACM corpus page: provided
  "without warranty of any kind", users must acknowledge the UN as the source and
  cite *Ziemski, Junczys-Dowmunt & Pouliquen (2016), The United Nations Parallel
  Corpus, LREC'16*. There is no non-commercial clause and no ban on derivatives, so
  we can publish chunk ids, questions and short excerpts.
- **Multi-hop structure comes built in.** Document symbols are regular strings that a
  regex can find, so building multi-hop questions is mechanical and repeatable
  (§5.4).

TED2020 was rejected for two reasons. The ND (no-derivatives) term makes publishing
chunked text and generated questions legally unclear. And talk transcripts don't
support factual, answerable questions well.

### 3.3 Known weaknesses of the UN corpus

These are costs of the choice, and the README will state them:

- **Boilerplate and near-duplicates.** Resolutions repeat preamble paragraphs word for
  word. If a question's answer appears in 40 near-identical chunks and only one is
  labelled gold, recall is underestimated. Mitigation is in §5.6.
- **Narrow domain.** Results describe formal institutional MSA. They don't tell us
  about dialect, news, or user-generated text.
- **Translationese.** Many Arabic UN documents are translations of English originals.
  Translated text tends to be more literal, which may make Arabic retrieval look
  *easier* than it would be on native Arabic writing. That biases the gap toward
  zero, so a gap we find is conservative.
- **Contamination.** UN parallel data is widely used to train multilingual models.
  The embedding models, and certainly the LLM, may have seen these documents. For
  retrieval this is hard to control and gets reported as a limitation. For generation
  it gets *measured*, through the closed-book run.
- **Dates.** The documents cover 1990–2014, so the LLM will know many of the answers.
  That is one more reason the closed-book control is required.

### 3.4 Secondary check: XQuAD, if you agree

XQuAD has 240 Wikipedia paragraphs and 1,190 human-written questions, professionally
translated into Arabic (CC BY-SA 4.0). It is too small to be the main corpus. It is
useful as a **check on the synthetic questions**: if the EN–AR gap on XQuAD's human
questions points the same way as the gap on our generated questions, the synthetic
pipeline isn't what's creating the result. It adds very little cost because no LLM is
needed to build questions. See question Q2.

---

## 4. Retrieval unit (chunking), principles only

Details and code come in Phase 2. These constraints are fixed now because the whole
design depends on them:

1. **Chunk boundaries must match across languages.** Chunk `k` in English must hold
   the same content as chunk `k` in Arabic. Otherwise the gold passage id means
   different things in the two languages. We chunk over *aligned sentence
   sequences*, so a boundary is placed after aligned sentence `i` in both languages
   at once.
2. **Size by the tighter language.** Arabic uses more subword tokens per sentence than
   English in most multilingual tokenizers. If chunks are sized on English, Arabic
   chunks can exceed an embedding model's maximum length (512 tokens for the e5
   family) and be **silently truncated**. That would create a fake Arabic penalty.
   Phase 2 will measure token counts per model, per language, per chunk, and check
   that no chunk is truncated. This is the same "check what actually reaches the
   model" lesson as the OCR coverage measurement in the receipt project.
3. **Follow document structure where it exists.** UN documents have numbered
   paragraphs, so chunks should prefer paragraph boundaries over fixed windows.
4. **No overlap, or overlap recorded in the labels.** Overlapping windows mean an
   answer can sit in two chunks, and both must count as relevant.

---

## 5. Building the question set

The goal is a pipeline where someone with the same cached LLM responses gets exactly
the same question set, and someone without the cache gets a statistically similar
one.

### 5.1 Reproducibility

- Seeded sampling at every step, with the seed stored in the config.
- Every LLM call goes through one client that caches the response on disk. The cache
  key is `sha256(model id + all generation params + full prompt)`. Temperature is 0
  wherever the provider allows it.
- Prompts are version-controlled files, not strings buried in code.
- The dataset is written as JSONL. Each record stores the prompt hashes that
  produced it, so every question can be traced to the exact calls behind it.
- Every filter logs its rejection reason. The README reports the funnel: *N
  passages sampled → M questions generated → K survived each filter*. This mirrors
  the receipt project's coverage table.

### 5.2 Passage sampling

- Sample documents stratified by document type, which the UN symbol prefix gives us
  (resolutions, reports, letters, meeting records).
- Exclude chunks that can't carry a question: tables, agenda lists, signature
  blocks, and chunks below a minimum length. Heuristics are documented and tested.

### 5.3 Single-hop questions

For each sampled chunk pair (EN, AR):

1. **Generate** a question, a short answer, and a **verbatim evidence quote** from the
   source-language chunk.
2. **Check the quote automatically.** It must be an exact substring of the chunk (after
   whitespace normalization). If not, reject. This catches hallucinated evidence
   without needing an LLM.
3. **Translate** the question and answer into the other language. Then check that the
   translated answer appears in the aligned target chunk, after Arabic orthographic
   normalization or fuzzy matching.
4. **Answerability check.** A separate call answers the question using *only* the gold
   chunk. The answer must match the reference.
5. **Decontextualization check.** Reject questions that point at the source ("in this
   document", "the above resolution", "المذكور أعلاه"), using a regex plus an LLM
   check. A retrieval question has to make sense without its passage.

**Balancing translation direction.** The questions are human-translated documents plus
*machine-translated* questions. Whichever language the question was machine-translated
into gets the less natural phrasing. To keep that from favouring one language, **half
the questions start from the English chunk and are translated to Arabic, and half go the
other way**. Direction is stored on each record, and results are reported by direction.
If the gap changes sign between directions, the effect comes from translation, not
language, and we report that instead.

### 5.4 Multi-hop questions

Two types, following HotpotQA's categories:

- **Bridge.** Chunk A mentions document X by symbol; chunk B is from document X. The
  question needs a fact from B, but can only be identified through A. Example shape:
  "What deadline was set in the report requested by the 2009 resolution on
  small-arms trafficking?" The **symbol must not appear in the question**, or BM25
  gets a free exact match.
- **Comparison.** Two chunks about entities of the same type (two sessions, two
  missions) share an attribute that can be compared.

Pairs are found by regex matching document symbols. The symbols are identical Latin
strings in the Arabic documents too, so the same links exist in both languages.

**Shortcut filter.** This is the check that makes a multi-hop question genuinely
multi-hop, adapted from MuSiQue. An LLM tries to answer with chunk A alone, then with
chunk B alone. If either one is enough, the question is rejected. Without this filter,
many "multi-hop" questions are single-hop questions with extra words.

### 5.5 Lexical overlap control

- The generation prompt tells the model to paraphrase and not reuse phrases from the
  chunk.
- For every question we compute the **analyzed token overlap** with its gold chunk
  (share of question terms present in the chunk after normalization and stemming).
- All retrieval tables are also reported **by overlap tertile**. If BM25 only wins on
  high-overlap questions, the table will show it.

### 5.6 Handling duplicates and missing labels

- **Near-duplicate groups.** Run MinHash over the English chunks. Chunks above a
  similarity threshold form a group, and retrieving *any* chunk in the group counts
  as retrieving the gold chunk. Because chunks are aligned, the groups apply to
  Arabic for free.
- **Pooled relevance judgments (TREC-style).** After Phase 3 runs, take the top-5 from
  every configuration in both languages. An LLM judge reviews every non-gold chunk in
  that pool to see if it also answers the question; any that do are added as
  relevant. Judgments are made on the English text only, so both languages get
  identical labels. Known weakness: this favours systems that were in the pool, which
  is all of ours, so the comparison stays fair.

### 5.7 Human audit

A seeded random sample of about 100 final questions, with both language versions, is
checked by a person against a written rubric: fluent? correct answer? self-contained?
really multi-hop? The README reports the acceptance rate. This is how "synthetic but
reproducible" becomes "synthetic, reproducible, and checked". See Q3.

### 5.8 Target size and why

Proposed: **about 600 single-hop and 300 multi-hop questions**, each in both languages.

Precision estimate, computed rather than measured: for a proportion near 0.7 with
n = 600, the standard error is √(0.7·0.3/600) ≈ 0.019, so a 95% CI of about ±3.7
points *per language*. The EN–AR difference is **paired** (same questions), which
removes question difficulty from the variance, so the CI on the difference will be
narrower than that. With n = 300 multi-hop questions, per-language CIs are about ±5
points, which is enough to see large differences between strategies but not small
ones.

---

## 6. Retrieval systems: how they work

### 6.1 BM25 (lexical)

BM25 scores a document *d* for query *q* by adding up, for each query term *t*:

```
IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
```

- **IDF**: rare terms count more (a match on "Rwanda" matters more than a match on "the").
- **tf with saturation (k1)**: the 5th occurrence of a term adds much less than the
  first. `rank_bm25` defaults to k1 = 1.5.
- **Length normalization (b)**: long documents contain more terms by chance, so they
  are penalized. b = 0.75 by default.

BM25 only matches **exact tokens**, so for Arabic the tokenizer decides everything.
Three variants in Arabic:

| Variant | What it does |
|---|---|
| `bm25-raw` | Unicode whitespace + punctuation split |
| `bm25-norm` | + orthographic normalization: remove diacritics (tashkeel) and tatweel; أ إ آ → ا; ة → ه; ى → ي; Arabic-Indic digits → Western |
| `bm25-light` | + light stemming: strip common prefixes (ال، وال، بال، كال، فال، لل، و) and suffixes (ها، ان، ات، ون، ين، يه، ية، ه، ة، ي), following Larkey et al.'s light stemmer, the same approach as Lucene's `ArabicStemmer` behind Elasticsearch's `arabic` analyzer |

English gets the equivalent treatment: lowercasing plus a Porter/Snowball stemmer. We
compare the best BM25 in each language, and report the others too.

Light stemming does not handle **broken plurals** (كتاب → كتب). Those are internal
vowel-pattern changes, not affixes. A root-based stemmer like ISRI handles them but
merges unrelated words. This is a known and real limitation of Arabic lexical
retrieval, and a likely entry for your "What doesn't work" section.

### 6.2 Dense retrieval (bi-encoder embeddings)

- A transformer **encoder** reads the text and produces one vector per token. These
  are **pooled** into a single vector (mean pooling, or the CLS token, depending on
  the model) and L2-normalized. Once normalized, cosine similarity is the same as the
  dot product.
- Queries and passages are encoded **separately**, which is why it's called a
  bi-encoder. So every passage can be encoded once, offline, and search is a nearest
  neighbour lookup.
- Models are trained **contrastively**: pull (query, relevant passage) vectors
  together, and push apart the other passages in the batch ("in-batch negatives").
  The model learns whatever notion of relevance its training pairs contain. Those
  pairs are overwhelmingly English, which is the basis for claim C2.
- **Asymmetric prefixes**: the e5 family expects `"query: "` and `"passage: "`
  prefixes. Leaving them out quietly lowers quality. The tests will check that
  they're applied.
- **Why Arabic can do worse**: less Arabic training data; worse tokenizer fertility
  (more subwords per word, so less meaning per token and more truncation); and
  morphology spreads one lemma over many surface forms.

Proposed models, at least two as the brief requires, final list confirmed in Phase 3:

| Model | Why |
|---|---|
| `intfloat/multilingual-e5-base` (or `-large` if the GPU allows) | Widely used, strong, contrastive training on multilingual data |
| `BAAI/bge-m3` | XLM-R-large backbone, 8,192-token context (removes the truncation question), different training recipe |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (optional) | Made **multilingual by distillation**: trained so a sentence's embedding in another language matches the English teacher's embedding (Reimers & Gurevych 2020, the TED2020 paper). An interesting contrast: it is explicitly trained to *copy* English behaviour into Arabic |
| An Arabic-specialized embedding model (optional) | Tests whether a native model closes the gap; the candidate and its licence are checked in Phase 3 |

**Cheap extra: a query × document language 2×2.** Since all embeddings already exist,
we can also run Arabic queries against English documents and the reverse, at almost
no cost:

| | EN docs | AR docs |
|---|---|---|
| EN query | baseline | document-side degradation |
| AR query | query-side degradation | full Arabic |

This splits the Arabic gap into "the model encodes Arabic *questions* badly" versus
"the model encodes Arabic *passages* badly", which call for different fixes. See Q5.

### 6.3 Indexing: exact search, and what ANN would add

- **Exact (flat) search**: compare the query vector to every passage vector. The cost
  is one matrix multiply. With roughly tens of thousands of chunks × 768–1024
  dimensions, that is a few million floats and effectively instant on a laptop. FAISS
  `IndexFlatIP` does exactly this.
- **ANN indexes** trade accuracy for speed once corpora reach millions of vectors:
  - **IVF** clusters vectors with k-means and searches only the `nprobe` clusters
    closest to the query. A relevant vector in a cluster that isn't searched is lost.
  - **HNSW** builds a layered proximity graph and walks it greedily from coarse to
    fine. The walk can stop at a local optimum.
- **Decision**: all reported numbers use exact search, so "retriever quality" isn't
  mixed with "index approximation". If time allows, one side table reports HNSW
  recall *relative to exact search*, to show what ANN would cost here.

### 6.4 Hybrid: reciprocal rank fusion (RRF)

BM25 scores have no upper bound and depend on the query. Cosine scores fall in [−1, 1]
and cluster tightly. **You can't add them meaningfully.** RRF ignores scores and uses
only rank positions:

```
RRF(d) = Σ over retrievers r of  1 / (k + rank_r(d))      k = 60
```

- A document ranked 1st by one retriever gets 1/61; one ranked 10th gets 1/70. The
  constant *k* flattens the curve so that no single retriever's top result dominates,
  and documents that rank well in *both* lists rise to the top.
- k = 60 comes from Cormack, Clarke & Büttcher (2009), who found it robust across
  collections. We use it without tuning, because tuning on the test questions would
  inflate the result.
- Fuse the top 100 from each retriever. A document missing from a list contributes 0
  from that list.
- Configurations: best-BM25 + each dense model.

### 6.5 Cross-encoder reranking

- A **cross-encoder** reads the query and passage *together* as one input
  (`[CLS] query [SEP] passage`), so every query token can attend to every passage
  token. It outputs one relevance score.
- It's more accurate than a bi-encoder because it sees how the two texts interact,
  rather than comparing two separately compressed vectors. But nothing can be
  precomputed: each (query, passage) pair needs its own forward pass. That's why it
  only **reranks** a shortlist, here the hybrid top-50, and never searches the corpus.
- Candidate: `BAAI/bge-reranker-v2-m3` (multilingual). Latency per query is recorded,
  because reranking is where the cost goes.

### 6.6 Configuration matrix

| Config | EN | AR |
|---|---|---|
| BM25 (raw / norm / light) | ✓ | ✓ |
| Dense ×2–4 models | ✓ | ✓ |
| Hybrid RRF (best BM25 + each dense) | ✓ | ✓ |
| Hybrid + reranker (best hybrid) | ✓ | ✓ |
| (optional) cross-lingual 2×2 for dense | ✓ | ✓ |

---

## 7. Metrics

### 7.1 Retrieval

Relevance comes from the gold chunk's near-duplicate group plus pooled judgments
(§5.6). Binary relevance throughout.

- **Hit@k / Recall@k**
  - Single-hop: 1 if any relevant chunk is in the top-k, else 0, averaged over
    questions. (With several relevant groups, recall@k = |relevant groups hit in
    top-k| / |relevant groups|. For single-hop the two definitions almost always
    agree.)
  - Multi-hop: **AllRecall@k** is 1 only if *every* supporting chunk is in the top-k.
    This is the metric that matters, since the model can't answer with one hop.
    Partial recall is also reported.
  - k ∈ {1, 5, 10, 20}. **k = 5 is the context size given to the LLM, so Recall@5 is
    the retrieval ceiling for Phase 4.**
- **MRR@10**: the average of 1/(rank of the first relevant chunk), with 0 if it isn't
  in the top 10. It rewards putting the answer *first*, which matters because LLMs
  make less use of evidence deep in the context.
- **nDCG@10**: DCG = Σᵢ relᵢ / log₂(i+1), divided by the best achievable DCG. Unlike
  MRR it credits *every* relevant chunk and discounts by position, which makes it the
  right summary for multi-hop, where two chunks both matter.

**Statistics.** Every EN–AR comparison is paired by question.
- 95% CIs from a **paired bootstrap** over questions (10,000 resamples, fixed seed).
- **McNemar's test** on hit/miss at k = 5 for the EN vs AR comparison of each
  configuration. It looks only at questions where the two languages *disagree*,
  which is the right test for paired binary outcomes.
- Many configurations are tested, so p-values are reported with Holm correction.
  Or, more simply, CIs are reported and differences whose CI crosses zero aren't
  called a finding.

### 7.2 Generation

- **Answer accuracy**, two ways:
  - *Normalized exact match and token F1* against the reference short answer. For
    Arabic this needs the same normalization as `bm25-norm`, plus stripping ال, plus
    digit normalization. Without it, "٢٠٠٩" ≠ "2009" and correct answers are scored
    wrong.
  - *LLM judge* for semantic equivalence ("the Security Council" vs "the Council").
    The judge sees the question, the reference and the candidate; it does **not** see
    retrieved passages, so it can't reward answers just for matching the context.
  - Judge validity: a human labels about 100 judge decisions, and we report agreement
    (Cohen's κ) **separately for Arabic and English**. A judge that is worse in Arabic
    would create a false generation gap, so this check isn't optional.
- **Faithfulness / hallucination**: an answer is *supported* if at least one passage
  in the context entails it, according to an LLM judge in NLI style. **Hallucination
  rate** = share of answers not supported by any retrieved passage (excluding explicit
  abstentions like "not in the documents").
- **Abstention rate**: how often the model says it can't answer. A model that abstains
  often scores low on hallucination for a bad reason, so both are always reported
  together.

---

## 8. Attributing errors to retrieval or generation

### 8.1 The 2×2, per question and per language

"Retrieved" means **the gold group, or all supporting chunks for multi-hop, was in the
context actually given to the LLM**, the top 5. Not "somewhere in the top 100".

| | Answer correct | Answer wrong |
|---|---|---|
| **Gold retrieved** | ✅ System worked | **Generation failure**: the model had the evidence and didn't use it |
| **Gold missed** | **Parametric answer**: the model answered from memory or a lucky guess | **Retrieval failure** (maybe generation too; not identifiable) |

Hallucination is recorded alongside: an answer can be correct *and* unsupported.
That's the parametric cell again, seen from the faithfulness side.

### 8.2 Two control runs that make the 2×2 interpretable

| Run | Context | What it measures |
|---|---|---|
| **Closed-book** | none | What the model knows without retrieval. If closed-book accuracy is high, the "missed/correct" cell is memory, not luck, and C1 is weaker for this corpus. |
| **Oracle** | gold chunk(s) + 4 random distractors, shuffled | The **generation ceiling**: the best accuracy the model can reach when retrieval is perfect. |
| RAG | top 5 from a config | The actual system |

With these, the accuracy loss breaks down into:
- `oracle − RAG` = cost of imperfect retrieval
- `100 − oracle` = cost of imperfect generation

**And the same breakdown per language** is what directly tests C3: does the Arabic
answer gap mostly come from `oracle − RAG` (retrieval) or from `100 − oracle`
(generation)?

For "retrieved/wrong" cases we also record where the gold passage sat in the context,
to check for "lost in the middle" position effects.

**Which retrieval configurations get generation runs**: not all, for cost. Proposed:
the best BM25, the best dense, and hybrid + reranker. Three systems plus two controls
= **5 runs × 2 languages × about 900 questions**. See §10.

---

## 9. Phase 5 evaluation plan (outline)

The agent chooses at each step: `search(query)`, `reformulate`, or `answer`, with a hop
limit. It is evaluated on the **same questions**, against the single-shot pipeline
using the **same retriever and the same LLM**, so the only thing that changes is
the control loop.

Reported: accuracy, AllRecall of evidence actually seen, hallucination rate, **LLM
calls per question, tokens per question, and wall-clock latency**. The headline is
accuracy *per unit of cost*. Prediction to test: the agent helps on multi-hop
(bridge questions need a second, different query) and does nothing or hurts on
single-hop. If it doesn't help on multi-hop either, that's the result.

---

## 10. Budget

Nothing is spent in Phase 1. Spending phases are **2** (question generation, filters,
pooled judgments), **4** (generation + judging) and **5** (agent). Token volumes below
are *estimates from the design*, not measurements. Dollar amounts depend on the
provider you choose (Q1), and I'll quote them from the provider's current pricing page
before any spending run.

| Phase | Calls (estimate) | Tokens per call (estimate) | Main driver |
|---|---|---|---|
| 2: single-hop | ~5 calls per candidate question (generate, translate, answerability, decontext, retry) × ~2 candidates per kept question × 600 | ~1–1.5k in, ~150 out | Filter rejection rate, unknown until run |
| 2: multi-hop | ~6 calls (+2 shortcut checks) × ~3 candidates per kept question × 300 | ~2k in | Shortcut filter will reject many |
| 2: pooled judgments | up to (configs × 5 − overlap) per question × 2 languages; judged once in English only | ~1k in | Overlap between configurations |
| 4: generation | 5 runs × 2 languages × 900 = 9,000 | ~2.5k in (5 chunks; Arabic uses more tokens) | Number of configs sent to the LLM |
| 4: judging | ~2 × 9,000 (correctness + support) | ~1–2.5k in | Could use a cheaper judge if κ holds |

Levers if the total is too high: a smaller question set (400 + 200), fewer configs
sent to generation, a local model for generation with the API used only for judging, or
answer checking via exact match/F1 first with the LLM judge only where EM fails.

**Caching**: every call is cached by prompt hash (§5.1), so re-running analysis never
costs anything again. Only prompt changes do.

---

## 11. Threats to validity (draft, to extend as we go)

| Threat | Direction of bias | Mitigation / how reported |
|---|---|---|
| Arabic side is often translated from English | Arabic looks easier, so the gap is underestimated | Stated; gap is a lower bound on UN text |
| Questions are machine-translated | Favours the source language | Balanced direction, reported per direction (§5.3) |
| Synthetic questions copy passage wording | Favours BM25 | Paraphrase prompt; overlap-stratified tables (§5.5) |
| Duplicate passages and missing labels | Recall underestimated for all systems | MinHash groups + pooled judgments (§5.6) |
| Silent truncation of Arabic chunks | False Arabic penalty for dense retrieval | Token-length audit per model (§4) |
| LLM judge less accurate in Arabic | False generation gap | Human κ per language (§7.2) |
| Pretraining contamination | Unknown for retrieval; measured for generation | Closed-book control (§8.2) |
| One corpus, one register | Limited generalization | Stated plainly; XQuAD check if accepted |

---

## 12. Test plan

As in the receipt project, fast unit tests for everything a number comes from:

- **Metrics**: recall@k, AllRecall@k, MRR@10, nDCG@10 on hand-worked rankings,
  including edge cases (no relevant items retrieved, several relevant items, ties,
  k larger than the list length).
- **RRF**: hand-computed fused ranking for two small lists, including documents that
  appear in only one list.
- **Arabic normalizer and stemmer**: a table of input → expected output cases,
  including the known broken-plural failure recorded as an expected limitation.
- **Chunk alignment**: EN and AR chunk counts and ids match for every document;
  no chunk exceeds any model's maximum token length.
- **Question filters**: evidence-substring check, decontextualization regex.
- **Cache**: same prompt → cache hit, no network call; changed params → cache miss.
- **Bootstrap / McNemar**: checked against a small known example.

---

## 13. Decisions (answered 2026-09-15)

| # | Decision | Effect on this design |
|---|---|---|
| 1 | Groq free tier for generation; a stronger paid API model for judging only; **hard cap $20 total**; cost quoted before every paid run | Phase 2 question generation runs on Groq ($0). The first paid step is judge calibration in Phase 3/4. |
| 2 | Include XQuAD | §3.4 is now in scope. It is the control on the synthetic-question pipeline. |
| 3 | The project owner (native Arabic speaker) does the human audit: 100 questions + 100 judge decisions | §5.7 and §7.2 are in scope. |
| 4 | **Pilot first: 150 questions**, full pipeline end to end, one real number, then decide what to scale | §5.8 target sizes are deferred. The pilot is 100 single-hop + 50 bridge multi-hop; comparison questions are deferred. CIs at n=150 are wide (about ±7 points per language at p≈0.7), so the pilot finds design flaws; it does not support a headline claim. |
| 5 | Include the query × document language 2×2 | §6.2 table is in scope for dense models. |
| 6 | Colab free T4 → `multilingual-e5-base` | See note below. |
| 7 | Package renamed `ragEval` → `rageval` | Done. |
| 8 | No full UN corpus download. Deterministic subsample of a few thousand documents; document ids committed; selection rule documented; data stored locally on D: | See `docs/corpus.md` (Phase 2). |

**Scope rule:** each phase stops at "pilot set, full pipeline, one real number". Anything
beyond that is flagged as a scale-up decision, not done by default.

### Changes made while building Phase 2

| Planned (§3–§5) | Built | Why |
|---|---|---|
| UN corpus from the UN's tar.gz archives | Same documents from OPUS zip files, fetched one document at a time with HTTP range requests | A tar.gz has no index; a subsample would still stream 3+ GB. The zip index makes it ~20 MB plus the documents themselves. See `docs/corpus.md`. |
| Stratified sampling by document type | One random content chunk per document, in seeded order | At 150 questions, strata of rare types (`ccw`, `iccd`) would hold 0–1 questions each. The type distribution of accepted questions is reported instead. |
| Translation step sees the target passage | Translation is **blind**: question and answer only. The target-language answer is then taken from the verifier's exact span in the target passage | Showing the passage would let the translated question borrow its wording, inflating lexical overlap and favouring BM25 in whichever language was translated into. |
| Answerability and shortcut checks by "an LLM" | Generator `openai/gpt-oss-120b`, verifier `qwen/qwen3.6-27b` (different model families), both Groq free tier. **The verifier was withdrawn on 2026-09-15** and replaced by `qwen/qwen3.8-27b`; every single-hop and numeric candidate was re-verified with it. | A model checking its own questions tends to agree with itself. Qwen3.6-27b was a Groq *preview* model, and the original note here said the cache would keep the pilot reproducible if it were withdrawn. That was too optimistic. It was withdrawn with no deprecation notice (HTTP 404 `model_not_found`, nothing on Groq's deprecations page), and the cache only reproduces runs whose inputs are unchanged. Any new candidate, fixed check or prompt edit needs the verifier again. **Anyone re-running this pipeline cannot reproduce the qwen3.6 verdicts; only the cached responses record them.** The re-verification measures how much the question set depends on which verifier happened to be available (`docs/corpus.md` §3). |
| Decontextualization check by regex + LLM | Regex only (English and Arabic phrase lists) | Saves one call per candidate on a 200K-tokens/day quota. The human audit measures how many context-dependent questions get through. |
| Multi-hop: bridge and comparison | Bridge only, 50 questions | Comparison questions are deferred until the pilot shows whether bridge construction works at all. |
| Lexical overlap computed at build time | Deferred to Phase 3 | It must use the same analyzers as BM25 (normalized, light-stemmed), which are Phase 3 code. |
| "A few thousand documents" | 1,250 documents (1,000 seed + 250 cited), 33,476 chunks; cut from a first build of 2,000 documents / 53,587 chunks | Embedding twice per model on a free T4 at 53k chunks was hours per iteration while the pipeline was still finding bugs, with no gain in validity. Cut before any questions were built on it. |
| Phase 3 waits for the UN questions | Phase 3 retrieval and metric code is built and tested against XQuAD while the UN questions generate | Groq's daily quota spreads question generation over days; XQuAD needs no API calls. |
| Answer acceptance by token F1 ≥ 0.5 | Verifier judgement of whether two answers state the same fact; F1 recorded only | F1 accepted a wrong answer (the Habitat Agenda vs. the UN Human Settlements Programme, F1 0.75). See `docs/corpus.md` §3. |
| Multi-hop: bridge and comparison questions | **None in the pilot.** Citation bridges 0/20 (three runs), comparison questions 0/20; both stopped at a pre-set threshold of 5/20 | Citation links exist only in metadata and describe cited documents by title; shared subject terms give topical, not parallel, pairs. Documented as negative results. |
| Numeric answers occur at their natural rate | A 40-question numeric group from chunks with corrected Arabic numbers | Needed for a confidence interval on what the digit-group correction is worth. |
| Third multi-hop source: comparisons from mission-financing resolutions | Probed, **not built**: 101 usable pairs (gate 60) but 2 of 30 sampled records wrong or not comparable (criterion ≤ 1), after three measurement fixes | Both rules were committed before the run they judged; the probe stopped rather than patch extraction until a sample passed. See `docs/corpus.md` §3. |

**What `multilingual-e5-large` would likely change** (expectation, not measured):
e5-large has ~560M parameters against base's ~280M, with the same 512-token limit and
the same training recipe family. Larger multilingual encoders usually gain more on
lower-resource languages than on English, because the extra capacity is spent where
the smaller model was underfitting. So the Arabic gap measured with base is likely an
**upper bound on the gap for that model family**. bge-m3 (XLM-R-large sized, and it runs
on a T4) partly covers this: if bge-m3 shows a much smaller gap than e5-base, model
capacity is a plausible explanation, and the README must say that e5-large was not
tested.

---

## Sources

- UN Parallel Corpus v1.0 terms: https://www.un.org/dgacm/en/content/uncorpus
- TED2020 licence (CC BY-NC-ND 4.0, via OPUS): https://opus.nlpl.eu/TED2020/en&de/v1/TED2020
- XQuAD: https://github.com/google-deepmind/xquad
- MLQA: https://github.com/facebookresearch/MLQA
- Belebele: https://github.com/facebookresearch/belebele
- Swan and ArabicMTEB: https://aclanthology.org/2025.findings-naacl.263.pdf
- MMTEB: https://arxiv.org/html/2502.13595v4
