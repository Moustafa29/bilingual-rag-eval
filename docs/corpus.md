# Corpus and question set (Phase 2)

Every number here comes from `data/corpus/*/stats.json` and
`data/questions/unpc_pilot_funnel.json`, produced by the scripts named in each section.
Empty cells mean the step has not been run yet.

## First result: Arabic chunks are truncated, English chunks are not

Chunks are aligned, so each one holds the same content in both languages. Their size is
counted in the XLM-R tokenizer shared by `multilingual-e5-base` and `bge-m3`.

| | Over e5's 512-token limit |
|---|---|
| Arabic only | **40** |
| English only | **0** |
| Both | 10 |

- **Across all 33,476 chunks,** Arabic needs 1.12× the tokens of the same English content
  (median).
- **In the 40 Arabic-only chunks,** the ratio is 1.44× (median). They are table-like runs of
  numbers, names and short items, where Arabic's extra token cost is largest.
- **What e5 sees:** Arabic passages cut off at 512 tokens, while their English versions fit in
  full (397–498 English tokens against 571–773 Arabic in the examples).
- **Why it matters:** an English-only benchmark can't show this failure. It's the tokenizer
  cost of Arabic, turned into lost passage text before retrieval even starts.
- **Scale:** 0.15% of chunks. It's a mechanism, not a headline effect size, and it stays
  measurable in Phase 3 because `bge-m3` (8,192-token limit) does not truncate these chunks.

## Second result: Arabic numbers are stored with reversed digit groups

In the UN corpus as distributed, **3,598 of 4,003 numbers (89.9%)** that English writes with
thousands separators appear in the aligned Arabic chunk with their digit groups in reverse
order. English `50,000` is stored as `000 50`, and `278,707` as `707 278`. It happens in every
year from 1993 to 2014. The cause is not verified.

| How the Arabic chunk writes the English number | As distributed | After correction |
|---|---|---|
| Reversed groups (`707 278`) | **3,598** | 1 |
| Forward groups (`278 707`) | 15 | 3,612 |
| With comma (`278,707`) | 175 | 175 |
| Joined (`278707`) | 22 | 22 |
| Not found in the Arabic chunk | 193 | 193 |

**Why it matters.** Left in place, this is an Arabic-only defect that would look like a language
effect:
- **BM25:** an Arabic query containing "50,000" cannot match `000 50`.
- **Embeddings:** the models read the digits in the wrong order.
- **Answer scoring:** numeric answers fail exact-match checks.

Any Arabic–English retrieval gap measured on the uncorrected text would include this storage
artifact, and it would be indistinguishable from a real language effect in the results table.

**Correction** (`rageval.corpus.numbers`):
- **Alignment-guided:** a reversed sequence is rewritten only when the aligned English chunk
  contains the same number in normal order. No Arabic number is changed on the strength of the
  Arabic text alone.
- **Formatting preserved:** digit script (Western or Arabic-Indic) and separator characters are
  kept; only the group order changes.
- **Known gap: decimal amounts.** The correction matches whole digit groups, so a decimal amount
  stays reversed. English "$5,538.6 million" appears in the Arabic passage as `538.6 5 ملايين دولار`.
  It surfaced as a rejected question (§3, evaluation fragility part 3). How many such amounts remain
  has not been counted.
- **Scale:** 3,600 corrections in 1,604 chunks.
  - Corrections exceed the reversed count by 2 because a number written once in English can
    appear more than once in its Arabic chunk.
  - The one remaining reversed match is `000 517 1` (1,517,000) in a chunk whose English says
    1,517: a different figure, correctly left alone.

**Uncorrected copy.** The text as distributed is kept as a separate corpus, `unpc_uncorrected`,
with identical chunk ids and English text. `scripts/evaluate.py --subset numeric
--baseline-corpus unpc_uncorrected` measures what the correction is worth on questions that
involve such numbers. The comparison needs a question set with enough of them; see §3.

**How the count was established.** An ad-hoc measurement first gave 3,405 of 3,798. A unit test
written for the committed version showed that the number pattern skipped numbers followed by a
clause comma ("$1,974,200, and …"). The fixed pattern finds 205 more numbers; the reversed share
barely moves (89.7% → 89.9%).

## 1. UN Parallel Corpus subsample

### Source and access

- **Corpus:** United Nations Parallel Corpus v1.0 (Ziemski, Junczys-Dowmunt & Pouliquen,
  LREC 2016). The paper describes the documents as public domain and lists the disclaimer as
  the only terms: no warranty, and the UN must be acknowledged as the source.
- **Distribution used:** OPUS, not the UN's own tar.gz archives.
  - A tar.gz file has no index, so extracting a few thousand documents means streaming through
    the whole 1.5–1.9 GB archive for each language.
  - OPUS publishes the same original TEI documents as zip files (`raw/en.zip`, `raw/ar.zip`),
    and a zip file ends with an index of every member's byte offset.
  - `rageval.corpus.remote_zip` reads that index once, about 20 MB, then fetches each selected
    document with two HTTP range requests.
- **Alignment:** `xml/ar-en.xml.gz` (199 MB) is the corpus's own sentence alignment. It uses the
  same paragraph:sentence ids as the TEI documents. SHA-256 is pinned in `configs/pilot.yaml`.

### Selection rule

Implemented in `rageval.corpus.unpc`; the output is committed as `manifests/unpc_docs.tsv`.

1. **Pair index.** 114,047 English–Arabic document pairs in the alignment file. For each pair
   the index records the number of alignment links, and the number that are one-to-one.
2. **Eligibility.** 35,979 pairs pass:
   - **At least 30 one-to-one links.** Shorter documents are mostly cover pages, notes verbales
     and agendas.
   - **At most 400 one-to-one links**, about the 90th percentile. Beyond that, a single long
     report would contribute hundreds of chunks.
   - **At least 85% of links are one-to-one.** A document whose alignment is mostly 1-0, 0-1
     or 2-1 is one where the English and Arabic versions don't say the same thing sentence
     by sentence, which breaks the "only the language differs" design.
3. **Seeds.** Eligible documents are sorted by `sha256("rageval-unpc-v1|" + doc_id)` and the
   first 1,000 are taken.
   - Hash order is a random order anyone can recompute, and it doesn't depend on library
     versions.
   - Adding or removing an unrelated document never reorders the rest.
4. **Cited documents.** An eligible document is added if a seed's English text references it.
   Additions follow the same hash order, capped at 250.
   - **Reference forms recognised:**
     - full symbols (`A/60/88`, `S/RES/1711(2006)`)
     - joint symbols split into both parts (`A/55/432-S/2000/921`)
     - General Assembly shorthand (`resolution 58/241`, sessions 40–69 only)
     - Security Council shorthand (`resolution 1325 (2000)`)
   - **Symbol-to-path rule:** lower-case the symbol, keep "/" as folders, and replace each
     ".", space, hyphen or parenthesis with "_". It reproduces the header symbol of all 1,250
     selected documents. (The first version failed on some documents; see "Bugs found while
     building".)
   - Without this step, bridge questions are nearly impossible: a random 1.3% sample of
     documents rarely contains both ends of a reference.

Distribution of pairs before eligibility (from `data/unpc/pairs.tsv`):

| Percentile | 10 | 25 | 50 | 75 | 90 | 95 | 99 |
|---|---|---|---|---|---|---|---|
| Alignment links | 13 | 27 | 69 | 251 | 484 | 738 | 1,839 |
| One-to-one links | 9 | 19 | 53 | 203 | 389 | 589 | 1,406 |
| One-to-one share | 0.556 | 0.71 | 0.857 | 0.94 | 0.981 | 1.0 | 1.0 |

### Chunking

Implemented in `rageval.corpus.chunking`.

- **Unit.** A chunk is a run of consecutive *beads*, where a bead is one alignment link.
  Boundaries always fall between beads, so chunk `k` of a document covers the same aligned
  sentences in both languages and has one id in both: `<doc_id>#<k>`.
- **Size.** Measured in XLM-R subword tokens, the tokenizer shared by `multilingual-e5-base`
  and `bge-m3`. The larger of the two languages decides.
  - A chunk closes at the end of an English paragraph once it holds 200 tokens.
  - It never exceeds 350 tokens in either language.
  - A single bead longer than 350 tokens becomes its own chunk and may be truncated by e5. It is
    counted below and excluded from question sampling.
- **Why paragraph ends.** UN documents number their paragraphs, and a paragraph is usually one
  point. Cutting mid-paragraph would split a fact from its subject.
- **Why count per language.** On the same aligned text, Arabic takes more tokens than English
  (ratio below). Sizing on English alone would push Arabic chunks past the 512-token limit, and
  e5 would silently drop the end of the Arabic passage only.

### Corpus size cut

**The cut:** the first build used 1,500 seed + 500 cited documents, giving 53,587 chunks. It was
cut to 1,000 + 250 before any questions were built on it.

**Why:** the retrieval corpus is embedded once per language, per dense model, on a free Colab
T4. At 53,587 chunks that is hours of GPU time per iteration, while the pipeline is still
turning up bugs. A larger distractor pool does not make the Arabic/English comparison more
valid; it only makes each iteration slower.

**Cost:** fewer bridge candidates. There are 233 now, below the cap of 300 (§3).

### Results (`python scripts/build_unpc.py`)

| | |
|---|---|
| Documents selected (seed + cited) | 1,250 (1,000 + 250) |
| Documents dropped (no links / alignment error / no chunks) | 0 |
| Chunks | 33,476 (median 21 per document, max 88; median 6 beads per chunk) |
| English tokens per chunk, p50 / p90 / p99 / max | 226 / 295 / 341 / 1,082 |
| Arabic tokens per chunk, p50 / p90 / p99 / max | 251 / 330 / 352 / 1,101 |
| Arabic ÷ English tokens per chunk | median 1.119, mean 1.139 |
| Chunks over e5's 512 tokens | 50 (0.15%): 40 Arabic only, 0 English only, 10 both |
| Symbol-to-path rule reproduces the document's own symbol | 1,250 / 1,250 |
| Content chunks (eligible for question writing) | 28,696 |
| Chunks with under 80% one-to-one beads (excluded from question writing) | 3,592 |

Document types among the 1,250 (symbol prefix): `a` 692 · `s` 176 · `e` 133 · `cedaw` 35 ·
`ccpr` 26 · `td` 20 · `npt` 19 · `dp` 17 · others 132.

- **Over-limit chunks are single long sentences.** All 50 are a single bead: one sentence too
  long to split, usually a table flattened into one line. The top of this document covers why
  the split by language matters.
- **They stay in the corpus.** These chunks remain as retrieval distractors and are excluded as
  question sources.

### Bugs found while building

All three were caught during Phase 2 and fixed before any questions were built.

1. **Quadratic fetch check.**
   - *Bug:* `build_unpc.py` called `fetch_all` inside a list-comprehension condition, so the
     file-existence check re-ran once per document. That is about 4.5 million filesystem
     checks and 3,000 thread pools for 1,500 documents.
   - *How found:* the process burned CPU and wrote no files for minutes; a `py-spy` stack
     dump showed thread pool #502.
   - *Effect:* time only, no wrong data.
2. **Memory.**
   - *Bug:* token counting loaded the tokenizer through `transformers`, pulling in a large
     library stack. On a 7.7 GB laptop with under 1 GB free, the build was stopped twice.
   - *Fix:* count with the `tokenizers` library directly. Counts are identical on all 480
     XQuAD passage versions (240 English, 240 Arabic).
3. **Symbol-to-path rule.**
   - *Bug:* the first rule ignored parentheses and hyphens, so it failed on 90 of 2,000
     documents, mostly Security Council resolutions. Security Council shorthand
     (`resolution 1325 (2000)`) and joint symbols (`A/55/432-S/2000/921`) weren't recognised.
   - *Effect:* citations to those documents were silently not followed, which shaped which
     documents were added as cited.
   - *Fix:* the rule was corrected and the selection rebuilt, which swapped 33 of the 500
     cited documents in the 2,000-document build.

## 2. XQuAD

- **Files:** `xquad.en.json` and `xquad.ar.json` from google-deepmind/xquad (CC BY-SA 4.0).
  - 48 articles, 240 paragraphs, 1,190 questions.
  - The Arabic file is a professional translation with identical question ids and paragraph
    order, checked by the loader.
- **Passages and gold:** one passage per paragraph. Each question's gold passage is its own
  paragraph.
- **Truncation:** 3 of 240 passages exceed e5's 512 tokens (`xquad/15/01`, `xquad/15/02`,
  `xquad/26/01`). They are the same three in both languages, so truncation doesn't favour
  either language.
- **Arabic ÷ English tokens per passage:** median 1.14.

## 3. Question set

See `rageval.questions.builder` for the step-by-step pipeline and the reason each step exists.

### Evaluation fragility: what changes the question set

Three changes to how questions are checked were measured, each on the same candidates:
1. A 0.5 token-overlap threshold let a wrong answer through.
2. Swapping the verifier model changed one verdict in 68.
3. The judgement that replaced the threshold rejects some correct answers.

What decides which questions exist is mainly the checking rule, not which model version was
available.

#### 1. A 0.5 token-overlap threshold accepted a wrong answer

The first pipeline accepted a question when the verifier's answer span, found in the
target-language passage, shared at least half its tokens with the blind translation of the
source answer (token F1 ≥ 0.5). That rule let a wrong answer into the question set.

| | |
|---|---|
| Question (en) | Which agenda is referenced in the General Assembly resolution that admitted UN-Habitat as a full member of the Inter-Agency Standing Committee? |
| Source answer (en) | **the Habitat Agenda** |
| Blind translation of the answer (ar) | برنامج المستوطنات البشرية ("Human Settlements *Programme*") |
| Verified span in the Arabic passage | برنامج الأمم المتحدة للمستوطنات البشرية ("**UN Human Settlements Programme**", the organization) |
| Token F1 | **0.75**, accepted |

**How it failed:**
1. The translator rendered "Agenda" as "Programme" (برنامج).
2. The verifier then found a real Programme in the passage, UN-Habitat itself.
3. The two Arabic strings share المستوطنات البشرية ("human settlements"). Overlap measures
   shared words, not shared meaning, so it passed a different entity.

**The threshold was fragile, not just unlucky.** Four of the fourteen single-hop questions
accepted in the same run scored 0.50, 0.55, 0.55 and 0.55. All four turned out to be correct:
- "CAD 42.2 million per year" against "CAN$ 42.2 million annually"
- "US$ 10,509,700" against "10 509 700 من دولارات الولايات المتحدة"
- "Human Resources Managers Network" against "Africa Public Sector Human Resource Managers' Network"
- "السيدة MOTOC" against "السيدة موتوك" (the same name in Latin and Arabic script)

They were correct by luck. The same score range contains correct answers written differently and
a wrong answer that happens to share words, and the threshold cannot tell them apart. That is
worse in Arabic, where multi-word institutional names share many tokens (الأمم المتحدة,
المستوطنات البشرية) across different entities.

**Replacement.** The verifier now judges whether the source answer and the verified target span
state the same fact (`prompts/judge_equivalence.txt`). F1 is still recorded on every question
(`answer_f1_vs_translation`) but no longer decides.
- **Validation on the known case:** the judgement rejects the Habitat answer. The prompt's
  examples do not include it.
- **Its rejections, inspected: 8 of 13 correct.** Five reject an answer stating the same fact (part 3
  below). An earlier version of this section said all 5 inspected rejections were correct. That held
  only for the 5 checked at the time: later runs added rejections that had not been inspected.
- **Not yet measured at scale:** the human audit of 100 judgement decisions.

#### 2. Swapping the verifier model changed one verdict in 68

**The verifier was withdrawn.** `qwen/qwen3.6-27b`, a Groq preview model that checked every
question, was withdrawn on 2026-09-15 with no deprecation notice: the API returned HTTP 404
`model_not_found`, and Groq's deprecations page has no entry for it. This is exactly the preview-model
risk `docs/design.md` had flagged.
- **Reproducibility cost:** anyone re-running this pipeline cannot reproduce the qwen3.6 verdicts.
  Only the cached responses record them.
- **Why the cache wasn't enough:** it reproduces a run only while nothing changes. Any new
  candidate, fixed check or edited prompt needs the verifier again.

**How the swap was measured.** Every recorded single-hop and numeric attempt was replayed with
`qwen/qwen3.8-27b`, keeping its exact chunk, translation direction and cached generator responses
(`scripts/reverify.py`).
- **Paired, not rebuilt:** the normal build would change later candidates after any flipped verdict,
  because it balances translation directions by acceptances so far.
- **Method check:** rejections decided before any verifier call must be unaffected. All 11 were
  identical.

| Kind | Reached the verifier | Accept → reject | Reject → accept | Rejection reason changed | Accepted: qwen3.6 → qwen3.8 |
|---|---|---|---|---|---|
| Single-hop | 21 | 0 | 1 | 1 | 17 → 18 |
| Numeric | 47 | 0 | 0 | 0 | 40 → 40 |
| **Total** | **68** | **0** | **1** | **1** | **57 → 58** |

- **The one reject → accept is span copying, not judgement.**
  - Question: "Which session of the Conference of the Parties made the request…", answer "its fifth
    session".
  - The old verifier answered الدورة الخامسة, which is not word-for-word in the passage (it says
    دورته الخامسة). The new verifier copied the exact wording, so the span check passed.
  - The answer is correct. The Arabic question renders "session" as جلسة ("meeting"), which is left
    for the human audit.
- **The changed rejection is the misread question** "National Automated Preventive Subcommittee"
  (see the earlier diagnosis). The old verifier found no answer; the new one answered "Subcommittee"
  and judged it a different fact. Rejected either way.
- **Identical responses:** on every traced candidate, the two models returned identical responses.

**Limit:** qwen3.6-27b and qwen3.8-27b are successive versions of the same 27B family. A verifier
from a different model family was not tested and could disagree far more. One verdict in 68 shows the
set is stable across this swap, not across verifiers in general.

**Paired with part 1:** a threshold let a wrong answer through; changing the model version barely
moved anything.

#### 3. The same-answer judgement rejects some correct answers

All 13 same-answer rejections made so far (after the single-hop rebuild with `qwen3.8-27b`) have
been inspected by hand.

| Verdict | Count | Cases |
|---|---|---|
| Correct rejection | 8 | the Habitat Agenda vs the UN Human Settlements Programme; a section letter "جيم" (C) mistranslated as the name "Jim"; three bridge answers where the verifier found a different fact; "150 000" where the passage states a range, "150,000 to 200,000"; the misread subcommittee name; "24/11" vs "42/11", two different resolution numbers |
| **Same fact, rejected** | **5** | see the three patterns below |

The five same-fact rejections fall into three patterns:

| Pattern | Count | Cases |
|---|---|---|
| A lower-bound qualifier treated as a different fact | 3 | "75 000" vs "over 75,000"; "2 500" vs "more than 2,500 languages"; "5,000" vs ما يزيد على 5 000 مرشح ("more than 5,000 candidates") |
| A context-implied detail treated as different | 1 | "before the end of 2005" vs قبل نهاية العام ("before the end of the year"), in a paragraph that has just said "during 2005" |
| A number still reversed in the Arabic passage | 1 | "$5,538.6 million" vs `538.6 5 ملايين دولار`, a decimal amount the digit-group correction does not handle (see the second result) |

- **Against the prompt:** the judgement prompt says extra qualifying words do not make answers
  different. The qualifier pattern appeared with both verifier models on every case replayed.
- **Direction of the error:** wrong rejections lose valid questions; they never admit a wrong answer.
  The numeric group still reached 40, but it is biased against facts stated as "more than N".
- **The reversed-number case cuts both ways.** The fact matched, but accepting it would have stored
  a garbled Arabic gold answer, so the rejection kept corrupted text out of the set, for the wrong
  reason.
- **Not fixed:** changing the prompt would change the question set again and need another
  re-verification. The error is documented, and the human audit of 100 judgement decisions measures
  its rate on a larger sample.

### Multi-hop questions: three measured attempts, none built

Three different constructions were tried on this corpus. Each failed for a different, measured
reason, and each was stopped by a rule fixed before its test. The pilot therefore has no multi-hop
questions.

| # | Construction | How it was tested | Result | Cause |
|---|---|---|---|---|
| 1 | Citation bridges: passage A cites document B, and the answer is in B | 20 candidates, three runs; stop if fewer than 5 of 20 are accepted | 0 valid of 20 | The citation link exists only in metadata: B's text names its own symbol in 3 of 233 candidates. Citations describe documents by title, so B alone answers. |
| 2 | Comparisons between two documents sharing a UNBIS subject term | 20 candidates; same stop rule | 0 of 20 | Shared subject terms are topical, not parallel: the generator declined 15 of 20 pairs, and the rest were contrived. |
| 3 | Deterministic comparisons of mission-financing appropriations | Probe with no LLM calls; gate of at least 60 pairs, plus at most 1 of 30 sampled records wrong | 101 pairs but 2 of 30 wrong; not built | Even template-parallel documents need per-template rules (liquidation budgets, support-account shares, partial years). The probe stopped rather than patch extraction until a sample passed. |

**What this rules out for the pilot:**
- **No multi-hop retrieval results.** AllRecall@k, where hybrid retrieval and reranking were expected
  to separate from dense-only retrieval, cannot be reported from this corpus.
- **No query-decomposition test in Phase 5.** It needs multi-subject questions.

The sections below give the evidence for each attempt.

**Verifier note:** attempts 1 and 2 were run with the withdrawn verifier `qwen/qwen3.6-27b`. Their
funnels are kept as recorded and were not re-verified: multi-hop is stopped, so re-running them would
change no decision. Attempt 3 used no LLM.

#### Attempts 1 and 2: generated two-passage questions

Both two-passage constructions were tested on 20 candidates against a threshold fixed in
advance: fewer than 5 accepted means stop. Both stopped. The pilot therefore has no multi-hop
questions.

**1. Citation bridges** (passage A cites document B; the question needs a fact from B,
identifying B only through A).

| Run | Accepted | Main rejection reasons |
|---|---|---|
| First | 0/20 | not answerable after translation 7, B alone answers 6 |
| + document-symbol headers | 1/20, and that one was the Habitat false accept | evidence not in passage 5, answer mismatch 5, B alone answers 5 |
| + strict shortcut rule and same-answer judgement | **0/20** | B alone answers 10, answers state different facts 4 |

- **The link exists only in metadata.** Passage B's text names its own document symbol in 3 of
  233 candidates, so without headers nothing in the text connects A's description to B's
  content. Headers fixed this: "not answerable" fell from 7 to 1.
- **A UN citation rarely describes the cited document beyond its title.** A is often an agenda
  line or a report title, so the only description available is B's own topic, and B alone then
  answers the question. That rejection rate doubled once any single-passage answer counted.
- **Weak answer spans.** When a link did exist, the generator's spans were poor: whole clauses,
  answers restating the question, fragments.

This is a property of the corpus, not a prompt bug.

**2. Comparison questions** (two documents indexed under the same UNBIS subject term; the
question names one subject from each and asks which one an attribute favours).

| Run | Accepted | Rejection reasons |
|---|---|---|
| Only run | **0/20** | generator declined the pair 15, one passage alone decides 3, option not named in question 1, context reference 1 |

- **A shared subject term is topical, not parallel.** Pairs under WESTERN SAHARA, CHEMICAL
  WEAPONS or LEBANON are about the same topic but rarely state the same attribute for two
  comparable subjects. One chunk records Iraq's destruction of chemical weapons; its pair lists
  what Syria must declare.
- **The comparisons the generator did produce were contrived.** UNDP's funding target against
  UNITAR's 1996 surplus; a 1995 house search against a 2009 justice reform. One passage plus
  common knowledge of dates decided them.

**What this costs the project.** The pilot has no multi-hop questions. AllRecall@k, where
hybrid retrieval and reranking were expected to separate from dense-only retrieval, cannot be
reported from this question set. Phase 5's query-decomposition test also needs another source of
multi-subject questions.

### Third multi-hop attempt: comparisons from mission-financing resolutions (probe, not built)

The two failures above are properties of the corpus: citations are metadata-only, and subject terms
are topical rather than parallel. General Assembly resolutions titled "Financing of the <mission>"
are parallel by construction. Each appropriates an amount for a stated budget period, so two
missions' appropriations for the same period give a comparison question with a deterministic
answer.

A probe with no LLM calls (`scripts/probe_financing.py`) counted usable pairs before anything was
built. Both decision rules were committed before the run they judged:
- **Gate:** at least 60 usable pairs. A usable pair is two missions with the identical budget
  period, each amount confirmed in the Arabic resolution, and each record used in at most one pair.
- **Acceptance** (added for run 3): a fresh seeded sample of 30 records may contain at most 1 wrong
  or non-comparable record.

| Run | Commit | Usable pairs | Sampled records wrong or not comparable | Errors found |
|---|---|---|---|---|
| 1 | `0f18ff4` | 113 | 3 of 10 | amounts in words skipped for a quoted figure; an apportionment read as an appropriation; truncated mission titles |
| 2 | `7c046c1` | 108 | 5 of 30 | support-account shares (one named only "Mission"); partial-year appropriations ("in addition to … already appropriated") |
| 3 | `297cf49` | **101** | **2 of 30** (fresh seed) | a liquidation budget for several combined forces; a support-account share worded "comprising … for the support account" after the period |

**Outcome: not built.** Run 3 passes the pair gate but fails the acceptance criterion: 2 of 30 sampled
records against at most 1. Each remaining error type could be patched with another rule. But
patching extraction after each sample until one passes is what a fixed acceptance rule exists to
prevent, so the probe stops at run 3.

**What it shows.**
- **Parallel structure isn't enough.** Even template-parallel UN documents don't yield clean
  deterministic comparisons without per-template rules: liquidation budgets, support-account
  shares, partial years, apportionments, amounts written in words.
- **The data exists, but checking it has a cost.** About 6–7% of extracted records remained wrong
  after three rounds of measurement fixes. For a question built from two records, that is roughly
  1 in 8 questions with a questionable gold answer.

### Numeric group (40 questions)

Single-hop questions answered by a grouped number, drawn from chunks where the Arabic digit groups
were corrected. The group exists so the corrected-vs-uncorrected comparison has enough questions for
a confidence interval.

| Run | Attempted | Accepted | Main rejections |
|---|---|---|---|
| First | 102 | 34 (20 en→ar, 14 ar→en), then stopped at the daily token limit | answer "not a grouped number" 42, context reference 16 |
| After the two fixes below | 54 | **40 (20 en→ar, 20 ar→en)** | context reference 6, answers state different facts 4 |

The first run exposed two check bugs, both affecting only Arabic-source questions:

1. **Space-grouped answers rejected.**
   - *Symptom:* all 42 "not a grouped number" rejections were ar→en.
   - *Cause:* the blind English translation kept the Arabic grouping ("75 000", "31 523 100
     dollars"), and the answer check accepted only commas.
   - *Cost:* those 42 rejections consumed most of that day's generator quota.
   - *Fix:* answers now accept comma or space grouping. The corpus audit pattern is unchanged, so
     the 3,598 of 4,003 figures stand.
2. **Arabic "according to the document" missed on the source side.**
   - *Symptom:* 12 of the 16 context-reference rejections were Arabic questions saying وفقًا للوثيقة,
     وفقًا للتقرير or حسب النص. They passed the Arabic filter and were caught only after
     translation into English.
   - *Fix:* the Arabic filter now catches them before translation.
   - *What it shows about the generator:* see "Prompt instructions are followed less reliably in
     Arabic" below.

### Prompt instructions are followed less reliably in Arabic

The generation prompt forbids referring to the source ("never refer to 'the passage', 'this
document', ... or similar"). In the numeric group's first run, the same model with the same
English-language prompt broke that rule far more often when writing in Arabic:

| Question written in | Attempts | Questions referring to "the document / text / report" |
|---|---|---|
| Arabic (ar→en) | 80 | **16 (20%)** |
| English (en→ar) | 22 | 0 (0%) |

Fisher exact test, two-sided: p = 0.020.

This is the same class of finding as the Arabic-only truncation and the digit-group reversal: the
pipeline behaves differently in Arabic for reasons that have nothing to do with retrieval, and
without a measurement the effect would land silently in the results.

**Limits:**
- **One model and one prompt:** `openai/gpt-oss-120b`, with instructions in English for both
  directions.
- **Small English side:** 22 English-source attempts.
- **Uneven samples:** attempts per direction were set by the pipeline's direction balancing, not by
  design.
- **Partly downstream of the filter bug:** 12 of the 16 were caught only after translation.

Whether this holds for Arabic-language instructions or other generators is not tested.

The combined run used 31 network calls, all on the verifier. The 42 recovered candidates already
had their generation and translation cached.

### Context-reference filter: measured, not assumed

- **Caught:** smoke question #5, "according to the mentioned decision" / وفقاً للقرار المذكور,
  once the phrase list was extended.
- **Still missed:** smoke question #10, "How many topics were discussed at the meeting?", which
  never says which meeting.
- **Audit plan:** both stay flagged in the audit export (`audit.flag_chunks`), and every
  filter rejection is exported with its question text. The audit measures misses and false
  rejections instead of assuming either.

**Candidates available** (`--dry-run`, no API calls):
- **Single-hop:** 250, the cap.
- **Bridge:** 233, below the cap of 300. Filling 50 bridge questions needs about 21% of them
  accepted; 30 needs about 13%.

**Decision rule for bridge**, set before the smoke run:

| Smoke run accepts | Action |
|---|---|
| ≥ 15 of 20 | Keep 50 |
| 5–14 of 20 | Cut to 30, keep multi-hop |
| < 5 of 20 | Stop: the generation approach is the problem, not the count |

**Before running:** put `GROQ_API_KEY=...` in an untracked `.env` file, then run a smoke
test on 20 candidates of each kind:

```
python scripts/build_questions.py --kind single --max-candidates 20
python scripts/build_questions.py --kind bridge --max-candidates 20
```

Responses are cached, so the full run reuses these 40 candidates at no cost.

### Results (`python scripts/build_questions.py`)

Built with generator `openai/gpt-oss-120b` and verifier `qwen/qwen3.8-27b`. Multi-hop kinds are
covered under "three measured attempts" above.

| | Single-hop | Numeric |
|---|---|---|
| Candidates tried | 111 | 54 |
| Accepted | **82** (target 100; stopped at the daily token limit, resumable) | **40** (target reached) |
| Accepted en→ar / ar→en | 45 / 37 | 20 / 20 |
| Rejections | context reference 8, not answerable after translation 7, answers state different facts 5, generator skipped 4, verified answer not in passage 3, evidence not in passage 2 | context reference 6, answers state different facts 4, not answerable after translation 2, verified answer not in passage 1, evidence not in passage 1 |

- **The numeric group is identical to its earlier build** with the withdrawn verifier: re-verification
  changed no verdict in it.
- **Audit flags behave as intended:**
  - Smoke-run #5 ("according to the mentioned decision") is now rejected by the context filter.
  - #10 ("How many topics were discussed at the meeting?") is still accepted, a filter miss.
  - Both appear in the audit export.
