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

### Metric-design failure: a 0.5 token-overlap threshold accepted a wrong answer

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
- **All 5 of its rejections in the revalidation run were inspected, and all 5 are correct:**
  the Habitat case, a section letter "جيم" (C) mistranslated as the name "Jim", and three bridge
  answers where the verifier found a different fact from the one generated.
- **Not yet measured:** its false-rejection rate on correct answers. The human audit of 100
  judgement decisions covers that.

### Multi-hop questions: two negative results

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

| | Single-hop | Bridge |
|---|---|---|
| Candidates tried | | |
| Accepted | | |
| Accepted en→ar / ar→en | | |
| Largest rejection reason | | |
