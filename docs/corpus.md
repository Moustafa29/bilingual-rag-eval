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
