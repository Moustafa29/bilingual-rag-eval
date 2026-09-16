# Running dense retrieval and reranking on Kaggle Notebooks

The laptop runs BM25 but not the neural models: it has 7.7 GB of RAM with under 1 GB free.
`multilingual-e5-base`, `bge-m3` and `bge-reranker-v2-m3` need a GPU.

**Kaggle is the primary environment.** A GPU session runs up to 12 hours against a 30 hours/week
quota, so the whole UN corpus pipeline — rebuild, BM25, dense, hybrid, rerank, evaluate — fits in one
session instead of being cut in half. `docs/colab.md` is kept as the fallback; two Colab sessions were
lost mid-run.

## 0. Session settings, before running anything

In the notebook editor's right-hand panel:

| Setting | Value | Why |
|---|---|---|
| Accelerator | GPU (T4 or P100) | the dense and reranker steps |
| Internet | **On** | pip, the model downloads from Hugging Face, and the corpus rebuild fetching from OPUS |
| Persistence | Files only (or files and variables) | keeps `/kaggle/working` across session restarts |

Internet requires a phone-verified Kaggle account. With it off, the install cell fails and so does
everything after it.

**Watch the session clock** in the same panel. A session that hits 12 hours stops where it is. The
steps below are ordered so that the cheap ones come first and the long dense step is not repeated.

## 1. Setup

`/kaggle/working` is the writable directory and is what Kaggle keeps as notebook output. Everything
here assumes the repository sits inside it.

```
!git clone https://github.com/Moustafa29/bilingual-rag-eval.git /kaggle/working/bilingual-rag-eval
%cd /kaggle/working/bilingual-rag-eval
```

Every script writes to `data/` relative to the working directory, so stay in this directory for the
rest of the session. A `%cd` in one cell holds for later cells.

### Install against Kaggle's own torch

Do **not** install `requirements.txt`: it upgrades torch, which orphans the preinstalled torchvision
and torchaudio. That broke a Colab session, and the same failure exists here. Install
`requirements-colab.txt` (no torch, torchvision or torchaudio; `faiss-cpu` included) constrained to
whatever Kaggle ships:

```
# Pin Kaggle's own torch stack, then install everything else against it.
!pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt
!cat /tmp/torch-constraints.txt
!pip install -q -r requirements-colab.txt -c /tmp/torch-constraints.txt
!pip install -q -e . --no-deps
```

The file is named for Colab but contains no Colab-specific pin; it lists what the pipeline needs and
deliberately omits the torch stack, which is why the same file works on both.

**Kaggle's torch and CUDA versions are not the same as Colab's, and they change.** The constraints
file is read from the running image rather than assumed, so the cell adapts on its own. What it cannot
do is satisfy an impossible requirement:

- **If pip reports a conflict** (a pinned package needs a torch that Kaggle doesn't ship), it stops
  without installing rather than dragging torch with it. That is the intended behaviour. Print the
  conflict and the contents of `/tmp/torch-constraints.txt` and decide from there; do not remove the
  `-c` flag to make it pass.
- **If `sentence-transformers==6.0.1` is the package that conflicts**, that pin is the one to revisit,
  not the constraints. It is pinned because it decides retrieval results, so changing it means the
  XQuAD results in `docs/results.md` were produced under a different version and the change belongs in
  the write-up.

### Gate 1: does the environment work?

```
!python scripts/colab_check.py --require-cuda
```

Despite the name, the check is environment-agnostic. It imports torch, transformers, tokenizers,
sentence-transformers (both `SentenceTransformer` and `CrossEncoder`), faiss and the rest up front, and
fails with a clear message before any model download. It also fails if no GPU is visible, which on
Kaggle means the accelerator setting was left on None.

Nothing continues past a failure.

## 2. Data: rebuild the corpus, add the questions as a dataset

`data/` is git-ignored, so nothing but code arrives with the clone.

**Rebuild both corpus versions:**

```
!python scripts/build_unpc.py
```

- **Writes** `data/corpus/unpc/` (Arabic digit groups corrected) and `data/corpus/unpc_uncorrected/`
  (Arabic text as distributed), both from the committed manifest, so the selection is not re-decided.
- **Downloads** about 207 MB. Needs Internet on.
- **Takes** several minutes: it streams the alignment index before fetching documents.

**The frozen question set has to be uploaded,** because it cannot be regenerated without the LLM cache.
Kaggle has no drag-and-drop into `/kaggle/working`, so it arrives as a dataset:

1. In the notebook, **+ Add Input → Upload → New Dataset**, upload `data/questions/unpc_pilot.jsonl`
   from the laptop, and give the dataset a name such as `unpc-pilot-questions`.
2. It mounts read-only at `/kaggle/input/<dataset-slug>/unpc_pilot.jsonl`.
3. Copy it into place:

```
!mkdir -p data/questions
!cp /kaggle/input/unpc-pilot-questions/unpc_pilot.jsonl data/questions/
```

The dataset is reusable: later sessions add the same input instead of uploading again.

### Gate 2: is the data the frozen data?

```
!python scripts/colab_check.py --require-cuda --verify-data
```

- **Compares** the question file and both rebuilt chunk files against the fingerprints in
  `configs/pilot.yaml`, and fails if any is missing or differs.
- **Why the rebuild needs checking:** chunk boundaries depend on token counts. A different `tokenizers`
  version cuts chunks differently, and the questions' gold chunk ids would then point at different text
  with no error anywhere. `requirements-colab.txt` pins `tokenizers==0.23.2` for that reason; the check
  is what proves the pin held.
- **Line endings are ignored,** so a file written on Windows and one written here match.

Do not run retrieval past a failure here. The results would be silently wrong rather than missing.

## 3. Commands, in dependency order

Each step reads what the steps before it wrote.

```
# 1. BM25 on both corpus versions. The hybrid in step 3 reads the bm25-light runs, so both exist here.
!python scripts/run_retrieval.py --corpus unpc --configs bm25-raw,bm25-norm,bm25-light
!python scripts/run_retrieval.py --corpus unpc_uncorrected --configs bm25-raw,bm25-norm,bm25-light

# 2. Dense. The long step: 33,476 chunks per language, per model, per corpus version.
!python scripts/run_retrieval.py --corpus unpc --configs e5-base,bge-m3 --cross-lingual --device cuda
!python scripts/run_retrieval.py --corpus unpc_uncorrected --configs e5-base,bge-m3 --device cuda

# 3. Hybrids (RRF over steps 1 and 2). No GPU needed.
!python scripts/run_retrieval.py --corpus unpc --configs hybrid:bm25-light+e5-base,hybrid:bm25-light+bge-m3

# 4. Reranked hybrid: the retrieval condition Phase 4 generation uses.
!python scripts/run_retrieval.py --corpus unpc --configs rerank:hybrid:bm25-light+bge-m3 --device cuda

# 5. Evaluation.
!python scripts/evaluate.py --corpus unpc
!python scripts/evaluate.py --corpus unpc --subset numeric --baseline-corpus unpc_uncorrected
```

- **Step 4 is the one Phase 4 blocks on.** If the session is running short, it matters more than the
  `unpc_uncorrected` half of step 2, which only feeds the corrected-vs-uncorrected comparison.
- **Sanity check** in the last command: every en-en row of "unpc − unpc_uncorrected" must be exactly
  zero, because the correction does not touch English text. A non-zero English difference means the two
  runs differ for some other reason, and the Arabic difference can no longer be attributed to the
  correction.
- **Why dense is re-embedded for `unpc_uncorrected`:** the Arabic text differs and the embedding cache
  key covers the whole passage list, so English is re-encoded too.

## 4. Getting the results back

Unlike Colab, nothing needs downloading through the browser during the run: `/kaggle/working` is kept as
the notebook's output. **Save Version → Quick Save** writes the current files, and the Output tab of the
saved version lists them for download or for use as an input to another notebook.

What has to come back:

| Path | Needed for |
|---|---|
| `data/runs/unpc/` | every UN retrieval table; the reranked-hybrid RAG condition in Phase 4 |
| `data/runs/unpc_uncorrected/` | the dense corrected-vs-uncorrected comparison |
| `data/results/unpc/retrieval.json` | the UN corpus retrieval table |
| `data/results/unpc/retrieval_numeric.json` | the numeric-subset comparison |

The corpus itself does not need to come back: `build_unpc.py` rebuilds it, and the fingerprints in
`configs/pilot.yaml` prove the rebuild matched.

A single archive is easier to move than a directory tree, and keeps the output small:

```
!zip -qr /kaggle/working/unpc_session.zip data/runs/unpc data/runs/unpc_uncorrected data/results/unpc
!ls -la /kaggle/working/unpc_session.zip
```

Then Save Version, and download `unpc_session.zip` from the version's Output tab.

**If the session is about to expire mid-pipeline,** save a version anyway, and include the embedding
cache in what you keep:

```
!zip -qr /kaggle/working/unpc_partial.zip data/runs data/results data/embeddings
```

- **Runs already written stay valid.** A step that finished does not need re-running; the commands are
  separate for that reason.
- **Re-running a dense step does not re-embed.** Embeddings are cached under
  `data/embeddings/<corpus>/<model>/` by a fingerprint of the exact passage list, so the encoder is
  skipped when the same corpus is rebuilt and passes gate 2. That cache is what makes a second session
  cheap; without it the dense step starts from nothing.
- **Nothing else is skipped automatically.** `run_retrieval.py` overwrites the run files for the configs
  it is given, so re-run only the steps that did not finish.
