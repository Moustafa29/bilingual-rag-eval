# Running dense retrieval and reranking on Colab (free T4)

The laptop runs BM25 but not the neural models: it has 7.7 GB of RAM with under 1 GB free.
`multilingual-e5-base`, `bge-m3` and `bge-reranker-v2-m3` run on Colab.

## 1. Setup

Run these cells in order. Do **not** install `requirements.txt` on Colab: it upgrades torch and orphans
Colab's preinstalled torchvision and torchaudio, which broke an earlier session.

```
!git clone https://github.com/Moustafa29/bilingual-rag-eval.git
%cd bilingual-rag-eval
```

```
# Keep Colab's own torch, torchvision and torchaudio: record their versions and install against them.
# If anything below needs a different torch, pip stops with a conflict instead of breaking torchvision.
!pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt
!cat /tmp/torch-constraints.txt
!pip install -q -r requirements-colab.txt -c /tmp/torch-constraints.txt
!pip install -q -e . --no-deps
```

`requirements-colab.txt` includes `faiss-cpu`, which was missing from an earlier session.

```
# Fails immediately, with a clear message, if anything cannot be imported or no GPU is attached.
# Nothing is downloaded by this check. Do not continue past a failure.
!python scripts/colab_check.py --require-cuda
```

## 2. XQuAD session (done: results committed at `7486ae2`)

```
!python scripts/build_xquad.py
!python scripts/run_retrieval.py --corpus xquad --configs bm25-raw,bm25-norm,bm25-light
!python scripts/run_retrieval.py --corpus xquad --configs e5-base,bge-m3 --cross-lingual --device cuda
!python scripts/run_retrieval.py --corpus xquad --configs hybrid:bm25-light+e5-base,hybrid:bm25-light+bge-m3
!python scripts/run_retrieval.py --corpus xquad --configs rerank:hybrid:bm25-light+bge-m3 --device cuda
!python scripts/evaluate.py --corpus xquad
```

**Bring back:** `data/results/xquad/retrieval.json` and `data/runs/xquad/`.

## 3. UN corpus session

### Data: rebuild the corpus, upload only the question set

`data/` is git-ignored.

**Rebuild both corpus versions on Colab.** Don't upload them.

```
!python scripts/build_unpc.py
```

- **What it writes:** `data/corpus/unpc/` (Arabic digit groups corrected) and
  `data/corpus/unpc_uncorrected/` (Arabic text as distributed), both from the committed manifest.
- **Download:** about 207 MB, the alignment index plus the selected documents. That's faster over
  Colab's connection than uploading from the laptop.
- **Time:** several minutes, because it streams the alignment index to build the pair index and the
  sentence links.

**Upload only the frozen question set,** `data/questions/unpc_pilot.jsonl` (88 single-hop + 40 numeric),
keeping its path. It cannot be regenerated without the LLM cache.

**Verify before any GPU time:**

```
!python scripts/colab_check.py --require-cuda --verify-data
```

- **What it compares:** the question file and both rebuilt chunk files, against fingerprints committed
  in `configs/pilot.yaml`. It fails if a file is missing or differs.
- **Why the chunks matter:** chunk boundaries depend on token counts. A different `tokenizers` version
  would cut chunks differently, and the questions' gold chunk ids would then point at different text
  without any error. `requirements-colab.txt` pins `tokenizers==0.23.2` for this reason.
- **Line endings don't matter:** the fingerprints ignore Windows CRLF versus Linux LF.

Do not continue past a failure.

### Commands, in dependency order

Each step reads the runs written by the steps before it, so run them in this order.

```
# 1. BM25 on both versions. The hybrid in step 3 reads the bm25-light runs, so they must exist here too.
!python scripts/run_retrieval.py --corpus unpc --configs bm25-raw,bm25-norm,bm25-light
!python scripts/run_retrieval.py --corpus unpc_uncorrected --configs bm25-raw,bm25-norm,bm25-light

# 2. Dense retrieval. The longest step: 33,476 chunks per language, per model, per corpus version.
!python scripts/run_retrieval.py --corpus unpc --configs e5-base,bge-m3 --cross-lingual --device cuda
!python scripts/run_retrieval.py --corpus unpc_uncorrected --configs e5-base,bge-m3 --device cuda

# 3. Hybrids (RRF over steps 1 and 2). No GPU needed.
!python scripts/run_retrieval.py --corpus unpc --configs hybrid:bm25-light+e5-base,hybrid:bm25-light+bge-m3

# 4. Reranked hybrid: the retrieval condition Phase 4 generation uses first.
!python scripts/run_retrieval.py --corpus unpc --configs rerank:hybrid:bm25-light+bge-m3 --device cuda

# 5. Evaluation.
!python scripts/evaluate.py --corpus unpc
!python scripts/evaluate.py --corpus unpc --subset numeric --baseline-corpus unpc_uncorrected
```

- **Sanity check** in the last command: every en-en row of "unpc − unpc_uncorrected" must be exactly
  zero, because the correction does not touch English text. A non-zero English difference means the
  two runs differ for some other reason, so the Arabic difference cannot be attributed to the
  correction.
- **Why dense is re-embedded for `unpc_uncorrected`:** the Arabic text differs, and the embedding
  cache key covers the whole list of passages, so English is re-encoded too.

### Bring back

| Path | Needed for |
|---|---|
| `data/runs/unpc/` | every UN retrieval table; the reranked-hybrid RAG condition in Phase 4 |
| `data/runs/unpc_uncorrected/` | the dense corrected-vs-uncorrected comparison |
| `data/results/unpc/retrieval.json` | the UN corpus retrieval table |
| `data/results/unpc/retrieval_numeric.json` | the numeric-subset comparison |

```
!zip -qr unpc_session.zip data/runs/unpc data/runs/unpc_uncorrected data/results/unpc
from google.colab import files; files.download("unpc_session.zip")
```
