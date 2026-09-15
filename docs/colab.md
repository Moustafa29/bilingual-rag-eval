# Running dense retrieval and reranking on Colab (free T4)

The laptop can run BM25 but not the neural models: it has 7.7 GB of RAM with under 1 GB free,
and the corpus build was already stopped twice for memory. `multilingual-e5-base`, `bge-m3` and
`bge-reranker-v2-m3` run on Colab.

## Setup (one cell)

```
!git clone https://github.com/Moustafa29/bilingual-rag-eval.git
%cd bilingual-rag-eval
!pip install -q -r requirements.txt && pip install -q -e . --no-deps
```

`data/` is git-ignored, so the corpus has to be rebuilt or uploaded:

- **XQuAD:** `!python scripts/build_xquad.py` (about 2 MB, no API calls).
- **UN corpus:** upload `data/corpus/unpc/chunks.jsonl` and `data/questions/unpc_pilot.jsonl`
  from the laptop, or rebuild with `!python scripts/build_unpc.py`, which reproduces the
  committed manifest and downloads about 250 MB.

## XQuAD retrieval table

Order matters: hybrid and rerank configs read the runs they combine.

```
!python scripts/run_retrieval.py --corpus xquad --configs bm25-raw,bm25-norm,bm25-light
!python scripts/run_retrieval.py --corpus xquad --configs e5-base,bge-m3 --cross-lingual --device cuda
!python scripts/run_retrieval.py --corpus xquad --configs hybrid:bm25-light+e5-base,hybrid:bm25-light+bge-m3
!python scripts/run_retrieval.py --corpus xquad --configs rerank:hybrid:bm25-light+bge-m3 --device cuda
!python scripts/evaluate.py --corpus xquad
```

- **Reranker cost:** 1,190 questions × 2 languages × 50 passages = 119,000 cross-encoder passes.
- **Embedding cache:** embeddings are stored under `data/embeddings/`, keyed by content, so
  re-running evaluation or adding a hybrid config does not re-encode anything.
- **What to bring back:** download `data/runs/xquad/` and `data/results/xquad/retrieval.json`.
