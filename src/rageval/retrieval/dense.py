"""Dense retrieval: bi-encoder embeddings and exact inner-product search.

A bi-encoder encodes queries and passages separately into one vector each (mean-pooled token
vectors), L2-normalized so the inner product equals cosine similarity. Passages are encoded once
and cached; a query is compared with every passage.

Search is exact (FAISS `IndexFlatIP`: one matrix multiply). Approximate indexes (IVF, HNSW) trade
recall for speed at millions of vectors; at tens of thousands they would only add their own
misses to what is being measured as the retriever's.

Asymmetric prefixes matter: the e5 family was trained with "query: " and "passage: " and loses
quality silently without them. bge-m3's dense mode uses no prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseModel:
    name: str
    hf_id: str
    query_prefix: str
    passage_prefix: str
    max_seq_length: int


MODELS = {
    "e5-base": DenseModel("e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: ", 512),
    "bge-m3": DenseModel("bge-m3", "BAAI/bge-m3", "", "", 8192),
}


def texts_fingerprint(texts: list[str], prefix: str, model: DenseModel) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([model.hf_id, model.max_seq_length, prefix]).encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()[:16]


class Encoder:
    """Wraps a SentenceTransformer; embeddings are cached on disk by content fingerprint."""

    def __init__(self, model: DenseModel, cache_dir: Path, batch_size: int = 32, device: str | None = None):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.device = device
        self._st = None

    def _load(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model.hf_id, device=self.device)
            self._st.max_seq_length = self.model.max_seq_length
        return self._st

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = self.model.query_prefix if kind == "query" else self.model.passage_prefix
        path = self.cache_dir / self.model.name / f"{kind}-{texts_fingerprint(texts, prefix, self.model)}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._load().encode(
            texts,
            prompt=prefix or None,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


def exact_search(passages: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k passage indices and scores per query by inner product (cosine on normalized vectors)."""
    import faiss

    index = faiss.IndexFlatIP(passages.shape[1])
    index.add(np.ascontiguousarray(passages, dtype=np.float32))
    scores, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), min(k, passages.shape[0]))
    return ids, scores
