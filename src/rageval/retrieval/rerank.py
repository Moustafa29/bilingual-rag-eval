"""Cross-encoder reranking.

A cross-encoder reads the query and one passage together, `[CLS] query [SEP] passage`, so every
query token attends to every passage token, and outputs one relevance score. It sees the
interaction between the two texts instead of comparing two separately compressed vectors, which
makes it more accurate than a bi-encoder, but nothing can be precomputed: every (query, passage)
pair is a full forward pass. It therefore only reorders a shortlist from a cheaper retriever.
"""

from __future__ import annotations

from collections.abc import Sequence

RERANKER_ID = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, hf_id: str = RERANKER_ID, max_length: int = 1024, batch_size: int = 16, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(hf_id, max_length=max_length, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """`candidates` are (chunk_id, text); returns (chunk_id, score) best first."""
        if not candidates:
            return []
        scores = self.model.predict([(query, text) for _, text in candidates], batch_size=self.batch_size, show_progress_bar=False)
        return sorted(((cid, float(s)) for (cid, _), s in zip(candidates, scores)), key=lambda item: (-item[1], item[0]))
