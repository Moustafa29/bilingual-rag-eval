"""BM25 over a sparse term-document matrix.

For each query term t and document d:

    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

- idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)), the Lucene/Elasticsearch form. It is always
  positive. `rank_bm25`'s BM25Okapi uses log((N - df + 0.5) / (df + 0.5)), which goes negative
  for terms in more than half the documents and is then replaced by a floor; that makes common
  terms score arbitrarily. It also scores with a Python loop over documents per query term.
  This version is one sparse matrix product per query.
- tf saturation (k1): the fifth occurrence of a term adds much less than the first.
- length normalization (b): long documents contain more terms by chance and are discounted.

Query terms are deduplicated: a word repeated in a question counts once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse


class BM25:
    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        rows, cols, values = [], [], []
        lengths = np.zeros(len(documents), dtype=np.float64)
        for i, tokens in enumerate(documents):
            lengths[i] = len(tokens)
            for term, count in Counter(tokens).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                values.append(count)
        n_docs = len(documents)
        tf = sparse.csr_matrix((values, (rows, cols)), shape=(n_docs, len(self.vocab)), dtype=np.float64)
        df = np.diff(tf.tocsc().indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avgdl = lengths.mean() if n_docs else 0.0
        norm = k1 * (1 - b + b * lengths / avgdl) if avgdl else np.full(n_docs, k1)
        # Precompute the saturated term weight for every non-zero (doc, term) entry.
        tf = tf.tocoo()
        weights = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: Sequence[str]) -> np.ndarray:
        term_ids = sorted({self.vocab[t] for t in query if t in self.vocab})
        if not term_ids:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, term_ids] @ self.idf[term_ids]).ravel()

    def search(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        return top_k(scores, k, positive_only=True)


def top_k(scores: np.ndarray, k: int, positive_only: bool = False) -> list[tuple[int, float]]:
    """Indices and scores of the k highest scores; ties broken by lower index for determinism."""
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(candidates) == 0:
        return []
    order = np.lexsort((candidates, -scores[candidates]))[:k]
    return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]
