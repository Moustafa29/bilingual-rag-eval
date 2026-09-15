import math

import numpy as np

from rageval.retrieval.bm25 import BM25, top_k

DOCS = [["council", "decided", "council"], ["assembly", "decided"], ["mission", "extended", "mandate", "council"]]


def expected(term_tf, doc_len, df, n=3, avgdl=3.0, k1=1.2, b=0.75):
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * term_tf * (k1 + 1) / (term_tf + k1 * (1 - b + b * doc_len / avgdl))


def test_scores_match_hand_computed_formula():
    index = BM25(DOCS)
    scores = index.scores(["council"])
    assert np.allclose(scores, [expected(2, 3, 2), 0.0, expected(1, 4, 2)])


def test_multi_term_scores_add_and_query_terms_are_deduplicated():
    index = BM25(DOCS)
    both = index.scores(["council", "decided"])
    assert np.allclose(both, index.scores(["council"]) + index.scores(["decided"]))
    assert np.allclose(index.scores(["council", "council"]), index.scores(["council"]))


def test_idf_is_positive_even_for_very_common_terms():
    index = BM25([["the", "a"], ["the", "b"], ["the", "c"]])
    assert index.scores(["the"]).min() > 0


def test_unknown_terms_score_zero_and_search_skips_zero_scores():
    index = BM25(DOCS)
    assert not index.scores(["nonexistent"]).any()
    assert index.search(["nonexistent"], 5) == []
    assert [i for i, _ in index.search(["decided"], 5)] == [1, 0]  # shorter doc 1 first


def test_top_k_breaks_ties_by_index():
    assert top_k(np.array([0.5, 0.9, 0.5, 0.9]), 3) == [(1, 0.9), (3, 0.9), (0, 0.5)]
