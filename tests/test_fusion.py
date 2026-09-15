import pytest

from rageval.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_documents_ranked_by_both_retrievers_rise():
    order = [doc for doc, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)]
    assert order == ["b", "a", "d", "c"]


def test_ties_are_broken_by_id_and_depth_limits_each_ranking():
    assert [d for d, _ in reciprocal_rank_fusion([["y"], ["x"]])] == ["x", "y"]
    assert [d for d, _ in reciprocal_rank_fusion([["a", "b", "c"]], depth=2)] == ["a", "b"]
