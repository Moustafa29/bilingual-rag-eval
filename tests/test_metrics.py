import math

import pytest

from rageval.eval.metrics import first_hit_ranks, question_metrics


def test_single_group_hit_at_rank_three():
    m = question_metrics(["x", "y", "gold", "z"], [["gold"]])
    assert m["recall@1"] == 0.0 and m["recall@5"] == 1.0
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(4))


def test_miss_scores_zero_everywhere():
    m = question_metrics(["x", "y"], [["gold"]])
    assert all(v == 0.0 for v in m.values())


def test_near_duplicate_counts_as_the_group():
    m = question_metrics(["copy", "x"], [["gold", "copy"]])
    assert m["recall@1"] == 1.0 and m["mrr@10"] == 1.0 and m["ndcg@10"] == 1.0


def test_duplicates_of_one_group_are_credited_once():
    assert first_hit_ranks(["gold", "copy", "b"], [["gold", "copy"], ["b"]]) == [1, 3]
    m = question_metrics(["gold", "copy", "b"], [["gold", "copy"], ["b"]])
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_bridge_needs_both_groups_for_all_recall():
    m = question_metrics(["a", "x", "y", "z", "w", "b"], [["a"], ["b"]])
    assert m["recall@5"] == 0.5 and m["all_recall@5"] == 0.0
    assert m["recall@10"] == 1.0 and m["all_recall@10"] == 1.0
    assert m["mrr@10"] == 1.0


def test_k_larger_than_ranking_and_empty_ranking():
    assert question_metrics(["gold"], [["gold"]])["recall@20"] == 1.0
    assert question_metrics([], [["gold"]])["ndcg@10"] == 0.0


def test_question_without_groups_is_rejected():
    with pytest.raises(ValueError):
        question_metrics(["a"], [])
