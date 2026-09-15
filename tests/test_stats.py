import pytest

from rageval.eval.stats import mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_hand_computed():
    # 3 questions hit only under A, 1 only under B: p = 2 * P(X <= 1), X ~ Bin(4, 0.5) = 2 * 5/16.
    a = [1, 1, 1, 0, 1, 0]
    b = [0, 0, 0, 1, 1, 0]
    result = mcnemar_exact(a, b)
    assert (result["a_only"], result["b_only"]) == (3, 1)
    assert result["p_value"] == pytest.approx(10 / 16)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def test_mcnemar_strong_effect_is_significant():
    assert mcnemar_exact([1] * 20, [0] * 20)["p_value"] < 1e-5


def test_bootstrap_constant_difference_has_zero_width():
    ci = paired_bootstrap_ci([1.0, 0.5, 0.2], [0.9, 0.4, 0.1])
    assert ci["mean_diff"] == pytest.approx(0.1)
    assert ci["ci_low"] == pytest.approx(0.1) and ci["ci_high"] == pytest.approx(0.1)


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    a, b = [1, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0]
    first, second = paired_bootstrap_ci(a, b, seed=3), paired_bootstrap_ci(a, b, seed=3)
    assert first == second
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [1])
