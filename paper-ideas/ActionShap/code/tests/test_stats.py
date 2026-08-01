"""Tests for the significance machinery."""

from __future__ import annotations

import numpy as np
import pytest

from actionshap.stats import compare_methods, holm_bonferroni


def test_holm_matches_worked_example():
    # Classic step-down: multipliers 4, 3, 2, 1 against sorted p-values.
    p = [0.01, 0.02, 0.03, 0.04]
    assert holm_bonferroni(p) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = rng.random(rng.integers(2, 20))
        adj = holm_bonferroni(p)
        assert np.all(adj <= 1.0) and np.all(adj >= p - 1e-12)
        # Sorting by raw p must not un-sort the corrected values.
        assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_holm_is_never_more_lenient_than_uncorrected():
    p = [0.001, 0.5, 0.9]
    assert np.all(holm_bonferroni(p) >= np.asarray(p))


def test_holm_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_bonferroni([0.5, 1.4])


def test_clear_separation_survives_correction():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    # Each method gets its own noise so the paired differences vary; a
    # constant offset would make the difference degenerate.
    scores = {
        "good": base + 2.0 + rng.normal(scale=0.5, size=200),
        "bad": base + rng.normal(scale=0.5, size=200),
        "worse": base - 2.0 + rng.normal(scale=0.5, size=200),
    }
    out = {(c.method_a, c.method_b): c for c in compare_methods(scores)}
    assert len(out) == 3
    assert all(c.significant for c in out.values())
    assert all(c.effect_label == "large" for c in out.values())


def test_identical_methods_are_not_significant():
    v = np.arange(50.0)
    (c,) = compare_methods({"a": v, "b": v.copy()})
    assert c.p_corrected == 1.0
    assert c.cohens_dz == 0.0
    assert not c.significant


def test_noise_only_is_mostly_not_significant():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(50):
        a, b = rng.normal(size=100), rng.normal(size=100)
        (c,) = compare_methods({"a": a, "b": b})
        hits += c.significant
    # Nominal 5% false-positive rate; allow generous slack for 50 trials.
    assert hits <= 8


def test_effect_size_sign_follows_the_difference():
    rng = np.random.default_rng(11)
    v = rng.normal(size=40)
    (c,) = compare_methods({"aaa": v + 1.0 + rng.normal(scale=0.3, size=40), "bbb": v})
    # Methods are sorted, so "aaa" is method_a and the difference is positive.
    assert c.method_a == "aaa"
    assert c.mean_difference > 0
    assert c.cohens_dz > 0


def test_constant_nonzero_difference_is_an_infinite_effect_not_a_zero_one():
    """Zero variance with a non-zero mean is a perfectly consistent effect.

    Reporting d_z = 0 here would state the exact opposite of the truth.
    """
    v = np.arange(40.0)
    (c,) = compare_methods({"aaa": v + 1.0, "bbb": v})
    assert c.mean_difference == pytest.approx(1.0)
    assert np.isposinf(c.cohens_dz)
    assert c.effect_label == "large"

    (c2,) = compare_methods({"aaa": v - 1.0, "bbb": v})
    assert np.isneginf(c2.cohens_dz)


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError, match="equally many units"):
        compare_methods({"a": np.ones(5), "b": np.ones(6)})


def test_single_method_is_rejected():
    with pytest.raises(ValueError, match="at least two methods"):
        compare_methods({"a": np.ones(5)})


def test_too_few_units_is_rejected():
    with pytest.raises(ValueError, match="at least 3 paired units"):
        compare_methods({"a": np.ones(2), "b": np.zeros(2)})


def test_normality_flag_fires_on_skewed_differences():
    rng = np.random.default_rng(7)
    base = rng.normal(size=300)
    (c,) = compare_methods({"a": base + rng.exponential(2.0, size=300), "b": base})
    assert c.normality_suspect
    assert not np.isnan(c.wilcoxon_p)
