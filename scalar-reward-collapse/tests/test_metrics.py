"""Tests for metric computations."""

import numpy as np
import pytest

from scalar_collapse.core.metrics import (
    alive_fraction,
    damage_onset,
    policy_entropy,
    proxy_true_divergence,
    sigma_count,
)


class TestPolicyEntropy:
    def test_uniform_two_arms(self):
        """Entropy of uniform distribution over 2 arms = ln(2)."""
        p = np.array([0.5, 0.5])
        assert np.isclose(policy_entropy(p), np.log(2))

    def test_uniform_n_arms(self):
        """Entropy of uniform distribution over n arms = ln(n)."""
        for n in [3, 5, 10]:
            p = np.ones(n) / n
            assert np.isclose(policy_entropy(p), np.log(n))

    def test_degenerate(self):
        """Entropy of degenerate distribution = 0."""
        p = np.array([1.0, 0.0])
        assert policy_entropy(p) == 0.0

    def test_near_degenerate(self):
        """Near-degenerate should have very low entropy."""
        p = np.array([0.99, 0.01])
        assert policy_entropy(p) < 0.1


class TestAliveFraction:
    def test_all_alive(self):
        assert alive_fraction(np.ones(100, dtype=bool)) == 1.0

    def test_all_dead(self):
        assert alive_fraction(np.zeros(100, dtype=bool)) == 0.0

    def test_half_alive(self):
        alive = np.zeros(100, dtype=bool)
        alive[:50] = True
        assert alive_fraction(alive) == 0.5

    def test_empty(self):
        assert alive_fraction(np.array([])) == 0.0


class TestProxyTrueDivergence:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert proxy_true_divergence(x, x) == 0.0

    def test_proxy_higher(self):
        proxy = np.array([2.0, 4.0, 6.0])
        true = np.array([1.0, 2.0, 3.0])
        assert proxy_true_divergence(proxy, true) == 6.0  # (1+2+3)

    def test_different_lengths(self):
        proxy = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0])
        # Should use min length
        assert proxy_true_divergence(proxy, true) == 0.0


class TestSigmaCount:
    def test_hand_computed(self):
        # proxy goes up, true goes down at steps 1->2 and 3->4
        proxy = np.array([1.0, 2.0, 3.0, 2.0, 3.0])
        true = np.array([5.0, 4.0, 3.0, 4.0, 3.0])
        # proxy_delta: [+1, +1, -1, +1]
        # true_delta:  [-1, -1, +1, -1]
        # sigma at: step 0->1 (proxy up, true down), 1->2, 3->4 = 3
        assert sigma_count(proxy, true) == 3

    def test_no_divergence(self):
        proxy = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert sigma_count(proxy, true) == 0

    def test_too_short(self):
        assert sigma_count(np.array([1.0]), np.array([1.0])) == 0


class TestDamageOnset:
    def test_crosses_floor(self):
        series = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        assert damage_onset(series, 0.5) == 3  # first below 0.5 at index 3

    def test_never_crosses(self):
        series = np.array([1.0, 0.9, 0.8, 0.7])
        assert damage_onset(series, 0.5) == -1

    def test_starts_below(self):
        series = np.array([0.3, 0.2, 0.1])
        assert damage_onset(series, 0.5) == 0
