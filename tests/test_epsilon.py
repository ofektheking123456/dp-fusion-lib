"""Tests for epsilon computation functions."""

import math
import pytest

from dp_fusion_lib import compute_epsilon_single_group, compute_dp_epsilon


class TestComputeEpsilonSingleGroup:
    """Tests for compute_epsilon_single_group function."""

    def test_basic_computation(self, sample_divergences, alpha, delta, beta):
        """Test basic epsilon computation returns expected structure."""
        result = compute_epsilon_single_group(
            divergences=sample_divergences,
            alpha=alpha,
            delta=delta,
            beta=beta
        )

        assert "empirical" in result
        assert "T" in result
        assert "theoretical" in result
        assert result["T"] == len(sample_divergences)
        assert result["empirical"] >= 0
        assert result["theoretical"] >= 0

    def test_empirical_less_than_theoretical(self, sample_divergences, alpha, delta, beta):
        """Test that empirical epsilon <= theoretical (worst-case)."""
        result = compute_epsilon_single_group(
            divergences=sample_divergences,
            alpha=alpha,
            delta=delta,
            beta=beta
        )

        # Empirical should be <= theoretical since theoretical is worst-case
        assert result["empirical"] <= result["theoretical"] + 1e-9

    def test_empty_divergences(self, alpha, delta, beta):
        """Test with empty divergence list."""
        result = compute_epsilon_single_group(
            divergences=[],
            alpha=alpha,
            delta=delta,
            beta=beta
        )

        assert result["T"] == 0
        # With no tokens, epsilon is just the log(1/delta) term
        expected = math.log(1.0 / delta) / (alpha - 1.0)
        assert abs(result["empirical"] - expected) < 1e-9

    def test_alpha_validation(self, sample_divergences, delta):
        """Test that alpha <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 1"):
            compute_epsilon_single_group(
                divergences=sample_divergences,
                alpha=1.0,
                delta=delta
            )

        with pytest.raises(ValueError, match="alpha must be > 1"):
            compute_epsilon_single_group(
                divergences=sample_divergences,
                alpha=0.5,
                delta=delta
            )

    def test_delta_validation(self, sample_divergences, alpha):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            compute_epsilon_single_group(
                divergences=sample_divergences,
                alpha=alpha,
                delta=0.0
            )

        with pytest.raises(ValueError, match="delta must be in"):
            compute_epsilon_single_group(
                divergences=sample_divergences,
                alpha=alpha,
                delta=1.0
            )

    def test_without_beta(self, sample_divergences, alpha, delta):
        """Test that theoretical is not computed when beta is not provided."""
        result = compute_epsilon_single_group(
            divergences=sample_divergences,
            alpha=alpha,
            delta=delta,
            beta=None
        )

        assert "empirical" in result
        assert "T" in result
        assert "theoretical" not in result

    def test_higher_divergences_higher_epsilon(self, alpha, delta, beta):
        """Test that higher divergences lead to higher epsilon."""
        low_div = [0.01, 0.02, 0.01]
        high_div = [0.1, 0.2, 0.1]

        result_low = compute_epsilon_single_group(low_div, alpha, delta, beta)
        result_high = compute_epsilon_single_group(high_div, alpha, delta, beta)

        assert result_low["empirical"] < result_high["empirical"]

    def test_more_tokens_higher_epsilon(self, alpha, delta, beta):
        """Test that more tokens lead to higher epsilon."""
        short_div = [0.05] * 10
        long_div = [0.05] * 100

        result_short = compute_epsilon_single_group(short_div, alpha, delta, beta)
        result_long = compute_epsilon_single_group(long_div, alpha, delta, beta)

        assert result_short["empirical"] < result_long["empirical"]


class TestComputeDpEpsilon:
    """Tests for compute_dp_epsilon function (multi-group)."""

    def test_single_group_global(self, alpha, delta):
        """Test global mode with single group."""
        divergences = {"GROUP1": [0.05, 0.06, 0.04]}

        epsilon = compute_dp_epsilon(
            divergences=divergences,
            alpha=alpha,
            delta=delta,
            mode="global"
        )

        assert isinstance(epsilon, float)
        assert epsilon > 0

    def test_multi_group_global(self, alpha, delta):
        """Test global mode with multiple groups."""
        divergences = {
            "GROUP1": [0.05, 0.06, 0.04],
            "GROUP2": [0.03, 0.08, 0.05]
        }

        epsilon = compute_dp_epsilon(
            divergences=divergences,
            alpha=alpha,
            delta=delta,
            mode="global"
        )

        assert isinstance(epsilon, float)
        assert epsilon > 0

    def test_per_group_mode(self, alpha, delta):
        """Test per_group mode returns dict."""
        divergences = {
            "GROUP1": [0.05, 0.06, 0.04],
            "GROUP2": [0.03, 0.08, 0.05]
        }

        epsilons = compute_dp_epsilon(
            divergences=divergences,
            alpha=alpha,
            delta=delta,
            mode="per_group"
        )

        assert isinstance(epsilons, dict)
        assert "GROUP1" in epsilons
        assert "GROUP2" in epsilons
        assert epsilons["GROUP1"] > 0
        assert epsilons["GROUP2"] > 0

    def test_public_group_ignored(self, alpha, delta):
        """Test that PUBLIC group is ignored."""
        divergences = {
            "PUBLIC": [0.99, 0.99, 0.99],  # Should be ignored
            "PRIVATE": [0.05, 0.06, 0.04]
        }

        epsilon = compute_dp_epsilon(
            divergences=divergences,
            alpha=alpha,
            delta=delta,
            mode="global"
        )

        # Should compute based on PRIVATE only
        assert isinstance(epsilon, float)
        assert epsilon > 0

    def test_no_private_groups_error(self, alpha, delta):
        """Test error when only PUBLIC group exists."""
        divergences = {"PUBLIC": [0.05, 0.06, 0.04]}

        with pytest.raises(ValueError, match="No private groups"):
            compute_dp_epsilon(
                divergences=divergences,
                alpha=alpha,
                delta=delta
            )

    def test_unequal_lengths_error(self, alpha, delta):
        """Test error when groups have different lengths."""
        divergences = {
            "GROUP1": [0.05, 0.06, 0.04],
            "GROUP2": [0.03, 0.08]  # Different length
        }

        with pytest.raises(ValueError, match="unequal lengths"):
            compute_dp_epsilon(
                divergences=divergences,
                alpha=alpha,
                delta=delta
            )

    def test_invalid_mode_error(self, alpha, delta):
        """Test error for invalid mode."""
        divergences = {"GROUP1": [0.05, 0.06, 0.04]}

        with pytest.raises(ValueError, match="mode must be"):
            compute_dp_epsilon(
                divergences=divergences,
                alpha=alpha,
                delta=delta,
                mode="invalid"
            )
