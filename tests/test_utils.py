"""Tests for utility functions."""

import pytest
import torch

from dp_fusion_lib import (
    compute_renyi_divergence_clipped_symmetric,
    find_lambda,
)


class TestRenyiDivergence:
    """Tests for Renyi divergence computation."""

    def test_identical_distributions(self):
        """Test that identical distributions have zero divergence."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])

        div = compute_renyi_divergence_clipped_symmetric(p, q, alpha=2.0)

        assert div.item() < 1e-6

    def test_symmetric(self):
        """Test that symmetric divergence gives same result for p,q and q,p."""
        p = torch.tensor([0.4, 0.3, 0.2, 0.1])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])

        div_pq = compute_renyi_divergence_clipped_symmetric(p, q, alpha=2.0)
        div_qp = compute_renyi_divergence_clipped_symmetric(q, p, alpha=2.0)

        # Symmetric divergence should be the same
        assert abs(div_pq.item() - div_qp.item()) < 1e-6

    def test_positive_divergence(self):
        """Test that divergence is non-negative."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.2, 0.7])

        div = compute_renyi_divergence_clipped_symmetric(p, q, alpha=2.0)

        assert div.item() >= 0

    def test_higher_alpha_different_result(self):
        """Test that different alpha values give different results."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.3, 0.4, 0.3])

        div_2 = compute_renyi_divergence_clipped_symmetric(p, q, alpha=2.0)
        div_5 = compute_renyi_divergence_clipped_symmetric(p, q, alpha=5.0)

        # Results should be different for different alpha
        assert abs(div_2.item() - div_5.item()) > 1e-6

    def test_alpha_validation(self):
        """Test that alpha <= 1 raises error."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="alpha must be > 1"):
            compute_renyi_divergence_clipped_symmetric(p, q, alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be > 1"):
            compute_renyi_divergence_clipped_symmetric(p, q, alpha=0.5)

    def test_batch_computation(self):
        """Test that batch computation works."""
        p = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        q = torch.tensor([[0.5, 0.5], [0.3, 0.7]])

        div = compute_renyi_divergence_clipped_symmetric(p, q, alpha=2.0)

        assert div.shape == (2,)
        assert div[0].item() < 1e-6  # Identical
        assert div[1].item() > 0  # Different


class TestFindLambda:
    """Tests for lambda search function."""

    def test_identical_distributions_lambda_1(self):
        """Test that identical distributions give lambda=1."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])

        lam, div = find_lambda(p, q, alpha=2.0, beta=0.1)

        assert lam == 1.0
        assert div < 1e-6

    def test_beta_zero_lambda_zero(self):
        """Test that beta=0 gives lambda=0."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.2, 0.7])

        lam, div = find_lambda(p, q, alpha=2.0, beta=0.0)

        assert lam == 0.0
        assert div == 0.0

    def test_lambda_in_range(self):
        """Test that lambda is in [0, 1]."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.2, 0.7])

        lam, div = find_lambda(p, q, alpha=2.0, beta=0.5)

        assert 0.0 <= lam <= 1.0

    def test_divergence_respects_bound(self):
        """Test that returned divergence is <= beta."""
        p = torch.tensor([0.6, 0.3, 0.1])
        q = torch.tensor([0.2, 0.3, 0.5])
        beta = 0.3

        lam, div = find_lambda(p, q, alpha=2.0, beta=beta)

        assert div <= beta + 1e-6  # Allow small numerical error

    def test_higher_beta_higher_lambda(self):
        """Test that higher beta allows higher lambda."""
        p = torch.tensor([0.8, 0.15, 0.05])
        q = torch.tensor([0.1, 0.2, 0.7])

        lam_low, _ = find_lambda(p, q, alpha=2.0, beta=0.1)
        lam_high, _ = find_lambda(p, q, alpha=2.0, beta=0.5)

        assert lam_low <= lam_high
