"""Pytest configuration and fixtures for dp-fusion-lib tests."""

import pytest


@pytest.fixture
def sample_divergences():
    """Sample divergence values for testing epsilon computation."""
    return [0.05, 0.08, 0.03, 0.06, 0.04, 0.07, 0.02, 0.05, 0.09, 0.04]


@pytest.fixture
def alpha():
    """Standard Renyi order for tests."""
    return 2.0


@pytest.fixture
def beta():
    """Standard beta value for tests."""
    return 0.1


@pytest.fixture
def delta():
    """Standard delta value for tests."""
    return 1e-5
