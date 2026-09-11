"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_sensory_features():
    """Sample sensory feature vector."""
    return np.random.randn(10)


@pytest.fixture
def positive_features():
    """Positive (approach) sensory features."""
    return np.ones(10) * 0.5


@pytest.fixture
def negative_features():
    """Negative (avoid) sensory features."""
    return np.ones(10) * -0.5
