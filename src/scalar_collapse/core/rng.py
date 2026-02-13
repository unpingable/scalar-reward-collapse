"""Seed discipline for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy Generator with the given seed."""
    return np.random.default_rng(seed)


def seed_all(seed: int) -> np.random.Generator:
    """Seed numpy and stdlib random; return a numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)  # legacy, for any library that uses it
    return make_rng(seed)
