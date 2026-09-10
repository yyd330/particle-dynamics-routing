"""Shared pytest fixtures for the PDR test suite."""

import random

import pytest

from general_topology_implementation import TOPOLOGY
from planar_topology_implementation import planar_topology_creation


@pytest.fixture(scope="session")
def small_topology():
    """A 20-node planar topology (fast), built once per test session."""
    random.seed(42)
    topology, all_nodes = planar_topology_creation(
        TOPOLOGY(), num_nodes=20, grid_size=10.0, grid_division=4,
        radius=2.0, min_degree=2, avg_degree=4)
    return topology, all_nodes


@pytest.fixture(scope="session")
def medium_topology():
    """A 60-node planar topology for slightly larger-scale tests."""
    random.seed(7)
    topology, all_nodes = planar_topology_creation(
        TOPOLOGY(), num_nodes=60, grid_size=10.0, grid_division=6,
        radius=2.0, min_degree=2, avg_degree=4)
    return topology, all_nodes
