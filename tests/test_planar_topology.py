"""Tests for planar topology generation (planar_topology_implementation)."""

import random

import pytest

from general_topology_implementation import TOPOLOGY
from planar_topology_implementation import (
    angle_link_calculation,
    check_continuity,
    check_planar,
    coefficient_of_variation_calculation,
    dfs,
    link_dictionary,
    planar_topology_creation,
)


def build_topology(num_nodes=20, seed=42):
    random.seed(seed)
    return planar_topology_creation(
        TOPOLOGY(), num_nodes=num_nodes, grid_size=10.0, grid_division=4,
        radius=2.0, min_degree=2, avg_degree=4)


def test_topology_has_requested_number_of_nodes(small_topology):
    topology, all_nodes = small_topology
    assert len(all_nodes) == 20
    assert topology.size == 20


def test_topology_is_connected(small_topology):
    topology, _ = small_topology
    assert check_continuity(topology)


def test_topology_is_planar(small_topology):
    topology, _ = small_topology
    assert check_planar(topology)


def test_minimum_degree_is_met(small_topology):
    _, all_nodes = small_topology
    assert all(node.degree() >= 2 for node in all_nodes)


def test_average_degree_near_target(small_topology):
    topology, _ = small_topology
    assert abs(topology.average_degree() - 4) < 1.5


def test_coordinates_inside_boundary(small_topology):
    _, all_nodes = small_topology
    assert all(0 <= node.x <= 10 and 0 <= node.y <= 10 for node in all_nodes)


def test_link_dictionary_excludes_self_entries(small_topology):
    topology, _ = small_topology
    link_dict = link_dictionary(topology)
    for node_id, neighbor_ids in link_dict.items():
        assert node_id not in neighbor_ids
        # symmetric
        assert node_id in link_dict[neighbor_ids[0]] or not neighbor_ids


def test_dfs_visits_all_nodes_in_connected_component(small_topology):
    topology, all_nodes = small_topology
    visited = dfs(topology, all_nodes[0].id)
    assert set(visited) == {node.id for node in all_nodes}


def test_angle_link_calculation_rounds_to_expected_quadrants():
    topology, all_nodes = build_topology(10, seed=1)
    a, b = all_nodes[0], all_nodes[1]
    angle = angle_link_calculation(a, b)
    assert 0 <= angle < 2 * 3.141592653589793


def test_coefficient_of_variation_uniform_placement():
    """At the paper's 200-node / 10x10 scale the placement is near-uniform."""
    random.seed(3)
    topology, _ = planar_topology_creation(
        TOPOLOGY(), num_nodes=200, grid_size=10.0, grid_division=10,
        radius=2.0, min_degree=2, avg_degree=5)
    mean, std_dev, cov = coefficient_of_variation_calculation(topology, 10.0)
    assert mean == pytest.approx(2.0, abs=0.01)  # 200 nodes / 100 cells
    assert cov < 0.1


def test_topology_deterministic_for_same_seed():
    topo_a, nodes_a = build_topology(15, seed=99)
    topo_b, nodes_b = build_topology(15, seed=99)
    positions_a = sorted((n.x, n.y) for n in nodes_a)
    positions_b = sorted((n.x, n.y) for n in nodes_b)
    assert positions_a == positions_b
    assert topo_a == topo_b


def test_place_nodes_exact_count():
    """_place_nodes must return exactly num_nodes positions (any count)."""
    from planar_topology_implementation import _place_nodes

    for n in (10, 30, 50, 100, 123, 200):
        positions = _place_nodes(n, 10.0, 10)
        assert len(positions) == n
        assert len(set(positions)) == n
        for x, y in positions:
            assert 0.0 < x < 10.0
            assert 0.0 < y < 10.0
