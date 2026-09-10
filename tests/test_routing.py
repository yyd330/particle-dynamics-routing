"""Tests for the PDR routing algorithms (gravitational_algorithms).

These encode the regression coverage for the two historical crashes:
* recursive per-hop trajectory simulation -> RecursionError on real
  destinations (now iterative with a MAX_HOPS cap), and
* unbounded exponential speed search -> float64 OverflowError (now a
  bounded scan).
"""

import math

import pytest

from gravitational_algorithms import (
    DNE_NODE,
    MAX_HOPS,
    dijkstra_sp,
    find_reverse_demo_params,
    heat_map,
    heat_map_plotting,
    multiple_trajectories_plot,
    nonreachable_node_plot,
    reverse_trajectory_algorithm,
    search_reachability,
    single_trajectory_plot,
    speed_xy_calculation,
    trajectory_algorithm,
    trajectory_search,
)

GRAVITY = (10.0, math.pi)


def test_speed_xy_calculation():
    vx, vy = speed_xy_calculation(5.0, 0.0)
    assert vx == pytest.approx(5.0)
    assert vy == pytest.approx(0.0)
    vx, vy = speed_xy_calculation(5.0, math.pi / 2)
    assert vx == pytest.approx(0.0, abs=1e-6)
    assert vy == pytest.approx(5.0, abs=1e-6)


def test_trajectory_to_real_destination_is_finite_and_terminates(medium_topology):
    """Regression: used to hit RecursionError on multi-hop real destinations."""
    topology, all_nodes = medium_topology
    start = all_nodes[0]
    neighbor_ids = [n for n in start.neighbors if n != start.id]
    target = topology.get_node(neighbor_ids[-1])  # a far neighbour
    angle = 0.5
    output = trajectory_algorithm(topology, start.id, target.id,
                                  2.0, 4.0, angle, GRAVITY, 1, 0, [])
    assert isinstance(output, list)
    assert len(output) <= MAX_HOPS + 2
    # ends with the [node_id, vx, vy] terminal record or a node id
    assert isinstance(output[-1], list) or isinstance(output[-1], str)


def test_trajectory_to_dne_probe_terminates_quickly(medium_topology):
    topology, all_nodes = medium_topology
    for _ in range(3):
        output = trajectory_algorithm(topology, all_nodes[0].id, DNE_NODE,
                                      1.0, 3.0, 1.0, GRAVITY, 1, 0, [])
        assert len(output) <= MAX_HOPS + 2


def test_trajectory_search_does_not_overflow(medium_topology):
    """Regression: exponential bounds used to overflow float64."""
    topology, all_nodes = medium_topology
    start = all_nodes[0]
    result = trajectory_search(topology, start.id, DNE_NODE, 0.7, 1.0, GRAVITY)
    assert isinstance(result, dict)
    assert all(isinstance(speed, float) for speed in result)
    assert all(isinstance(traj, list) for traj in result.values())


def test_dijkstra_sp_covers_all_pairs(medium_topology):
    topology, all_nodes = medium_topology
    all_paths = dijkstra_sp(topology)
    n = len(all_nodes)
    assert len(all_paths) == n * (n - 1)
    sample = all_paths[(all_nodes[0].id, all_nodes[1].id)]
    assert sample[0] == all_nodes[0].id
    assert sample[-1] == all_nodes[1].id


def test_search_reachability_partition_is_complete(medium_topology):
    topology, all_nodes = medium_topology
    reachable, non_reachable, trajectories = search_reachability(
        topology, all_nodes[0].id, [-1, 0, 1], [0.0, math.pi / 2])
    other_ids = {n.id for n in all_nodes if n.id != all_nodes[0].id}
    assert not (set(reachable) & set(non_reachable))
    assert set(reachable) | set(non_reachable) == other_ids
    assert trajectories  # some sweeps were actually executed


def test_heat_map_tallies_match_reachable_sets(medium_topology):
    topology, all_nodes = medium_topology
    non_reachable = {n.id: [] for n in all_nodes[:5]}  # all-reachable sources
    result = heat_map(topology, 10, 4, non_reachable)
    assert result
    for key, (total, reachable, pct) in result.items():
        assert 1 <= key[0] <= 16 and 1 <= key[1] <= 16
        assert 0 <= reachable <= total
        assert pct == pytest.approx(reachable / total * 100, abs=1e-9)


def test_reverse_trajectory_terminates(medium_topology):
    topology, all_nodes = medium_topology
    angle = 0.9
    reverse = reverse_trajectory_algorithm(
        topology, all_nodes[0].id,
        ((angle, 2.0, 4.0), (angle + 0.1, 2.2, 4.0)))
    assert len(reverse) == 2
    assert all(isinstance(seq, list) and seq for seq in reverse)


def test_find_reverse_demo_params_or_none(medium_topology):
    topology, all_nodes = medium_topology
    params = find_reverse_demo_params(topology, all_nodes[0].id)
    assert params is None or (
        len(params) == 2 and all(len(profile) == 3 for profile in params))


def test_plots_save_files_headless(medium_topology, tmp_path):
    topology, all_nodes = medium_topology
    angle = 0.9

    traj = trajectory_algorithm(topology, all_nodes[0].id, DNE_NODE,
                                1.5, 3.0, angle, GRAVITY, 1, 0, [])
    out1 = single_trajectory_plot(topology, traj, "T", out_dir=str(tmp_path))
    assert out1 and out1.endswith(".png") and tmp_path.joinpath(out1.split("/")[-1]).exists()

    out2 = multiple_trajectories_plot(topology, {1.0: traj}, "MT", out_dir=str(tmp_path))
    assert out2 and tmp_path.joinpath(out2.split("/")[-1]).exists()

    out3 = nonreachable_node_plot(topology, all_nodes[0].id, [all_nodes[1].id],
                                  out_dir=str(tmp_path))
    assert out3 and tmp_path.joinpath(out3.split("/")[-1]).exists()

    out4 = heat_map_plotting(topology, 10, 4,
                             heat_map(topology, 10, 4, {n.id: [] for n in all_nodes}),
                             out_dir=str(tmp_path))
    assert out4 and tmp_path.joinpath(out4.split("/")[-1]).exists()
