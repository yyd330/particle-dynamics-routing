"""Smoke-test entry point for the PDR pipeline.

Builds a 200-node planar topology (the paper's configuration), prints its
structural statistics, and exercises the core routing primitives (one forward
trajectory, one Dijkstra path, one reverse pair, and a bounded reachability
sweep) on real nodes so the smoke test catches regressions in any of them.

Run:
    python general_test.py            # fast (200-node) smoke test
    python general_test.py --full     # larger reachability sweep (slower)
"""

import argparse
import logging
import math
import random
import time

import matplotlib
matplotlib.use("Agg")  # headless-safe before any pyplot import

from general_topology_implementation import TOPOLOGY
from planar_topology_implementation import (
    angle_link_calculation,
    coefficient_of_variation_calculation,
    planar_topology_creation,
)
from gravitational_algorithms import (
    GRAVITY_ACCEL,
    dijkstra_sp,
    heat_map,
    reverse_trajectory_algorithm,
    search_reachability,
    trajectory_algorithm,
)

log = logging.getLogger("general_test")

# Paper's topology configuration.
NUM_NODES = 200
GRID_SIZE = 10.0
GRID_DIVISION = 10
RADIUS = 2.0
MIN_DEGREE = 2
AVG_DEGREE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDR pipeline smoke test.")
    parser.add_argument("--full", action="store_true",
                        help="run a larger reachability/heat-map sweep")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def print_topology_stats(topology, all_nodes: list) -> None:
    """Print node-count and degree statistics for the built topology."""
    mean, std_dev, cov = coefficient_of_variation_calculation(topology, GRID_SIZE)
    degrees = [node.degree() for node in all_nodes]
    log.info("%d nodes total", len(all_nodes))
    log.info("Node density -> mean %.3f, std dev %.3f, coefficient of variation %.3f",
             mean, std_dev, cov)
    log.info("Degree -> average %.3f, max %d, min %d",
             topology.average_degree(), max(degrees), min(degrees))


def exercise_routing(topology, all_nodes: list) -> None:
    """Run one forward trajectory, one Dijkstra path, and one reverse pair."""
    start = all_nodes[0]
    neighbor_ids = [n for n in topology.get_neighbors(start)[1] if n != start.id]
    target = topology.get_node(neighbor_ids[0])

    # Forward trajectory toward a real neighbour.
    angle = angle_link_calculation(start, target)
    gravity_info = (GRAVITY_ACCEL, math.pi)
    t0 = time.process_time()
    traj = trajectory_algorithm(topology, start.id, target.id,
                                2.0, 4.0, angle, gravity_info, 1, 0, [])
    log.info("Forward trajectory %s -> %s: %d nodes (%.3fs)",
             start.id, target.id, len(traj), time.process_time() - t0)

    # Dijkstra all-pairs shortest paths (hop counts).
    t0 = time.process_time()
    all_paths = dijkstra_sp(topology)
    path = all_paths.get((start.id, target.id), [])
    log.info("Dijkstra all-pairs: %d pairs, %s -> %s = %d hops (%.3fs)",
             len(all_paths), start.id, target.id,
             len(path) - 1 if path else 0, time.process_time() - t0)

    # Reverse trajectory from two launch profiles sharing a downstream node.
    t0 = time.process_time()
    reverse = reverse_trajectory_algorithm(topology, start.id,
                                           ((angle, 2.0, 4.0), (angle, 2.5, 4.0)))
    log.info("Reverse trajectory produced %d node sequences (%.3fs)",
             len(reverse), time.process_time() - t0)


def run_reachability_sweep(topology, all_nodes: list, full: bool) -> None:
    """Bounded reachability + heat-map sweep over a sample of source nodes."""
    if full:
        sweep_nodes = all_nodes[: min(20, len(all_nodes))]
        speed_x_values = list(range(-5, 6))
        gravity_directions = [i * math.pi / 2 for i in range(4)]
    else:
        sweep_nodes = all_nodes[: min(5, len(all_nodes))]
        speed_x_values = list(range(-3, 4, 2))
        gravity_directions = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    non_reachable: dict = {}
    reachable_total = 0
    for start_node in sweep_nodes:
        reachable, non_reachable_ids, _ = search_reachability(
            topology, start_node.id, speed_x_values, gravity_directions)
        reachable_total += len(reachable)
        non_reachable[start_node.id] = non_reachable_ids

    possible = len(sweep_nodes) * (len(all_nodes) - 1)
    log.info("Sweep: %d sources, %d/%d node-pairs reachable (%.1f%%)",
             len(sweep_nodes), reachable_total, possible,
             100.0 * reachable_total / possible if possible else 0.0)

    heat_map_result = heat_map(topology, int(GRID_SIZE), 4, non_reachable)
    log.info("Heat map regions with data: %d", len(heat_map_result))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    start_time = time.process_time()
    random.seed(args.seed)

    log.info("Building planar topology (%d nodes)...", NUM_NODES)
    topology, all_nodes = planar_topology_creation(
        TOPOLOGY(), NUM_NODES, GRID_SIZE, GRID_DIVISION, RADIUS, MIN_DEGREE, AVG_DEGREE)

    print_topology_stats(topology, all_nodes)
    exercise_routing(topology, all_nodes)
    run_reachability_sweep(topology, all_nodes, full=args.full)

    log.info("Execution time: %.3f s", time.process_time() - start_time)


if __name__ == "__main__":
    main()
