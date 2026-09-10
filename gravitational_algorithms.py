"""Particle Dynamics Routing (PDR) -- simulation and analysis algorithms.

Implements the routing-by-physics model from "Particle Dynamics Routing for
Wireless Mesh Networks" (PLOS ONE): a packet is a point mass launched from a
source node with a chosen velocity (speed + angle); its next hop is the
topology neighbour whose link it physically reaches first, under a uniform
gravitational field of chosen direction.  This module provides the trajectory
integrator, its reverse, a launch-velocity (trajectory) search, Dijkstra
shortest paths, a reachability sweep, and the regional reachability heat map,
plus all plotting helpers.

Run ``python gravitational_algorithms.py --help`` for the experiment options.
"""
from general_topology_implementation import TOPOLOGY
from planar_topology_implementation import angle_link_calculation, planar_topology_creation, \
    coefficient_of_variation_calculation

import argparse
import logging
import math
import os
import random
import re
import sys
import time
import heapq

from concurrent.futures import ProcessPoolExecutor

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (replace former magic numbers)
# ---------------------------------------------------------------------------
GRAVITY_ACCEL = 10.0            # uniform gravity field, m/s^2
DNE_NODE = "0"                  # non-existent probe node used to trace full trajectories
MAX_HOPS = 500                  # hop cap: a trajectory longer than this is declared non-terminating
MAX_EXPONENT = 20               # trajectory-search upper bound 2**MAX_EXPONENT (guards float64 overflow)
DELTA_T = 0.01                  # base simulation step (s); scaled per hop inside the integrator
STOP_ANGLE = math.radians(85)   # boundary-stopping threshold: turn vs. gravity direction


def _finish_plot(title: str, out_dir: str | None = None) -> str | None:
    """Finalize a figure: optional save + non-blocking display.

    Returns the saved file path (or ``None`` when no ``out_dir``).  In
    headless environments (no display) the figure is only saved, never
    shown, so runs under CI/remote shells never block.
    """
    fig = plt.gcf()
    fig.tight_layout()
    path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", title)[:80]
        path = os.path.join(out_dir, safe + ".png")
        fig.savefig(path, dpi=150)
        log.info("Saved plot: %s", path)
    if sys.stdout.isatty():
        plt.show(block=False)
    plt.close(fig)
    return path


def _topology_arrays(topology: TOPOLOGY):
    """Return (node_x, node_y, edge_x, edge_y) arrays for plotting."""
    nodes = topology.get_all_nodes()
    node_x = [n.x for n in nodes]
    node_y = [n.y for n in nodes]
    edge_x, edge_y = [], []
    for node in nodes:
        for neighbor_id in topology.get_neighbors(node)[1]:
            neighbor = topology.get_node(neighbor_id)
            edge_x += [node.x, neighbor.x]
            edge_y += [node.y, neighbor.y]
    return node_x, node_y, edge_x, edge_y


def _draw_topology_axes(topology: TOPOLOGY, title: str, node_style: dict | None = None,
                        edge_style: dict | None = None):
    """Plot the topology skeleton; return the active axes."""
    plt.figure()
    node_style = node_style or {"marker": "o", "color": "gray", "s": 12}
    edge_style = edge_style or {"color": "lightgray", "linewidth": 0.8, "alpha": 0.5}
    node_x, node_y, edge_x, edge_y = _topology_arrays(topology)
    for i in range(0, len(edge_x), 2):
        plt.plot(edge_x[i:i + 2], edge_y[i:i + 2], zorder=1, **edge_style)
    plt.scatter(node_x, node_y, zorder=2, **node_style)
    plt.title(title, fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return plt.gca()


def _plot_trajectory(ax_or_none, topology: TOPOLOGY, trajectory: list, color: str,
                     start_color: str = "green", lw: float = 2.0, mark_end: bool = True):
    """Overlay one trajectory (list of node ids) on the current figure."""
    xs, ys = [], []
    for node_id in trajectory:
        if not isinstance(node_id, str):
            continue  # terminal [node_id, vx, vy] state record
        node = topology.get_node(node_id)
        xs.append(node.x)
        ys.append(node.y)
    if not xs:
        return
    plt.scatter(xs, ys, marker="o", s=18, color=color, zorder=3)
    plt.scatter(xs[0], ys[0], marker="o", s=45, color=start_color, zorder=4)
    if mark_end:
        plt.scatter(xs[-1], ys[-1], marker="o", s=45, color=color, zorder=4)
    if len(xs) > 1:
        plt.plot(xs, ys, color=color, linewidth=lw, alpha=0.85, zorder=3)

def trajectory_algorithm(topology: TOPOLOGY, curr_node_id: str, dest_node_id: str, curr_speed_x: float,
                         curr_speed_y: float, curr_angle_theta_n: float, gravity_vector: tuple, scalar_dt: int,
                         run_time: int, output: list):
    """Simulate the gravitational particle trajectory from ``curr_node_id``
    towards ``dest_node_id``.

    A particle is launched from the start node; at each hop it is projected
    forward (integrating gravity) until the trajectory intersects one of the
    start node's neighbours, which becomes the next node.  The process repeats
    until the destination node is reached or a hop limit / oscillation is
    detected.

    The simulation is iterative (not recursive).  A particle can bounce between
    two neighbours indefinitely on a real topology, so the original recursive
    form accumulated up to ``MAX_HOPS`` interpreter stack frames and raised
    ``RecursionError``.  Advancing hop-by-hop in a loop removes that bound and
    adds a visited-node guard that stops genuine oscillation.

    Parameters
    ----------
    topology : TOPOLOGY
    curr_node_id : str
        id of the node the particle currently occupies (start node).
    dest_node_id : str
        id of the destination node (``'0'`` is used as a "non-existent" probe).
    curr_speed_x, curr_speed_y : float
        incoming velocity components (m/s).
    curr_angle_theta_n : float
        launch / incoming angle (radians).
    gravity_vector : tuple
        ``(magnitude, direction)`` in m/s^2 and radians.
    scalar_dt, run_time : int
        legacy per-hop / elapsed-time counters kept for signature compatibility.
    output : list
        accumulated node-id list (mutated in place).

    Returns
    -------
    list
        The node-id path (start .. terminal).  When the destination is reached a
        final ``[node_id, vx, vy]`` record is appended describing the velocity at
        the last hop.
    """
    delta_t = 0.01
    acc_gravity = gravity_vector[0]
    gravity_direction = gravity_vector[1]

    output.append(curr_node_id)

    node_id = curr_node_id
    speed_x = curr_speed_x
    speed_y = curr_speed_y
    angle_n = curr_angle_theta_n

    visited = {curr_node_id}
    first_hop = True

    for _ in range(MAX_HOPS):
        curr_node = topology.get_node(node_id)
        next_node_found = 0
        not_reachable = 0
        next_speed_x = next_speed_y = next_angle_theta_n = 0.0
        next_node_id = ''
        scalar_dt_hop = scalar_dt if first_hop else 1

        if first_hop:
            # Launch hop: pick the neighbour whose link angle is closest to the
            # launch angle, then set the outgoing velocity.
            next_node_found = 1
            link_angles = []
            neighbor_nodes_ids = []
            for neighbor_id in curr_node.neighbors:
                neighbor = topology.get_node(neighbor_id)
                if neighbor != curr_node:
                    angle = angle_link_calculation(curr_node, neighbor)
                    link_angles.append(angle)
                    neighbor_nodes_ids.append(neighbor_id)
            link_diff = []
            for angle in link_angles:
                difference = abs(angle - curr_angle_theta_n)
                if difference >= math.pi:
                    difference = 2 * math.pi - difference
                link_diff.append(difference)
            link_desired = min(link_diff)
            node_desired_index = list(link_diff).index(link_desired)
            next_node_id = neighbor_nodes_ids[node_desired_index]
            next_node = topology.get_node(next_node_id)
            link_length = curr_node.geographical_distance(next_node)
            curr_velocity = math.sqrt(math.pow(speed_x, 2) + math.pow(speed_y, 2))
            if curr_velocity != 0:
                scalar_dt_hop = (link_length / curr_velocity) / delta_t
            else:
                scalar_dt_hop = link_length / delta_t
            next_speed_x = speed_x
            next_speed_y = speed_y - acc_gravity * scalar_dt_hop * delta_t
            next_angle_theta_n = angle_n
            first_hop = False

        while next_node_found == 0:
            g_angle_difference = gravity_direction - (3 * math.pi / 2)
            abstract_point_x = curr_node.x + speed_x * scalar_dt_hop * delta_t
            abstract_point_y = curr_node.y + speed_y * scalar_dt_hop * delta_t - 0.5 * acc_gravity * math.pow((scalar_dt_hop * delta_t), 2)

            if curr_node.x != abstract_point_x:
                angle_theta_p = math.atan((abstract_point_y - curr_node.y) / (abstract_point_x - curr_node.x))
                if curr_node.x > abstract_point_x:
                    angle_theta_p += math.pi
                elif curr_node.x < abstract_point_x and curr_node.y >= abstract_point_y:
                    angle_theta_p += 2 * math.pi
            else:
                if curr_node.y > abstract_point_y:
                    angle_theta_p = 3 * math.pi / 2
                else:
                    angle_theta_p = math.pi / 2
            angle_theta_p += g_angle_difference
            if angle_theta_p > 2 * math.pi:
                angle_theta_p -= 2 * math.pi
            elif angle_theta_p < 0:
                angle_theta_p += 2 * math.pi

            link_angles = []
            neighbor_nodes_ids = []
            for neighbor_id in curr_node.neighbors:
                neighbor = topology.get_node(neighbor_id)
                if neighbor != curr_node:
                    angle = angle_link_calculation(curr_node, neighbor)
                    link_angles.append(angle)
                    neighbor_nodes_ids.append(neighbor_id)

            link_diff = []
            for angle in link_angles:
                difference = abs(angle - angle_theta_p)
                if difference >= math.pi:
                    difference = 2 * math.pi - difference
                link_diff.append(difference)
            link_desired = min(link_diff)
            node_desired_index = list(link_diff).index(link_desired)
            next_node_id = neighbor_nodes_ids[node_desired_index]
            next_node = topology.get_node(next_node_id)
            link_length = curr_node.geographical_distance(next_node)
            distance_p_n = math.sqrt(math.pow((abstract_point_y - curr_node.y), 2) + math.pow((abstract_point_x - curr_node.x), 2))

            if distance_p_n >= link_length:
                next_speed_x = speed_x
                next_speed_y = speed_y - acc_gravity * scalar_dt_hop * delta_t
                next_angle_theta_n = angle_link_calculation(curr_node, next_node)
                next_node_found = 1

                # Stopping Criteria: near the down-field boundary a hop that
                # would turn back >85deg from the gravity direction is infeasible.
                if (gravity_direction >= (7 * math.pi / 4)) or (gravity_direction < (math.pi / 4)):
                    if curr_node.x >= 9.6:
                        angle_diff = abs(next_angle_theta_n - gravity_direction)
                        if angle_diff > math.radians(85):
                            not_reachable = 1
                elif (math.pi / 4) <= gravity_direction < (3 * math.pi / 4):
                    if curr_node.y >= 9.6:
                        angle_diff = abs(next_angle_theta_n - gravity_direction)
                        if angle_diff > math.radians(85):
                            not_reachable = 1
                elif (3 * math.pi / 4) <= gravity_direction < (5 * math.pi / 4):
                    if curr_node.x <= 0.4:
                        angle_diff = abs(next_angle_theta_n - gravity_direction)
                        if angle_diff > math.radians(85):
                            not_reachable = 1
                elif (5 * math.pi / 4) <= gravity_direction < (7 * math.pi / 4):
                    if curr_node.y <= 0.4:
                        angle_diff = abs(next_angle_theta_n - gravity_direction)
                        if angle_diff > math.radians(85):
                            not_reachable = 1
            else:
                scalar_dt_hop += 1

        # Record the node just reached.  The original recursive implementation
        # appended each node at the top of its own frame (one append per hop);
        # the loop equivalent is this single append per iteration.
        output.append(next_node_id)

        if dest_node_id == next_node_id:
            output.append([node_id, speed_x, speed_y])
            break

        if not_reachable:
            # Packet arrived at next_node_id but cannot continue the hop;
            # record it as the terminal node of the trajectory.
            break

        if next_node_id in visited:
            # Oscillating between neighbours -- the destination is unreachable.
            break

        visited.add(next_node_id)
        node_id, speed_x, speed_y, angle_n = next_node_id, next_speed_x, next_speed_y, next_angle_theta_n

    return output


def reverse_hop(topology: TOPOLOGY, curr_node_id: str, curr_speed_x: float, curr_speed_y: float,
                gravity_vector: tuple) -> tuple:
    """Advance one reverse hop.

    Mirrors the forward integrator but walks the trajectory backwards: given
    the node and the velocity at which the packet *arrives* there, recover the
    previous node (the neighbour the packet must have left from, travelling
    against the field) and the outgoing velocity at that node.

    Returns ``(next_node_id, next_speed_x, next_speed_y, next_angle,
    not_reachable_flag)``.
    """
    curr_node = topology.get_node(curr_node_id)
    acc_gravity, gravity_direction = gravity_vector

    scalar_dt = 1
    while True:
        # Abstract point p reached after one base step (signs reversed vs. forward).
        g_angle_difference = gravity_direction - (3 * math.pi / 2)
        abstract_point_x = curr_node.x - curr_speed_x * scalar_dt * DELTA_T
        abstract_point_y = (curr_node.y - curr_speed_y * scalar_dt * DELTA_T
                            + 0.5 * acc_gravity * math.pow(scalar_dt * DELTA_T, 2))

        if curr_node.x != abstract_point_x:
            angle_theta_p = math.atan((abstract_point_y - curr_node.y) / (abstract_point_x - curr_node.x))
            if curr_node.x > abstract_point_x:
                angle_theta_p += math.pi
            elif curr_node.x < abstract_point_x and curr_node.y >= abstract_point_y:
                angle_theta_p += 2 * math.pi
        else:
            angle_theta_p = (3 * math.pi / 2) if curr_node.y > abstract_point_y else math.pi / 2
        angle_theta_p += g_angle_difference
        if angle_theta_p > 2 * math.pi:
            angle_theta_p -= 2 * math.pi
        elif angle_theta_p < 0:
            angle_theta_p += 2 * math.pi

        # Choose the neighbour whose link best matches the abstract point.
        link_angles, neighbor_ids = [], []
        for neighbor_id in curr_node.neighbors:
            if neighbor_id == curr_node_id:
                continue
            neighbor = topology.get_node(neighbor_id)
            link_angles.append(angle_link_calculation(curr_node, neighbor))
            neighbor_ids.append(neighbor_id)
        link_diff = []
        for angle in link_angles:
            difference = abs(angle - angle_theta_p)
            if difference >= math.pi:
                difference = 2 * math.pi - difference
            link_diff.append(difference)
        node_desired_index = link_diff.index(min(link_diff))
        next_node = topology.get_node(neighbor_ids[node_desired_index])
        link_length = curr_node.geographical_distance(next_node)
        distance_p_n = math.hypot(abstract_point_x - curr_node.x, abstract_point_y - curr_node.y)

        # x-component is fixed; only the y-component changes under gravity.
        if distance_p_n < link_length:
            scalar_dt += 1
            continue

        next_speed_x = curr_speed_x
        next_speed_y = curr_speed_y + acc_gravity * scalar_dt * DELTA_T
        next_angle = angle_link_calculation(curr_node, next_node)

        # Stopping criterion: near the down-field boundary a hop that would
        # turn back more than STOP_ANGLE from the gravity direction is infeasible.
        not_reachable = 0
        boundary_x = 9.6
        boundary_y = 0.4
        if (gravity_direction >= (7 * math.pi / 4)) or (gravity_direction < (math.pi / 4)):
            hit = curr_node.x >= boundary_x
        elif (math.pi / 4) <= gravity_direction < (3 * math.pi / 4):
            hit = curr_node.y >= boundary_x
        elif (3 * math.pi / 4) <= gravity_direction < (5 * math.pi / 4):
            hit = curr_node.x <= boundary_y
        else:
            hit = curr_node.y <= boundary_y
        if hit and abs(next_angle - gravity_direction) > STOP_ANGLE:
            not_reachable = 1

        return neighbor_ids[node_desired_index], next_speed_x, next_speed_y, next_angle, not_reachable


def reverse_trajectory_algorithm(topology: TOPOLOGY, start_node_id: str, start_info_tuple: tuple,
                                 out_dir: str | None = None):
    """Reconstruct reverse trajectories for two launch profiles from ``start_node_id``.

    ``start_info_tuple`` = ``((angle_1, speed_x_1, speed_y_1), (angle_2, speed_x_2, speed_y_2))``.
    The two forward trajectories are traced to their first common node; from
    that node the trajectories are reversed back towards the source, and the
    result is plotted (and optionally saved to ``out_dir``).

    Iterative -- no per-hop recursion (the original implementation exceeded
    the recursion limit on real topologies).
    """
    gravity_info = (GRAVITY_ACCEL, 3 * math.pi / 2)
    (init_angle_1, init_speed_x_1, init_speed_y_1) = start_info_tuple[0]
    (init_angle_2, init_speed_x_2, init_speed_y_2) = start_info_tuple[1]

    trajectory_1 = trajectory_algorithm(topology, start_node_id, DNE_NODE, init_speed_x_1,
                                        init_speed_y_1, init_angle_1, gravity_info, 1, 0, [])
    trajectory_2 = trajectory_algorithm(topology, start_node_id, DNE_NODE, init_speed_x_2,
                                        init_speed_y_2, init_angle_2, gravity_info, 1, 0, [])

    # First node visited by both forward trajectories (excluding the source).
    common_node_id = None
    tail_1 = trajectory_1[1:-1] if isinstance(trajectory_1[-1], list) else trajectory_1[1:]
    tail_2 = trajectory_2[1:-1] if isinstance(trajectory_2[-1], list) else trajectory_2[1:]
    for node_id_1 in tail_1:
        if isinstance(node_id_1, str) and node_id_1 in tail_2:
            common_node_id = node_id_1
            break

    _draw_topology_axes(topology, "Forward Trajectories Before Reversal",
                        node_style={"marker": "o", "color": "cornflowerblue", "s": 12})
    _plot_trajectory(None, topology, trajectory_1, "black")
    _plot_trajectory(None, topology, trajectory_2, "darkorange")
    _finish_plot("Reverse Trajectory - Forward", out_dir)

    if common_node_id is None:
        log.warning("No common node between the two trajectories; nothing to reverse.")
        return [], []

    def _reverse_to_start(src_node: str, angle: float, speed_x: float, speed_y: float) -> list:
        """Trace one trajectory src -> common node, then reverse it to the start."""
        to_common = trajectory_algorithm(topology, src_node, common_node_id, speed_x, speed_y,
                                         angle, gravity_info, 1, 0, [])
        last = to_common[-1]
        reverse_output: list = [common_node_id]
        curr_id, curr_sx, curr_sy = common_node_id, last[1], last[2]
        seen = {common_node_id}
        hops = 0
        while hops < MAX_HOPS:
            hops += 1
            next_id, next_sx, next_sy, _angle, not_reachable = reverse_hop(
                topology, curr_id, curr_sx, curr_sy, gravity_info)
            reverse_output.append(next_id)
            if next_id == src_node or not_reachable or next_id in seen:
                break
            seen.add(next_id)
            curr_id, curr_sx, curr_sy = next_id, next_sx, next_sy
        return reverse_output

    reverse_trajectory_1 = _reverse_to_start(start_node_id, init_angle_1, init_speed_x_1, init_speed_y_1)
    reverse_trajectory_2 = _reverse_to_start(start_node_id, init_angle_2, init_speed_x_2, init_speed_y_2)

    _draw_topology_axes(topology, "Reverse Trajectory Test",
                        node_style={"marker": "o", "color": "cornflowerblue", "s": 12},
                        edge_style={"color": "lightsteelblue", "linewidth": 1})
    _plot_trajectory(None, topology, reverse_trajectory_1, "blue", lw=3)
    _plot_trajectory(None, topology, reverse_trajectory_2, "darkorange", lw=3)
    _finish_plot("Reverse Trajectory - Backwards", out_dir)
    return reverse_trajectory_1, reverse_trajectory_2


def find_reverse_demo_params(topology: TOPOLOGY, start_node_id: str) -> tuple | None:
    """Find a pair of launch profiles whose forward trajectories share a node.

    The reverse-trajectory experiment needs two forward paths from the same
    source that merge at some downstream node.  Rather than hard-coding
    seed-specific parameters (as the original script did), this searches a
    bounded grid of (speed_x, speed_y) and launch angles derived from the
    start node's real neighbour links.

    Returns ``((angle_1, speed_x_1, speed_y_1), (angle_2, speed_x_2, speed_y_2))``
    or ``None`` if no shared node was found in the searched grid.
    """
    start_node = topology.get_node(start_node_id)
    gravity_info = (GRAVITY_ACCEL, 3 * math.pi / 2)
    if len(start_node.neighbors) < 2:
        return None
    neighbor_ids = sorted(start_node.neighbors)
    angles = [angle_link_calculation(start_node, topology.get_node(nid)) for nid in neighbor_ids]

    def _node_set(speed_x: float, speed_y: float, angle: float) -> set:
        traj = trajectory_algorithm(topology, start_node_id, DNE_NODE,
                                    speed_x, speed_y, angle, gravity_info, 1, 0, [])
        return {nid for nid in traj if isinstance(nid, str)} - {start_node_id}

    for speed_x in (1, 2, 3, 4):
        for speed_y in (8, 10, 12, 15):
            for i, angle_1 in enumerate(angles):
                set_1 = _node_set(speed_x, speed_y, angle_1)
                if len(set_1) < 2:
                    continue
                for angle_2 in angles[i + 1:]:
                    common = set_1 & _node_set(speed_x, speed_y, angle_2)
                    if common:
                        return ((angle_1, speed_x, speed_y), (angle_2, speed_x, speed_y))
    return None


def single_trajectory_plot(topology: TOPOLOGY, output: list, title: str = "Single Trajectory",
                           out_dir: str | None = None) -> str | None:
    """Plot one trajectory (list of node ids) over the topology skeleton."""
    _draw_topology_axes(topology, title)
    if output:
        _plot_trajectory(None, topology, output, "red", lw=3)
    return _finish_plot(title, out_dir)


def multiple_trajectories_plot(topology: TOPOLOGY, output: dict,
                               title: str = "Multiple Trajectories",
                               out_dir: str | None = None) -> str | None:
    """Plot several trajectories (dict: key -> list of node ids) on one figure."""
    _draw_topology_axes(topology, title)
    palette = [name for name in mcolors.CSS4_COLORS
               if f"xkcd:{name}" in mcolors.XKCD_COLORS]
    palette.remove("black")
    palette.remove("red")
    for index, key in enumerate(output):
        _plot_trajectory(None, topology, output[key], palette[index % len(palette)], lw=2)
    return _finish_plot(title, out_dir)


def nonreachable_node_plot(topology: TOPOLOGY, init_node_id: str, non_reachable_list: list,
                           out_dir: str | None = None) -> str | None:
    """Plot the source node (red) and its unreachable nodes (black) on the topology."""
    _draw_topology_axes(topology, "Non-Reachable Nodes (Union of 8 Gravity Directions)",
                        node_style={"marker": "o", "color": "gray", "s": 3})
    for node_id in non_reachable_list:
        node = topology.get_node(node_id)
        plt.scatter(node.x, node.y, marker="o", color="black")
    init_node = topology.get_node(init_node_id)
    plt.scatter(init_node.x, init_node.y, marker="o", color="red")
    return _finish_plot("Non-Reachable Nodes", out_dir)


def speed_xy_calculation(speed: float, angle: float) -> tuple:
    """Decompose a launch speed into quadrant-correct (x, y) components."""
    if 0 <= angle < (math.pi / 2):
        return abs(speed * math.cos(angle)), abs(speed * math.sin(angle))
    if (math.pi / 2) <= angle < math.pi:
        return -abs(speed * math.cos(angle)), abs(speed * math.sin(angle))
    if math.pi <= angle < (3 * math.pi / 2):
        return -abs(speed * math.cos(angle)), -abs(speed * math.sin(angle))
    if (3 * math.pi / 2) <= angle < (2 * math.pi):
        return abs(speed * math.cos(angle)), -abs(speed * math.sin(angle))
    return speed * math.cos(angle), speed * math.sin(angle)


def dijkstra_sp(topology: TOPOLOGY) -> dict:
    """All-pairs hop-count shortest paths: {(src, dst): [node ids]}."""
    all_nodes = topology.get_all_nodes()
    all_paths: dict = {}
    for start_node in all_nodes:
        # Single multi-source Dijkstra from start_node gives paths to every dest.
        distances = {node.id: math.inf for node in all_nodes}
        previous_nodes = {node.id: None for node in all_nodes}
        distances[start_node.id] = 0
        priority_queue = [(0, start_node.id)]
        while priority_queue:
            current_distance, current_node_id = heapq.heappop(priority_queue)
            if current_distance > distances[current_node_id]:
                continue
            for neighbor_id in topology.get_node(current_node_id).neighbors:
                new_distance = current_distance + 1
                if new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous_nodes[neighbor_id] = current_node_id
                    heapq.heappush(priority_queue, (new_distance, neighbor_id))
        for dest_node in all_nodes:
            if start_node.id == dest_node.id:
                continue
            path = []
            current_id = dest_node.id
            while current_id is not None:
                path.insert(0, current_id)
                current_id = previous_nodes[current_id]
            all_paths[(start_node.id, dest_node.id)] = path
    return all_paths


def trajectory_search(topology: TOPOLOGY, init_node_id: str, dest_node_id: str, init_angle: float,
                      speed_x: float, gravity_vector: tuple) -> dict:
    """Enumerate the distinct trajectories reachable by varying launch speed.

    Per the paper's route-search procedure (Sec. 10.1): an exponential phase
    doubles the launch speed (capped at ``2**MAX_EXPONENT`` to avoid float64
    overflow when the destination is the non-existent probe node) until the
    trajectory stops changing, establishing the search interval.  A binary
    search inside that interval then records every speed at which the
    trajectory differs from both interval endpoints, i.e. every distinct
    reachable trajectory.

    Returns ``{launch_speed: [node ids]}``.
    """
    def _comparison_trajectory(trajectory: list) -> list:
        """Normalize a trajectory for comparison (drop later repeated nodes)."""
        fixed = []
        for node_id in trajectory:
            if not isinstance(node_id, str):
                continue
            index = trajectory.index(node_id)
            if index > 2 and node_id in fixed:
                continue
            fixed.append(node_id)
        return fixed

    def _run(speed_y: float) -> list:
        return trajectory_algorithm(topology, init_node_id, dest_node_id, speed_x, speed_y,
                                    init_angle, gravity_vector, 1, 0, [])

    def _exponential_bounds() -> tuple:
        lower_speed = 1.0
        lower_xy = speed_xy_calculation(lower_speed, init_angle)
        lower_trajectory = _run(lower_xy[1])
        previous_speed, previous_trajectory = lower_speed, lower_trajectory
        exponent_k = 0
        while True:
            if exponent_k >= MAX_EXPONENT:
                return lower_speed, lower_trajectory, previous_speed, previous_trajectory
            exponent_k += 1
            intermediate_speed = math.pow(2, exponent_k)
            intermediate_xy = speed_xy_calculation(intermediate_speed, init_angle)
            intermediate_trajectory = _run(intermediate_xy[1])
            if _comparison_trajectory(previous_trajectory) == _comparison_trajectory(intermediate_trajectory):
                if exponent_k > 3:
                    return lower_speed, lower_trajectory, intermediate_speed, intermediate_trajectory
            lower_speed, lower_trajectory = previous_speed, previous_trajectory
            previous_speed, previous_trajectory = intermediate_speed, intermediate_trajectory

    def _binary_search(speed_lower: float, speed_higher: float,
                       trajectory_lower: list, trajectory_higher: list, depth: int = 0) -> None:
        """Recursively record every distinct trajectory between the endpoints."""
        if depth >= 30:  # ~1e9 speed units -- beyond float64 resolution; stop
            return
        mid_speed = (speed_lower + speed_higher) / 2
        if mid_speed == speed_lower or mid_speed == speed_higher:
            return
        mid_xy = speed_xy_calculation(mid_speed, init_angle)
        mid_trajectory = _run(mid_xy[1])
        norm_mid = _comparison_trajectory(mid_trajectory)
        eq_lower = norm_mid == _comparison_trajectory(trajectory_lower)
        eq_higher = norm_mid == _comparison_trajectory(trajectory_higher)
        if eq_lower and not eq_higher:
            _record(mid_speed, mid_trajectory, norm_mid)
            _binary_search(mid_speed, speed_higher, mid_trajectory, trajectory_higher, depth + 1)
        elif eq_higher and not eq_lower:
            _record(mid_speed, mid_trajectory, norm_mid)
            _binary_search(speed_lower, mid_speed, trajectory_lower, mid_trajectory, depth + 1)
        elif not eq_lower and not eq_higher:
            # mid matches neither endpoint: two more intervals to explore.
            _record(mid_speed, mid_trajectory, norm_mid)
            _binary_search(mid_speed, speed_higher, mid_trajectory, trajectory_higher, depth + 1)
            _binary_search(speed_lower, mid_speed, trajectory_lower, mid_trajectory, depth + 1)
        # else: mid equals both endpoints -- no distinct trajectory here.

    def _record(speed: float, trajectory: list, normalized: list) -> None:
        """Store a trajectory once (by normalized node sequence)."""
        if any(_comparison_trajectory(existing) == normalized for existing in trajectory_dict.values()):
            return
        trajectory_dict[speed] = trajectory

    lower_speed, lower_trajectory, higher_speed, _ = _exponential_bounds()
    higher_trajectory = _run(speed_xy_calculation(higher_speed, init_angle)[1])
    trajectory_dict: dict = {lower_speed: lower_trajectory}
    _record(higher_speed, higher_trajectory, _comparison_trajectory(higher_trajectory))
    _binary_search(lower_speed, higher_speed, lower_trajectory, higher_trajectory)
    return trajectory_dict


def search_reachability(topology: TOPOLOGY, start_node_id: str, speed_x_values: list,
                        gravity_directions: list, dest_node_id: str = DNE_NODE,
                        acc_gravity: float = GRAVITY_ACCEL) -> tuple:
    """Full launch-parameter sweep from one node: which nodes are physically reachable?

    Sweeps every launch speed, gravity direction and (per-neighbour) launch
    angle from ``start_node_id`` and unions all reachable nodes.

    Returns ``(reachable_ids, non_reachable_ids, trajectories)`` where
    ``trajectories`` is the raw list of per-(speed, angle, gravity) search
    outputs.
    """
    start_node = topology.get_node(start_node_id)
    all_nodes = topology.get_all_nodes()
    node_reachable: list = []
    trajectories = []
    for speed_x in speed_x_values:
        for g_direction in gravity_directions:
            gravity_info = (acc_gravity, g_direction)
            for neighbor_id in start_node.neighbors:
                if neighbor_id == start_node_id:
                    continue
                init_angle = angle_link_calculation(start_node, topology.get_node(neighbor_id))
                search_output = trajectory_search(topology, start_node_id, dest_node_id,
                                                  init_angle, speed_x, gravity_info)
                trajectories.append(search_output)
                for trajectory in search_output.values():
                    for node_id in trajectory:
                        if isinstance(node_id, str) and node_id != start_node_id \
                                and node_id not in node_reachable:
                            node_reachable.append(node_id)
    non_reachable = sorted(node.id for node in all_nodes
                           if node.id != start_node_id and node.id not in set(node_reachable))
    return node_reachable, non_reachable, trajectories


def _sweep_one_source(task: tuple) -> tuple:
    """Worker target for parallel reachability sweeps.

    ``task`` is ``(topology, start_node_id, speed_x_values, gravity_directions)``;
    returns ``(start_node_id, reachable_ids, non_reachable_ids)``.  Top-level so
    it is picklable for ``ProcessPoolExecutor``.
    """
    topology, start_node_id, speed_x_values, gravity_directions = task
    reachable, non_reachable, _ = search_reachability(
        topology, start_node_id, speed_x_values, gravity_directions)
    return start_node_id, reachable, non_reachable


def search_reachability_by_direction(topology: TOPOLOGY, start_node_id: str,
                                     speed_x_values: list, gravity_directions: list,
                                     acc_gravity: float = GRAVITY_ACCEL) -> tuple:
    """Reachability swept *per gravity direction* (mPDR analysis).

    For each gravity direction in ``gravity_directions`` the full
    speed/angle sweep is run independently, so each direction yields its own
    reachable set.  Prefix unions of these sets give the effective mPDR
    reachability for 1, 2, 4, 8, ... field directions (paper Fig 9).

    Returns ``(per_direction, union_set)`` where ``per_direction[i]`` is the
    set of node ids reachable under the (i+1)-th gravity direction.
    """
    start_node = topology.get_node(start_node_id)
    per_direction: list = []
    union_set: set = set()
    for g_direction in gravity_directions:
        gravity_info = (acc_gravity, g_direction)
        reachable: set = set()
        for speed_x in speed_x_values:
            for neighbor_id in start_node.neighbors:
                if neighbor_id == start_node_id:
                    continue
                init_angle = angle_link_calculation(start_node, topology.get_node(neighbor_id))
                search_output = trajectory_search(
                    topology, start_node_id, DNE_NODE, init_angle, speed_x, gravity_info)
                for trajectory in search_output.values():
                    for node_id in trajectory:
                        if isinstance(node_id, str) and node_id != start_node_id:
                            reachable.add(node_id)
        per_direction.append(reachable)
        union_set |= reachable
    return per_direction, union_set


def _sweep_degree_config(args_tuple: tuple) -> tuple:
    """One (min_degree, avg_target) config for the degree experiment.

    Returns ``(min_degree, actual_avg_degree, dir_sets)`` where ``dir_sets``
    is a list (one per gravity direction) of per-source reachable-id sets.
    """
    min_degree, avg_target, num_nodes, grid_size, grid_division, radius, \
        speed_x_values, gravity_directions, sweep_fraction, seed = args_tuple
    random.seed(seed * 100003 + min_degree * 7 + int(round(avg_target * 10)))
    init_topo = TOPOLOGY()
    topology, all_nodes = planar_topology_creation(
        init_topo, num_nodes, grid_size, grid_division,
        radius, min_degree, int(avg_target))
    actual_degree = sum(n.degree() for n in all_nodes) / len(all_nodes)
    n_sweep = max(1, int(round(len(all_nodes) * sweep_fraction)))
    dir_sets: list = [[] for _ in gravity_directions]
    for start_node in all_nodes[:n_sweep]:
        per_direction, _ = search_reachability_by_direction(
            topology, start_node.id, speed_x_values, gravity_directions)
        for i, reachable in enumerate(per_direction):
            dir_sets[i].append(reachable)
    return min_degree, actual_degree, dir_sets


def degree_reachability_experiment(num_nodes: int, grid_size: float, grid_division: int,
                                   radius: float, degree_configs: list,
                                   speed_x_values: list, gravity_directions: list,
                                   direction_prefixes: list, sweep_fraction: float = 1.0,
                                   seed: int = 0, workers: int = 1) -> list:
    """Paper Fig 9 style study: average reachability vs. network degree and
    number of gravity directions (mPDR).

    ``degree_configs`` is a list of ``(min_degree, avg_target)`` pairs; for
    each one a fresh planar topology is generated (the planar mesh caps
    around avg degree ~5, so the pairs are chosen to yield well-separated
    actual degrees) and ``sweep_fraction`` of its nodes are swept.  Returns
    one record per config:

    ``{"avg_degree": float, "direction_counts": list[int],
       "direction_reach": list[float]}``

    where ``direction_reach[i]`` is the average (over swept sources) of
    ``100 * |union of first (direction_counts[i]) direction sets| / (n-1)``.
    With ``workers > 1`` the configs are processed in parallel.
    """
    total = num_nodes - 1
    n_sweep = max(1, int(round(num_nodes * sweep_fraction)))
    tasks = [(md, at, num_nodes, grid_size, grid_division, radius,
              speed_x_values, gravity_directions, sweep_fraction, seed)
             for md, at in degree_configs]
    if workers > 1:
        log.info("Degree experiment: %d configs with %d worker processes",
                 len(tasks), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_sweep_degree_config, tasks, chunksize=1))
    else:
        results = [_sweep_degree_config(t) for t in tasks]
    results.sort(key=lambda r: r[0])

    records: list = []
    for min_degree, actual_degree, dir_sets in results:
        direction_reach = []
        for k in direction_prefixes:
            union_reach = 0
            for j in range(n_sweep):
                union_reach += len(set.union(*[dir_sets[i][j] for i in range(k)]))
            direction_reach.append(100.0 * union_reach / (n_sweep * total))
        records.append({"avg_degree": actual_degree,
                        "direction_counts": list(direction_prefixes),
                        "direction_reach": direction_reach})
        log.info("Degree experiment: min_degree %d -> actual avg degree %.2f | "
                 "swept %d sources | reach by mPDR size: %s",
                 min_degree, actual_degree, n_sweep,
                 ", ".join(f"{c} dirs = {r:.1f}%" for c, r in
                           zip(direction_prefixes, direction_reach)))
    return records


def reachability_bar_plot(degree_results: list, out_dir: str | None = None) -> str | None:
    """Grouped bar plot (paper Fig 9 style): average reachability (%) on the y
    axis, network average degree groups on the x axis, one bar per number of
    gravity directions (mPDR size)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(degree_results)
    n_series = len(degree_results[0]["direction_counts"])
    group_width = 0.8
    bar_width = group_width / n_series
    colors = plt.cm.viridis([i / max(1, n_series - 1) for i in range(n_series)])

    for series in range(n_series):
        counts = degree_results[0]["direction_counts"][series]
        positions = [i + (series - (n_series - 1) / 2) * bar_width for i in range(n_groups)]
        heights = [rec["direction_reach"][series] for rec in degree_results]
        bars = ax.bar(positions, heights, width=bar_width,
                      label=f"{counts} gravity direction{'s' if counts > 1 else ''}",
                      color=colors[series], edgecolor="black", linewidth=0.4)
        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"{rec['avg_degree']:.2f}" for rec in degree_results])
    ax.set_xlabel("Average node degree", fontsize=13)
    ax.set_ylabel("Average reachability (%)", fontsize=13)
    ax.set_title("PDR/mPDR Reachability vs. Network Degree and Gravity Directions",
                 fontsize=14, pad=8)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              frameon=False)
    return _finish_plot("Reachability by Degree and Gravity Directions", out_dir)


def heat_map(topology: TOPOLOGY, grid_size: int, region_division: int,
             non_reachable_output: dict) -> dict:
    """Regional reachability heat map from per-node non-reachable lists.

    ``non_reachable_output`` maps start-node id -> list of node ids that node
    cannot reach (under the union of the swept gravity directions).  The grid
    is divided into ``region_division**2`` square regions; for every
    (start region, dest region) pair the number of possible routes and the
    number reachable are tallied.

    Returns ``{(start_region, dest_region): [total, reachable, percentage]}``
    with 1-based region numbers.
    """
    all_nodes = topology.get_all_nodes()

    # Only nodes that were actually swept have reachability data; the rest are
    # excluded so the heat map reflects real measurements, not defaults.
    swept_ids = set(non_reachable_output.keys())
    all_ids = {n.id for n in all_nodes}

    # Per-swept-node reachable set: every other node minus its non-reachable list.
    reachable_by_node: dict = {}
    for start_id in swept_ids:
        blocked = {b for b in non_reachable_output.get(start_id, []) if b != start_id}
        reachable_by_node[start_id] = all_ids - {start_id} - blocked

    # Region index (1-based) of each node by its coordinates.
    cell_size = grid_size / region_division
    def region_of(node) -> int:
        col = min(int(node.x / cell_size), region_division - 1)
        row = min(int(node.y / cell_size), region_division - 1)
        return row * region_division + col + 1

    node_region = {node.id: region_of(node) for node in all_nodes}

    # Tally (start_region, dest_region) -> [possible, reachable] over swept sources.
    totals: dict = {}
    for start_id in swept_ids:
        sr = node_region[start_id]
        for dest_node in all_nodes:
            if dest_node.id == start_id:
                continue
            key = (sr, node_region[dest_node.id])
            counts = totals.setdefault(key, [0, 0])
            counts[0] += 1
            if dest_node.id in reachable_by_node[start_id]:
                counts[1] += 1

    return {key: [total, reachable,
                  (reachable / total * 100) if total else 0.0]
            for key, (total, reachable) in sorted(totals.items())}


def heat_map_plotting(topology: TOPOLOGY, grid_size: int, region_division: int,
                      reachability_region: dict, out_dir: str | None = None) -> str | None:
    """Render the regional reachability heat map (cells coloured by %)."""
    cell_size = grid_size / region_division
    fig, ax = plt.subplots()
    ax.set_title("Regional Reachability, Union of Gravity Directions", fontsize=15)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)

    for bound in range(region_division + 1):
        plt.axvline(x=bound * cell_size, color="black", linewidth=0.5)
        plt.axhline(y=bound * cell_size, color="black", linewidth=0.5)

    color_list = ["green", "forestgreen", "limegreen", "springgreen", "lime",
                  "paleturquoise", "aqua", "deepskyblue", "dodgerblue", "royalblue",
                  "blue", "slateblue", "blueviolet", "darkviolet", "violet", "pink", "lightpink"]
    # Evenly tile the full 0-100% range with the color ramp (green = best).
    n_colors = len(color_list)
    color_code: dict = {}
    for i, color in enumerate(color_list):
        hi = 100.0 * (n_colors - i) / n_colors
        lo = 100.0 * (n_colors - i - 1) / n_colors
        color_code[(hi, lo)] = color

    def region_bounds(region: int):
        row = (region - 1) // region_division
        col = (region - 1) % region_division
        return ((col * cell_size, (col + 1) * cell_size),
                (row * cell_size, (row + 1) * cell_size))

    for start_region in range(1, region_division ** 2 + 1):
        total = reachable = 0
        for dest_region in range(1, region_division ** 2 + 1):
            entry = reachability_region.get((start_region, dest_region))
            if entry:
                total += entry[0]
                reachable += entry[1]
        if total == 0:
            continue
        percentage = reachable / total * 100
        for (hi, lo), color in color_code.items():
            if lo < percentage <= hi:
                xb, yb = region_bounds(start_region)
                plt.fill_between(xb, yb[0], yb[1], facecolor=color)
                plt.text(sum(xb) / 2, sum(yb) / 2, f"{start_region}\n{percentage:.0f}%",
                         ha="center", va="center", fontsize=9)
                break

    percentage_bounds = sorted({lo for _, lo in color_code} | {100.0})
    cmap = mcolors.ListedColormap(list(reversed(color_list)), under="lightpink")
    norm = mcolors.BoundaryNorm(percentage_bounds, cmap.N)
    import matplotlib.cm as mcm
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax,
                 boundaries=percentage_bounds, extend="min", extendfrac="auto",
                 ticks=percentage_bounds, spacing="uniform", orientation="vertical",
                 label="Reachability(%)")
    return _finish_plot("Regional Reachability Heat Map", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Particle Dynamics Routing experiments")
    parser.add_argument("--nodes", type=int, default=200, help="number of topology nodes (default 200)")
    parser.add_argument("--grid-size", type=float, default=10.0)
    parser.add_argument("--grid-division", type=int, default=10)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--avg-degree", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", choices=["fast", "full"], default="fast",
                        help="fast: reduced node/speed sweep for a quick end-to-end run; "
                             "full: paper-scale sweep (long-running)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes for the reachability sweep (default 1)")
    parser.add_argument("--experiment", choices=["all", "topology", "dijkstra",
                                                 "reachability", "heatmap",
                                                 "trajectories", "reverse", "degree"],
                        default="all")
    parser.add_argument("--out", default=None, help="directory to save plot PNGs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    t_wall = time.time()

    # ---- Degree / mPDR reachability experiment (paper Fig 9 style) ----------
    if args.experiment == "degree":
        degree_speeds = list(range(-5, 6, 2)) if args.scale == "fast" else list(range(-10, 11))
        degree_gravity = [i * 2 * math.pi / 8 for i in range(8)]
        degree_results = degree_reachability_experiment(
            num_nodes=args.nodes, grid_size=args.grid_size,
            grid_division=args.grid_division, radius=args.radius,
            degree_configs=[(2, 3), (2, 4), (2, 6), (4, 6)],
            speed_x_values=degree_speeds, gravity_directions=degree_gravity,
            direction_prefixes=[1, 2, 4, 8],
            sweep_fraction=0.5 if args.scale == "fast" else 1.0,
            seed=args.seed, workers=min(4, max(1, args.workers)))
        path = reachability_bar_plot(degree_results, out_dir=args.out)
        log.info("Degree experiment results: %s",
                 [ (r["avg_degree"], r["direction_reach"]) for r in degree_results ])
        log.info("Execution time: %.3f s (wall)", time.time() - t_wall)
        return path

    # ---- Topology ----------------------------------------------------------
    init_topo = TOPOLOGY()
    topology, all_nodes = planar_topology_creation(
        init_topo, args.nodes, args.grid_size, args.grid_division,
        args.radius, args.min_degree, args.avg_degree)

    degrees = [n.degree() for n in all_nodes]
    density_mean, density_sd, density_cv = coefficient_of_variation_calculation(topology, args.grid_size)
    log.info("%d nodes | mean degree %.2f (min %d / max %d) | density %.3f nodes/cell (sd %.3f, CV %.3f)",
             args.nodes, topology.average_degree(), min(degrees), max(degrees),
             density_mean, density_sd, density_cv)

    # ---- Reachability sweep ------------------------------------------------
    if args.experiment in ("all", "reachability", "heatmap"):
        if args.scale == "fast":
            sweep_nodes = all_nodes[: min(10, len(all_nodes))]
            speed_x_values = list(range(-5, 6, 2))
            gravity_directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        else:
            sweep_nodes = all_nodes
            speed_x_values = list(range(-10, 11))
            gravity_directions = [i * 2 * math.pi / 8 for i in range(8)]
        log.info("Reachability sweep: %d start nodes, %d speeds, %d gravity directions",
                 len(sweep_nodes), len(speed_x_values), len(gravity_directions))

        tasks = [(topology, node.id, speed_x_values, gravity_directions)
                 for node in sweep_nodes]
        if max(1, args.workers) > 1:
            log.info("Sweeping with %d worker processes", args.workers)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_sweep_one_source, tasks, chunksize=1))
        else:
            results = [_sweep_one_source(task) for task in tasks]

        extensive_non_reachable: dict = {}
        reachable_count = 0
        for start_id, reachable, non_reachable in results:
            reachable_count += len(reachable)
            extensive_non_reachable[start_id] = non_reachable
            total = len(all_nodes) - 1
            log.info("Node #%s: %d/%d reachable (%.1f%%) | non-reachable: %s",
                     start_id, len(reachable), total,
                     100.0 * len(reachable) / total if total else 0.0,
                     non_reachable if len(non_reachable) <= 10 else f"{len(non_reachable)} nodes")

        possible = len(sweep_nodes) * (len(all_nodes) - 1)
        log.info("Overall reachability: %d/%d (%.2f%%)",
                 reachable_count, possible, 100.0 * reachable_count / possible if possible else 0.0)

        if args.experiment in ("all", "heatmap"):
            log.info("Heat map: region division 4")
            heat_map_result = heat_map(topology, args.grid_size, 4, extensive_non_reachable)
            for region in range(1, 17):
                total = sum(heat_map_result.get((region, d), [0, 0])[0] for d in range(1, 17))
                reachable = sum(heat_map_result.get((region, d), [0, 0])[1] for d in range(1, 17))
                if total:
                    log.info("Region %2d: %3d possible, %3d reachable, %.1f%%",
                             region, total, reachable, 100.0 * reachable / total)
            heat_map_plotting(topology, args.grid_size, 4, heat_map_result, out_dir=args.out)

    # ---- Dijkstra all-pairs shortest paths ---------------------------------
    if args.experiment in ("all", "dijkstra"):
        t_dij = time.process_time()
        all_paths = dijkstra_sp(topology)
        n_pairs = len(all_paths)
        path_lengths = [len(p) for p in all_paths.values()]
        mean_len = sum(path_lengths) / n_pairs if n_pairs else 0.0
        max_len = max(path_lengths) if path_lengths else 0
        log.info("Dijkstra all-pairs: %d pairs | mean path %.2f hops | max %d hops | "
                 "computed in %.3f s", n_pairs, mean_len, max_len,
                 time.process_time() - t_dij)
        # Hop-length distribution plot (all lengths min..max, gaps shown as 0).
        fig, ax = plt.subplots(figsize=(8, 5))
        length_counts = {}
        for length in path_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        xs = list(range(min(length_counts), max(length_counts) + 1))
        ax.bar([x - 0.4 for x in xs], [length_counts.get(x, 0) for x in xs], width=0.8,
               color="steelblue", edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xlabel("Shortest-path length (hops)", fontsize=13)
        ax.set_ylabel("Number of node pairs", fontsize=13)
        ax.set_title(f"Dijkstra All-Pairs Path-Length Distribution ({args.nodes} nodes)",
                     fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        _finish_plot("Dijkstra All-Pairs Path-Length Distribution", args.out)

    # ---- Single-node trajectory demos --------------------------------------
    if args.experiment in ("all", "trajectories", "reverse"):
        gravity_info = (GRAVITY_ACCEL, math.pi)
        start_id = all_nodes[0].id
        init_angle = 1.1631419984385158
        log.info("Trajectory search demo from node %s (angle %.3f rad, speed_x in {-5, 0, 5})",
                 start_id, init_angle)
        demo_outputs = {}
        for speed_x in (-5, 0, 5):
            # trajectory_search sweeps launch speeds and returns
            # {launch_speed: [node ids]}; flatten so each distinct
            # trajectory is plotted as its own curve.
            for launch_speed, trajectory in trajectory_search(
                    topology, start_id, DNE_NODE, init_angle, speed_x, gravity_info
            ).items():
                demo_outputs[(speed_x, round(launch_speed, 2))] = trajectory
        multiple_trajectories_plot(topology, demo_outputs,
                                   title=f"Trajectory Search from Node {start_id}",
                                   out_dir=args.out)

    if args.experiment in ("all", "reverse"):
        start_id = all_nodes[0].id
        demo_params = find_reverse_demo_params(topology, start_id)
        if demo_params is None:
            log.warning("No two launch profiles share a downstream node from node %s; "
                        "reverse demo skipped.", start_id)
        else:
            (angle_1, speed_x_1, speed_y_1), (angle_2, speed_x_2, speed_y_2) = demo_params
            log.info("Reverse trajectory demo from node %s: launch profiles "
                     "((%.3f rad, vx %g, vy %g), (%.3f rad, vx %g, vy %g))",
                     start_id, angle_1, speed_x_1, speed_y_1, angle_2, speed_x_2, speed_y_2)
            reverse_trajectory_algorithm(topology, start_id, demo_params, out_dir=args.out)

    log.info("Execution time: %.3f s (wall, %d workers)",
             time.time() - t_wall, max(1, args.workers))


if __name__ == "__main__":
    main()
