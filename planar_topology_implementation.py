"""Planar topology implementation for Particle Dynamics Routing (PDR).

This module reconstructs the planar-mesh topology used by the PDR experiments
in:  S. Biswas, Y. Yang, A. K. Bhuyan, H. Dutta, S. Datta, "Leveraging particle
dynamics in force-fields for network packet routing", PLOS ONE 21(8): e0357202
(2026).

It provides the planar topology generation pipeline and the helper predicates
that the routing algorithms in ``gravitational_algorithms.py`` depend on.

Key convention (shared with ``general_topology_implementation.TOPOLOGY``):
    A node's ``neighbors`` list contains the ids of its adjacent nodes *and its
    own id*.  This is why ``NODE.degree()`` is defined as ``len(neighbors) - 1``
    and why the routing loops skip ``if neighbor != curr_node``.  All functions
    below respect this convention.
"""

import math
import random
from typing import Tuple

from general_topology_implementation import NODE, TOPOLOGY

# Grid boundary margin used when placing nodes (keeps them strictly inside the
# [0, grid_size] square so PDR "edge node" logic in the routing code is valid).
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def angle_link_calculation(node_a: NODE, node_b: NODE) -> float:
    """Angle (radians) of the directed link node_a -> node_b.

    Implements Eq. (7) of the paper:
        theta_a-b = atan2(y_a - y_b, x_a - x_b)

    atan2 is used (rather than atan) so the full 0..2*pi range is covered and the
    quadrant of the link is unambiguous.  The result is normalised to [0, 2*pi).
    """
    angle = math.atan2(node_a.y - node_b.y, node_a.x - node_b.x)
    if angle < 0:
        angle += 2 * math.pi
    return angle


def _segment_intersects(p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Return True if closed segment p1-p2 properly crosses segment p3-p4."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if v > _EPS:
            return 1
        if v < -_EPS:
            return -1
        return 0

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    # Collinear-on-segment cases (degenerate overlaps).
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def _on_segment(a, b, p):
    return (min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS and
            min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def _edge_crosses_existing(topology: TOPOLOGY, u: str, v: str) -> bool:
    """True if adding edge (u, v) would cross any *non-adjacent* existing edge.

    Edges that share a vertex with (u, v) (i.e. edges incident to u or to v) are
    adjacent by definition and are allowed to meet there, so they are skipped.
    """
    node_u = topology.get_node(u)
    node_v = topology.get_node(v)
    u_pt, v_pt = (node_u.x, node_u.y), (node_v.x, node_v.y)
    for node in topology.get_all_nodes():
        for nbr_id in node.neighbors:
            if nbr_id == node.id:
                continue
            a, b = node.id, nbr_id
            if {a, b} == {u, v}:
                continue
            # Skip adjacent edges (sharing u or v) — they meet, they don't cross.
            if a in (u, v) or b in (u, v):
                continue
            node_a = topology.get_node(a)
            node_b = topology.get_node(b)
            if _segment_intersects(u_pt, v_pt, (node_a.x, node_a.y), (node_b.x, node_b.y)):
                return True
    return False


# --------------------------------------------------------------------------- #
# Topology construction helpers                                               #
# --------------------------------------------------------------------------- #
def _place_nodes(num_nodes: int, grid_size: float, grid_division: int) -> list:
    """Place ``num_nodes`` nodes in a jittered grid.

    The [0, grid_size]^2 square is divided into ``grid_division`` x
    ``grid_division`` cells and nodes are distributed as evenly as possible
    across the cells (with random jitter), giving a near-uniform spatial
    distribution like the planar mesh used in the paper.
    """
    n_cells = grid_division * grid_division
    # Even distribution: floor(nodes/cells) per cell, the remainder of the
    # nodes fill one extra position in the first cells.  (No per-cell floor:
    # with fewer nodes than cells each node gets its own cell.)
    per_cell = num_nodes // n_cells
    remainder = num_nodes - per_cell * n_cells
    if per_cell == 0:
        # Spread the nodes across the cells instead of clustering them in the
        # first `remainder` cells.
        step = n_cells // num_nodes
        chosen = [cell * step for cell in range(num_nodes)]
    else:
        chosen = None

    positions = []
    cell_size = grid_size / grid_division
    for cell in range(n_cells):
        if chosen is not None:
            count = chosen.count(cell)
        else:
            count = per_cell + (1 if cell < remainder else 0)
        for _ in range(count):
            cx = (cell % grid_division) * cell_size
            cy = (cell // grid_division) * cell_size
            x = cx + random.random() * cell_size
            y = cy + random.random() * cell_size
            # Keep strictly inside the boundary.
            x = min(max(x, _EPS), grid_size - _EPS)
            y = min(max(y, _EPS), grid_size - _EPS)
            positions.append((x, y))
    return positions


def _add_planar_link(topology: TOPOLOGY, node_u: NODE, node_v: NODE) -> bool:
    """Add edge (u, v) if it does not break planarity. Returns True if added."""
    if node_u.id == node_v.id:
        return False
    if node_v.id in node_u.neighbors:
        return True
    if _edge_crosses_existing(topology, node_u.id, node_v.id):
        return False
    topology.add_to_topology(node_u, node_v)
    return True


def _knn_links(topology: TOPOLOGY, all_nodes: list, k: int) -> None:
    """Seed the mesh with k-nearest-neighbor links (kept planar)."""
    for node in all_nodes:
        distances = []
        for other in all_nodes:
            if other.id == node.id:
                continue
            distances.append((node.geographical_distance(other), other))
        distances.sort(key=lambda t: t[0])
        for _, other in distances[:k]:
            if node.id in other.neighbors:
                continue
            _add_planar_link(topology, node, other)


def link_dictionary(topology: TOPOLOGY) -> dict:
    """Map every node id to its neighbor ids (excluding the node itself)."""
    output = {}
    for node in topology.get_all_nodes():
        output[node.id] = [nid for nid in node.neighbors if nid != node.id]
    return output


def check_continuity(topology: TOPOLOGY) -> bool:
    """Return True if the topology is fully connected (single component)."""
    all_nodes = topology.get_all_nodes()
    if not all_nodes:
        return True
    seen = {all_nodes[0].id}
    stack = [all_nodes[0].id]
    while stack:
        cur = stack.pop()
        for nid in topology.get_node(cur).neighbors:
            if nid == cur:
                continue
            if nid not in seen:
                seen.add(nid)
                stack.append(nid)
    return len(seen) == len(all_nodes)


def dfs(topology: TOPOLOGY, start_id: str) -> list:
    """Depth-first search returning the visit order (node ids) from ``start_id``."""
    seen = {start_id}
    order = [start_id]
    stack = [start_id]
    while stack:
        cur = stack.pop()
        for nid in topology.get_node(cur).neighbors:
            if nid == cur:
                continue
            if nid not in seen:
                seen.add(nid)
                order.append(nid)
                stack.append(nid)
    return order


def check_planar(topology: TOPOLOGY) -> bool:
    """Return True if the straight-line drawing of the topology is planar.

    A graph admits a planar embedding if no two of its non-adjacent edges cross.
    We test exactly that on the given coordinates: the drawing is planar when no
    two edges geometrically intersect in their interiors.  Our construction
    rejects any link that would cross an existing one, so this holds by design.
    """
    edges = []
    seen = set()
    for node in topology.get_all_nodes():
        for nbr_id in node.neighbors:
            if nbr_id == node.id:
                continue
            key = tuple(sorted((node.id, nbr_id)))
            if key in seen:
                continue
            seen.add(key)
            other = topology.get_node(nbr_id)
            edges.append(((node.x, node.y), (other.x, other.y)))
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            # Adjacent edges (sharing a vertex) are allowed to meet there.
            shared = set(edges[i]) & set(edges[j])
            if shared:
                continue
            if _segment_intersects(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True


def delete_overlap(topology: TOPOLOGY) -> int:
    """Remove crossing (overlapping) links. Returns the number removed."""
    removed = 0
    changed = True
    while changed:
        changed = False
        for node in topology.get_all_nodes():
            for nbr_id in list(node.neighbors):
                if nbr_id == node.id:
                    continue
                if _edge_crosses_existing(topology, node.id, nbr_id):
                    topology.delete_link_from_topology(node, topology.get_node(nbr_id))
                    removed += 1
                    changed = True
                    break
    return removed


def achieve_minimum_degree(topology: TOPOLOGY, min_degree: int) -> None:
    """Ensure every node has at least ``min_degree`` links.

    A node below the threshold is connected to its closest non-neighbors
    (keeping the mesh planar) until the threshold is met.
    """
    for node in topology.get_all_nodes():
        attempts = 0
        while node.degree() < min_degree and attempts < 50:
            attempts += 1
            candidates = []
            for other in topology.get_all_nodes():
                if other.id == node.id or other.id in node.neighbors:
                    continue
                candidates.append((node.geographical_distance(other), other))
            if not candidates:
                break
            candidates.sort(key=lambda t: t[0])
            added = False
            for _, other in candidates:
                if _add_planar_link(topology, node, other):
                    added = True
                    break
            if not added:
                break


def achieve_average_degree(topology: TOPOLOGY, target_average_degree: int) -> None:
    """Nudge the mesh average degree toward ``target_average_degree``.

    If the average is below the target, low-degree nodes are connected to their
    closest non-neighbors (kept planar) until the target is reached.  If the
    average is above the target, the longest leaf/degree-2 edges are pruned.
    """
    # Raise the average.
    while topology.average_degree() < target_average_degree:
        before = topology.average_degree()
        # Connect the lowest-degree node to its closest non-neighbor.
        node = min((n for n in topology.get_all_nodes()), key=lambda n: n.degree())
        candidates = []
        for other in topology.get_all_nodes():
            if other.id == node.id or other.id in node.neighbors:
                continue
            candidates.append((node.geographical_distance(other), other))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        added = False
        for _, other in candidates:
            if _add_planar_link(topology, node, other):
                added = True
                break
        if not added or topology.average_degree() == before:
            break

    # Lower the average (rare with the kNN seed).
    while topology.average_degree() > target_average_degree:
        before = topology.average_degree()
        removable = []
        for node in topology.get_all_nodes():
            for nbr_id in node.neighbors:
                if nbr_id == node.id or node.id > nbr_id:
                    continue
                removable.append((node.geographical_distance(topology.get_node(nbr_id)),
                                  node, topology.get_node(nbr_id)))
        if not removable:
            break
        removable.sort(key=lambda t: t[0], reverse=True)
        for _, a, b in removable:
            if a.degree() > 1 and b.degree() > 1:
                topology.delete_link_from_topology(a, b)
                break
        if topology.average_degree() == before:
            break


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def coefficient_of_variation_calculation(topology: TOPOLOGY, grid_size: float) -> tuple:
    """Return (mean, standard_deviation, coefficient_of_variation) of node density.

    The [0, grid_size]^2 area is divided into a 10x10 grid of cells; the number of
    nodes per cell is treated as the density sample.  A low coefficient of
    variation indicates a spatially uniform node placement.
    """
    num_nodes = len(topology.get_all_nodes())
    if num_nodes == 0:
        return 0.0, 0.0, 0.0

    cells = 10 * 10
    cell_size = grid_size / 10
    counts = [0] * cells
    for node in topology.get_all_nodes():
        cx = min(9, max(0, int(node.x / cell_size)))
        cy = min(9, max(0, int(node.y / cell_size)))
        counts[cy * 10 + cx] += 1

    mean = sum(counts) / cells
    variance = sum((c - mean) ** 2 for c in counts) / cells
    std_dev = math.sqrt(variance)
    cov = (std_dev / mean) if mean else 0.0
    return round(mean, 3), round(std_dev, 3), round(cov, 3)


# --------------------------------------------------------------------------- #
# Top-level pipeline                                                          #
# --------------------------------------------------------------------------- #
def planar_topology_creation(topology: TOPOLOGY, num_nodes: int, grid_size: float,
                             grid_division: int, radius: float, min_degree: int,
                             avg_degree: int) -> tuple:
    """Generate a planar mesh topology and return (topology, all_nodes).

    Pipeline:
        1. Place ``num_nodes`` nodes uniformly on the [0, grid_size]^2 plane.
        2. Seed k-nearest-neighbor links (kept planar).
        3. Remove any overlapping (crossing) links.
        4. Enforce the minimum node degree.
        5. Adjust toward the target average degree.
        6. Guarantee connectivity (add a spanning edge if needed).
    """
    positions = _place_nodes(num_nodes, grid_size, grid_division)

    all_nodes = []
    for index, (x, y) in enumerate(positions):
        node = NODE(str(index), x, y)
        topology.add_to_topology(node)
        # Convention: a node's own id occupies slot 0 of its neighbor list so
        # that NODE.degree() (== len(neighbors) - 1) counts only true links and
        # the routing loops can skip the self-entry via `neighbor != curr_node`.
        node.neighbors.insert(0, node.id)
        all_nodes.append(node)

    # kNN seed.  k is chosen from the target average degree (each undirected edge
    # contributes 2 to the summed degree), keeping the mesh planar.
    k = max(2, min(6, int(avg_degree)))
    _knn_links(topology, all_nodes, k)

    delete_overlap(topology)
    achieve_minimum_degree(topology, min_degree)
    achieve_average_degree(topology, avg_degree)

    # Guarantee a single connected component.
    if not check_continuity(topology):
        _ensure_connected(topology, all_nodes)
        achieve_minimum_degree(topology, min_degree)
        delete_overlap(topology)

    # Sort by id (strings "0".."n-1") for deterministic downstream iteration.
    all_nodes.sort(key=lambda n: int(n.id))
    return topology, all_nodes


def _ensure_connected(topology: TOPOLOGY, all_nodes: list) -> None:
    """Connect any isolated component to the main one with the shortest bridge."""
    while not check_continuity(topology):
        # Find two components.
        comp_a, comp_b = set(), set()
        visited = set()
        for node in all_nodes:
            if node.id in visited:
                continue
            component = set(dfs(topology, node.id))
            visited |= component
            if not comp_a:
                comp_a = component
            elif node.id not in comp_a:
                comp_b = component
                break
        if not comp_b:
            break
        # Bridge with the shortest possible edge (kept planar when possible).
        best = None
        for a in all_nodes:
            if a.id not in comp_a:
                continue
            for b in all_nodes:
                if b.id not in comp_b:
                    continue
                d = a.geographical_distance(b)
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        topology.add_to_topology(a, b)
