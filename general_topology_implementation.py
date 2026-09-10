"""Graph data structures for Particle Dynamics Routing (PDR).

Defines the two core containers used by every PDR experiment:

* ``NODE``    -- a single vertex with an integer string id and planar
  coordinates ``(x, y)``.
* ``TOPOLOGY`` -- an adjacency map of nodes.

Key convention (shared with ``planar_topology_implementation`` and the
routing code in ``gravitational_algorithms``):
    A node's ``neighbors`` list contains the ids of its adjacent nodes *and
    its own id* (inserted at index 0 by the topology builders).  This is why
    ``NODE.degree()`` is ``len(neighbors) - 1`` and why routing loops skip the
    self-entry via ``if neighbor != curr_node``.
"""

import math

import matplotlib
matplotlib.use("Agg")  # headless-safe: plotting helpers must never require a display
import matplotlib.pyplot as plt

__all__ = ["NODE", "TOPOLOGY"]


class NODE:
    """A single planar graph vertex.

    Attributes:
        id: Unique node identifier (string form of an integer).
        neighbors: Ids of adjacent nodes, with the node's own id at index 0.
        x, y: Planar coordinates.
    """

    __slots__ = ("id", "neighbors", "x", "y")

    def __init__(self, node_id: str, x: float, y: float) -> None:
        self.id = node_id
        self.neighbors: list = []
        self.x, self.y = x, y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NODE):
            return NotImplemented
        return (self.id == other.id and self.x == other.x and self.y == other.y
                and set(self.neighbors) == set(other.neighbors))

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        neighbor_ids = ", ".join(self.neighbors)
        return f"<NODE id='{self.id}' neighbors=[{neighbor_ids}]>"

    def __str__(self) -> str:
        return self.__repr__()

    def degree(self) -> int:
        """Number of true links (the self-entry in ``neighbors`` is not a link)."""
        return len(self.neighbors) - 1

    def node_neighbors(self) -> list:
        """All neighbor ids including the self-entry (index 0)."""
        return self.neighbors

    def geographical_distance(self, other: "NODE") -> float:
        """Euclidean distance to ``other``."""
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)


class TOPOLOGY:
    """Adjacency container of :class:`NODE` objects keyed by node id.

    Attributes:
        size: Number of nodes stored.
        nodes: Mapping of node id -> NODE.
        plot_show: When True, :meth:`plot_topology` shows the figure
            interactively instead of only saving it.
    """

    __slots__ = ("size", "nodes", "plot_show")

    def __init__(self, plt_show: bool = False) -> None:
        self.size = 0
        self.nodes: dict = {}
        self.plot_show = plt_show

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TOPOLOGY):
            return NotImplemented
        if self.size != other.size or len(self.nodes) != len(other.nodes):
            return False
        for node_id, node in self.nodes.items():
            other_node = other.nodes.get(node_id)
            if other_node is None:
                return False
            if set(node.neighbors) != set(other_node.neighbors):
                return False
        return True

    def __repr__(self) -> str:
        return f"TOPOLOGY(size={self.size}, node_ids={sorted(self.nodes)})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_node(self, node_id: str) -> NODE:
        """Return the node with ``node_id`` (KeyError if absent)."""
        return self.nodes[node_id]

    def get_all_nodes(self) -> list:
        """Every node in insertion order."""
        return list(self.nodes.values())

    def get_neighbors(self, node: NODE) -> tuple:
        """Return ``(node, neighbor_ids)`` -- neighbor_ids includes the self-entry."""
        return node, self.nodes[node.id].neighbors

    def get_all_nodes_and_neighbors(self) -> dict:
        """Mapping of node id -> neighbor id list (self-entry included)."""
        return {node_id: node.neighbors for node_id, node in self.nodes.items()}

    def average_degree(self) -> float:
        """Mean node degree (the self-entry excluded by ``NODE.degree``)."""
        if not self.nodes:
            return 0.0
        return sum(node.degree() for node in self.nodes.values()) / len(self.nodes)

    def add_to_topology(self, node_1: NODE, node_2: NODE | None = None) -> None:
        """Insert a node, or a node and an undirected link to another node.

        With a single argument the node is added if not already present.  With
        two arguments both nodes are ensured present and the link is added
        (idempotently) to both adjacency lists.
        """
        if node_2 is None:
            if node_1.id not in self.nodes:
                self.nodes[node_1.id] = node_1
                self.size += 1
            return

        if node_1.id == node_2.id:
            if node_1.id not in self.nodes:
                self.nodes[node_1.id] = node_1
                self.size += 1
            return

        if node_1.id not in self.nodes:
            self.nodes[node_1.id] = node_1
            self.size += 1
        if node_2.id not in self.nodes:
            self.nodes[node_2.id] = node_2
            self.size += 1

        if node_2.id not in self.nodes[node_1.id].neighbors:
            self.nodes[node_1.id].neighbors.append(node_2.id)
        if node_1.id not in self.nodes[node_2.id].neighbors:
            self.nodes[node_2.id].neighbors.append(node_1.id)

    def delete_link_from_topology(self, node_1: NODE, node_2: NODE) -> None:
        """Remove the undirected link between ``node_1`` and ``node_2``.

        No-op if the link does not exist.
        """
        if node_2.id in self.nodes[node_1.id].neighbors:
            self.nodes[node_1.id].neighbors.remove(node_2.id)
        if node_1.id in self.nodes[node_2.id].neighbors:
            self.nodes[node_2.id].neighbors.remove(node_1.id)

    def plot_topology(self, out_path: str | None = None) -> str | None:
        """Draw the topology (nodes + links). Returns the saved path.

        Headless-safe: saves a PNG by default; with ``plot_show=True`` the
        figure is also shown interactively.
        """
        all_node_list = self.get_all_nodes()
        link_plot_x: list = []
        link_plot_y: list = []
        data = {"x": [], "y": []}
        for node in all_node_list:
            data["x"].append(node.x)
            data["y"].append(node.y)
            neighbor_ids = self.get_neighbors(node)[1]
            for neighbor_id in neighbor_ids:
                if neighbor_id == node.id:
                    continue
                neighbor = self.get_node(neighbor_id)
                link_plot_x += [node.x, neighbor.x]
                link_plot_y += [node.y, neighbor.y]

        plt.figure()
        plt.title("Topology Implemented", fontsize=20)
        plt.xlabel("x-axis", fontsize=20)
        plt.ylabel("y-axis", fontsize=20)
        plt.scatter(data["x"], data["y"], marker="o", color="cornflowerblue",
                    linewidth=5)
        for i in range(0, len(link_plot_x), 2):
            plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2],
                     color="lightsteelblue")
        saved = plt.savefig(out_path) if out_path else None
        if self.plot_show:
            plt.show()
        else:
            plt.close()
        return saved
