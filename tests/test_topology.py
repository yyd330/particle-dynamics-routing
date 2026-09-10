"""Tests for the NODE and TOPOLOGY data structures."""

from general_topology_implementation import NODE, TOPOLOGY


def make_node(node_id: str, x: float = 0.0, y: float = 0.0) -> NODE:
    node = NODE(node_id, x, y)
    node.neighbors.insert(0, node.id)  # project convention: self-entry at index 0
    return node


def test_node_degree_counts_only_true_links():
    node = make_node("1", 1.0, 2.0)
    node.neighbors += ["2", "3"]
    assert node.degree() == 2


def test_node_geographical_distance():
    assert make_node("1", 0.0, 0.0).geographical_distance(
        NODE("2", 3.0, 4.0)) == 5.0


def test_node_equality_and_hash():
    a = make_node("1", 1.0, 2.0)
    b = make_node("1", 1.0, 2.0)
    c = make_node("2", 1.0, 2.0)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_topology_add_single_node_idempotent():
    topo = TOPOLOGY()
    topo.add_to_topology(make_node("1"))
    topo.add_to_topology(make_node("1", 5.0, 5.0))  # duplicate: ignored
    assert topo.size == 1
    assert topo.get_node("1").x == 0.0


def test_topology_add_link_is_bidirectional_and_idempotent():
    topo = TOPOLOGY()
    a, b = make_node("1"), make_node("2")
    topo.add_to_topology(a, b)
    topo.add_to_topology(a, b)  # no duplicate edge
    assert "2" in a.neighbors and "1" in b.neighbors
    assert topo.size == 2
    assert a.degree() == 1 and b.degree() == 1


def test_topology_delete_link():
    topo = TOPOLOGY()
    a, b = make_node("1"), make_node("2")
    topo.add_to_topology(a, b)
    topo.delete_link_from_topology(a, b)
    assert "2" not in a.neighbors
    assert "1" not in b.neighbors
    # no-op on missing link
    topo.delete_link_from_topology(a, b)


def test_topology_get_neighbors_returns_self_entry():
    topo = TOPOLOGY()
    a, b = make_node("1"), make_node("2")
    topo.add_to_topology(a, b)
    node, neighbor_ids = topo.get_neighbors(a)
    assert node is a
    assert "1" in neighbor_ids and "2" in neighbor_ids


def test_topology_average_degree_empty():
    assert TOPOLOGY().average_degree() == 0.0


def test_topology_equality():
    def build():
        topo = TOPOLOGY()
        a, b = make_node("1"), make_node("2")
        topo.add_to_topology(a, b)
        return topo

    assert build() == build()
    other = build()
    other.add_to_topology(make_node("3"))
    assert build() != other
