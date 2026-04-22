from typing import TypeVar
import math
import matplotlib.pyplot as plt

T = TypeVar('T')
Matrix = TypeVar('Matrix')
Node = TypeVar('Node')
Topology = TypeVar('Topology')


class NODE:
    """Creation of node"""

    __slots__ = ['id', 'neighbors', 'x', 'y']

    def __init__(self, node_id: str, x: float, y: float):
        self.id = node_id
        self.neighbors = []
        self.x, self.y = x, y

    def __eq__(self, other: Node):
        if self.id != other.id:
            return False
        elif self.x != other.x:
            return False
        elif self.y != other.y:
            return False
        elif self.neighbors != other.neighbors:
            return False
        return True

    def __repr__(self):
        output_str = ""
        for node in self.neighbors:
            output_str += node
            output_str += ' '
        return f"<id: '{self.id}'" + ", Neighbors: " + "".join(output_str) + ">"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(self.id)

    def degree(self):
        return len(self.neighbors) - 1

    def node_neighbors(self):
        return self.neighbors

    def geographical_distance(self, other: Node):
        return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)


class TOPOLOGY:
    """Creation of topology"""

    __slots__ = ['size', 'nodes', 'plot_show']

    def __init__(self, plt_show: bool = False):
        self.size = 0
        self.nodes = {}  # {node_id: NODE}
        self.plot_show = plt_show

    def __eq__(self, other):
        if self.size != other.size or len(self.nodes) != len(other.nodes):
            return False
        else:
            for node_id, node in self.nodes.items():
                other_node = other.get_node(node_id)
                if other_node is None:
                    return False

                neighbors_set = set(node.neighbors)
                other_neighbors_set = set(other_node.neighbors)

                if not neighbors_set == other_neighbors_set:
                    return False
        return True

    def __repr__(self):
        return "Size: " + str(self.size) + ", Nodes: " + str(list(self.nodes.items()))

    def __str__(self):
        return repr(self)

    def average_degree(self):
        all_nodes = self.get_all_nodes()
        degree_count = 0
        for node in all_nodes:
            degree_count += node.degree()
        num_nodes = len(all_nodes)
        return degree_count / num_nodes

    def plot_topology(self):
        # Initialization
        all_node_list = self.get_all_nodes()
        link_plot_x = []
        link_plot_y = []

        data = {"x": [], "y": [], "id": []}
        for node in all_node_list:
            data["x"].append(node.x)
            data["y"].append(node.y)
            data["id"].append(node.id)

            node_neighbors = self.get_neighbors(node)[1]
            for neighbor_id in node_neighbors:
                neighbor = self.get_node(neighbor_id)
                link_plot_x.append(node.x)
                link_plot_y.append(node.y)
                link_plot_x.append(neighbor.x)
                link_plot_y.append(neighbor.y)

        # Show Nodes
        plt.figure()
        plt.title('Topology Implemented', fontsize=20)
        plt.xlabel('x-axis', fontsize=20)
        plt.ylabel('y-axis', fontsize=20)
        plt.scatter(data["x"], data["y"], marker='o', color='cornflowerblue', linewidth=5)
        # Show Node ids
        # for label, x, y in zip(data["id"], data["x"], data["y"]):
            # plt.annotate(label, xy=(x, y), fontsize=20)

        # Show Connections
        for i in range(0, len(link_plot_x), 2):
            plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2], color='lightsteelblue')

        # Show the Topology
        plt.show()

    def get_node(self, node_id: str):
        return self.nodes[node_id]

    def get_all_nodes(self):
        output = []
        for node in self.nodes:
            n = self.nodes[node]
            output.append(n)
        return output

    def get_neighbors(self, node: Node):
        return node, self.nodes[node.id].neighbors

    def get_all_nodes_and_neighbors(self):
        output = dict()
        for node_id in self.nodes:
            output[node_id] = self.nodes[node_id].neighbors
        return output

    def add_to_topology(self, node_1: Node, node_2: Node = None):
        if node_2 is not None:
            if node_1.id not in self.nodes and node_2.id not in self.nodes:
                if node_1.id == node_2.id:
                    self.nodes[node_1.id] = node_1
                    self.size += 1
                else:
                    self.nodes[node_1.id] = node_1
                    self.nodes[node_2.id] = node_2

                    self.nodes[node_1.id].neighbors.append(node_2.id)
                    self.nodes[node_2.id].neighbors.append(node_1.id)

                    self.size += 2

            elif node_1.id in self.nodes and node_2.id not in self.nodes:
                self.nodes[node_2.id] = node_2

                self.nodes[node_1.id].neighbors.append(node_2.id)
                self.nodes[node_2.id].neighbors.append(node_1.id)

                self.size += 1

            elif node_1.id not in self.nodes and node_2.id in self.nodes:
                self.nodes[node_1.id] = node_1

                self.nodes[node_1.id].neighbors.append(node_2.id)
                self.nodes[node_2.id].neighbors.append(node_1.id)

                self.size += 1

            else:
                if node_2.id not in self.nodes[node_1.id].neighbors:
                    self.nodes[node_1.id].neighbors.append(node_2.id)
                if node_1.id not in self.nodes.get(node_2.id).neighbors:
                    self.nodes[node_2.id].neighbors.append(node_1.id)
        else:
            if node_1.id not in self.nodes:
                self.nodes[node_1.id] = node_1
                self.size += 1

    def delete_link_from_topology(self, node_1: Node, node_2: Node):
        if node_2.id in self.nodes[node_1.id].neighbors:
            self.nodes[node_1.id].neighbors.remove(node_2.id)
            self.nodes[node_2.id].neighbors.remove(node_1.id)
