from general_topology_implementation import NODE, TOPOLOGY
from planar_topology_implementation import angle_link_calculation, planar_topology_creation, delete_overlap, \
    link_dictionary, achieve_minimum_degree, check_planar, achieve_average_degree, dfs, check_continuity, \
    coefficient_of_variation_calculation
from gravitational_algorithms import trajectory_algorithm, single_trajectory_plot, trajectory_search, \
    multiple_trajectories_plot, speed_xy_calculation, heat_map, dijkstra_sp

import math
import random
import time
import sys
import linecache
import ast


def main():
    start_time = time.process_time()
    random.seed(0)

    # Create Topology and setup
    init_topo = TOPOLOGY()
    grid_size = 10
    grid_division_num = 10
    num_node = 200
    radius = 2
    min_deg = 2
    avg_deg = 5

    acc_gravity = 10  # m/s^2

    topology, all_nodes = planar_topology_creation(init_topo, num_node, grid_size, grid_division_num, radius, min_deg,
                                                   avg_deg)

    # Coefficient of Variation Calculation
    mean, standard_deviation, coefficient_variation = \
        coefficient_of_variation_calculation(topology, grid_size)
    print()
    print(num_node, "Nodes Total.")
    print("Mean:", mean)
    print("Standard Deviation:", standard_deviation)
    print("Coefficient of Variation:", coefficient_variation)
    print()

    # Average Degree
    sum_deg = 0
    min_comp = 100000
    max_comp = 0
    for node in all_nodes:
        sum_deg += node.degree()
        if node.degree() > max_comp:
            max_comp = node.degree()
        if node.degree() < min_comp:
            min_comp = node.degree()
    avg_deg = topology.average_degree()
    print("Topology Information: ")
    print("   Average degree: " + str(avg_deg))
    print("   Maximum degree: " + str(max_comp))
    print("   Minimum degree: " + str(min_comp))

    all_node_list = topology.get_all_nodes()
    # Runtime Check
    print()
    print("Execution Time: " + str(time.process_time() - start_time) + " seconds")


if __name__ == '__main__':
    main()
