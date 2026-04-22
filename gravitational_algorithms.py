from general_topology_implementation import NODE, TOPOLOGY
from planar_topology_implementation import angle_link_calculation, planar_topology_creation, delete_overlap, \
    link_dictionary, achieve_minimum_degree, check_planar, achieve_average_degree, dfs, check_continuity, \
    coefficient_of_variation_calculation

import math
import random
import time
import sys
import array
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import operator
import numpy as np
import heapq

def trajectory_algorithm(topology: TOPOLOGY, curr_node_id: str, dest_node_id: str, curr_speed_x: float,
                         curr_speed_y: float, curr_angle_theta_n: float, gravity_vector: tuple, scalar_dt: int,
                         run_time: int, output: list):
    delta_t = 0.01
    run_time += 1
    output.append(curr_node_id)

    curr_node = topology.get_node(curr_node_id)

    next_node_found = 0
    not_reachable = 0

    acc_gravity = gravity_vector[0]
    gravity_direction = gravity_vector[1]

    next_speed_x = next_speed_y = next_angle_theta_n = 0.0
    next_node_id = ''

    if len(output) == 1:
        next_node_found = 1
        link_angles = []
        neighbor_nodes_ids = []

        neighbors_ids_list = curr_node.neighbors
        for neighbor_id in neighbors_ids_list:
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
        node_desired = neighbor_nodes_ids[node_desired_index]
        next_node_id = node_desired
        next_node = topology.get_node(next_node_id)
        link_length = curr_node.geographical_distance(next_node)

        curr_velocity = math.sqrt(math.pow(curr_speed_x, 2) + math.pow(curr_speed_y, 2))
        if curr_velocity != 0:
            scalar_dt = (link_length / curr_velocity) / delta_t
        else:
            scalar_dt = link_length / delta_t
        next_speed_x = curr_speed_x
        next_speed_y = curr_speed_y - acc_gravity * scalar_dt * delta_t
        next_angle_theta_n = curr_angle_theta_n

    while next_node_found == 0:
        g_angle_difference = gravity_direction - (3 * math.pi / 2)

        abstract_point_x = curr_node.x + curr_speed_x * scalar_dt * delta_t
        abstract_point_y = curr_node.y + curr_speed_y * scalar_dt * delta_t - 0.5 * acc_gravity * math.pow((scalar_dt * delta_t), 2)

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

        neighbors_ids_list = curr_node.neighbors
        for neighbor_id in neighbors_ids_list:
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
        node_desired = neighbor_nodes_ids[node_desired_index]
        next_node_id = node_desired
        next_node = topology.get_node(next_node_id)
        link_length = curr_node.geographical_distance(next_node)
        distance_p_n = math.sqrt(math.pow((abstract_point_y - curr_node.y), 2) + math.pow((abstract_point_x - curr_node.x), 2))

        # Change sy only, sx is fixed.
        if distance_p_n >= link_length:
            next_speed_x = curr_speed_x
            next_speed_y = curr_speed_y - acc_gravity * scalar_dt * delta_t
            next_angle_theta_n = angle_link_calculation(curr_node, next_node)
            next_node_found = 1

            # Stopping Criteria Attempt
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
            scalar_dt += 1
    '''
    # Debug section
    print("Current Node:", curr_node_id)
    print("Incoming Horizontal Velocity:", curr_speed_x)
    print("Incoming Vertical Velocity:", curr_speed_y)
    print("Incoming Angle:", curr_angle_theta_n)
    # print("Point p Angle:", angle_theta_p)
    print("Links:", link_angles)
    print("Link Comparison:", link_diff)
    print("Next node:", next_node_id)
    print("Outgoing Horizontal Velocity:", next_speed_x)
    print("Outgoing Vertical Velocity:", next_speed_y)
    print("Outgoing angle:", next_angle_theta_n)
    print("Run Time:", run_time)
    print()
    '''

    if dest_node_id == next_node_id:
        output.append(next_node_id)
        output.append([curr_node_id, curr_speed_x, curr_speed_y])
    else:
        if run_time <= 900:
            if not_reachable == 0:
                trajectory_algorithm(topology, next_node_id, dest_node_id, next_speed_x,
                           next_speed_y, next_angle_theta_n, gravity_vector, 1, run_time, output)
        else:
            output.append(next_node_id)

    return output


def reverse_trajectory_algorithm(topology: TOPOLOGY, start_node_id: str, start_info_tuple: tuple):

    def reverse_trajectory(curr_node_id: str, dest_node_id: str, curr_speed_x: float, curr_speed_y: float,
                           curr_angle_theta_n: float, gravity_vector: tuple, scalar_dt: int,
                             run_time: int, output: list):
        delta_t = 0.01
        run_time += 1
        output.append(curr_node_id)

        curr_node = topology.get_node(curr_node_id)

        next_node_found = 0
        not_reachable = 0

        acc_gravity = gravity_vector[0]
        gravity_direction = gravity_vector[1]

        next_speed_x = next_speed_y = next_angle_theta_n = 0.0
        next_node_id = ''

        while next_node_found == 0:
            g_angle_difference = gravity_direction - (3 * math.pi / 2)
            abstract_point_x = curr_node.x - curr_speed_x * scalar_dt * delta_t
            abstract_point_y = curr_node.y - curr_speed_y * scalar_dt * delta_t + 0.5 * acc_gravity * math.pow((scalar_dt * delta_t), 2)

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

            neighbors_ids_list = curr_node.neighbors
            for neighbor_id in neighbors_ids_list:
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
            node_desired = neighbor_nodes_ids[node_desired_index]
            next_node_id = node_desired
            next_node = topology.get_node(next_node_id)
            link_length = curr_node.geographical_distance(next_node)
            distance_p_n = math.sqrt(math.pow((abstract_point_y - curr_node.y), 2) + math.pow((abstract_point_x - curr_node.x), 2))

            # Change sy only, sx is fixed.
            if distance_p_n >= link_length:
                next_speed_x = curr_speed_x
                next_speed_y = curr_speed_y + acc_gravity * scalar_dt * delta_t
                next_angle_theta_n = angle_link_calculation(curr_node, next_node)
                next_node_found = 1

                # Stopping Criteria Attempt
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
                scalar_dt += 1
        '''
        # Debug section
        print("Current Node:", curr_node)
        print("Incoming Horizontal Velocity:", curr_velocity_x)
        print("Incoming Vertical Velocity:", curr_velocity_y)
        print("Incoming Angle:", curr_angle_theta_n)
        print("Point p Angle:", angle_theta_p)
        print("Links:", link_angles)
        print("Link Comparison:", link_diff)
        print("Next node:", next_node)
        print("Outgoing Horizontal Velocity:", next_velocity_x)
        print("Outgoing Vertical Velocity:", next_velocity_y)
        print("Outgoing angle:", next_angle_theta_n)
        print("Run Time:", run_time)
        print()
        '''

        if dest_node_id == next_node_id:
            output.append(next_node_id)
        else:
            if run_time <= 900:
                if not_reachable == 0:
                    reverse_trajectory(next_node_id, dest_node_id, next_speed_x,
                               next_speed_y, next_angle_theta_n, gravity_vector, 1, run_time, output)
            else:
                output.append(next_node_id)

        return output


    gravity_info = (10, 3 * math.pi / 2)

    start_info_1 = start_info_tuple[0]
    start_info_2 = start_info_tuple[1]

    init_angle_1 = start_info_1[0]
    init_speed_x_1 = start_info_1[1]
    init_speed_y_1 = start_info_1[2]

    init_angle_2 = start_info_2[0]
    init_speed_x_2 = start_info_2[1]
    init_speed_y_2 = start_info_2[2]

    trajectory_1 = trajectory_algorithm(topology, start_node_id, '0', init_speed_x_1, init_speed_y_1, init_angle_1,
                                        gravity_info, 1, 0, [])
    trajectory_2 = trajectory_algorithm(topology, start_node_id, '0', init_speed_x_2, init_speed_y_2, init_angle_2,
                                        gravity_info, 1, 0, [])

    common_node_id = '0'
    for node_id_1 in trajectory_1[1:-1]:
        for node_id_2 in trajectory_2[1:-1]:
            if node_id_1 == node_id_2:
                common_node_id = node_id_1
                break
        if common_node_id != '0':
            break

    multiple_trajectories_plot(topology, {(init_speed_x_1, init_speed_y_1): trajectory_1,
                                          (init_speed_x_2, init_speed_y_2): trajectory_2})
    if common_node_id != '0':
        common_node_info_1 = \
        trajectory_algorithm(topology, start_node_id, common_node_id, init_speed_x_1, init_speed_y_1, init_angle_1,
                             gravity_info, 1, 0, [])[-1]
        common_node_info_2 = \
        trajectory_algorithm(topology, start_node_id, common_node_id, init_speed_x_2, init_speed_y_2, init_angle_2,
                             gravity_info, 1, 0, [])[-1]

        neighbor_node_id_1 = common_node_info_1[0]
        reverse_init_speed_x_1 = common_node_info_1[1]
        reverse_init_speed_y_1 = common_node_info_1[2]
        reverse_init_angle_1 = angle_link_calculation(topology.get_node(common_node_id),
                                                      topology.get_node(neighbor_node_id_1))

        neighbor_node_id_2 = common_node_info_2[0]
        reverse_init_speed_x_2 = common_node_info_2[1]
        reverse_init_speed_y_2 = common_node_info_2[2]
        reverse_init_angle_2 = angle_link_calculation(topology.get_node(common_node_id),
                                                      topology.get_node(neighbor_node_id_2))

        reverse_trajectory_1 = reverse_trajectory(common_node_id, start_node_id, reverse_init_speed_x_1, reverse_init_speed_y_1, reverse_init_angle_1, gravity_info, 1, 0, [])
        reverse_trajectory_2 = reverse_trajectory(common_node_id, start_node_id, reverse_init_speed_x_2, reverse_init_speed_y_2, reverse_init_angle_2, gravity_info, 1, 0, [])

        # reverse_trajectory_1 = trajectory_algorithm(topology, common_node_id, start_node_id, reverse_init_speed_x_1, reverse_init_speed_y_1, reverse_init_angle_1, gravity_info, 1, 0, [])
        # reverse_trajectory_2 = trajectory_algorithm(topology, common_node_id, start_node_id, reverse_init_speed_x_2, reverse_init_speed_y_2, reverse_init_angle_2, gravity_info, 1, 0, [])

        reverse_trajectory_plot(topology, [reverse_trajectory_1, reverse_trajectory_2])


def single_trajectory_plot(topology: TOPOLOGY, output: list):
    # Initialization
    all_node_list = topology.get_all_nodes()
    link_plot_x = []
    link_plot_y = []

    data = {"x": [], "y": [], "id": []}
    for node in all_node_list:
        data["x"].append(node.x)
        data["y"].append(node.y)
        data["id"].append(node.id)

        node_neighbors = topology.get_neighbors(node)[1]
        for neighbor_id in node_neighbors:
            neighbor = topology.get_node(neighbor_id)
            link_plot_x.append(node.x)
            link_plot_y.append(node.y)
            link_plot_x.append(neighbor.x)
            link_plot_y.append(neighbor.y)

    # Show Nodes
    plt.figure()
    # plt.title('Trajectory Test', fontsize=20)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    # Show Connections
    for i in range(0, len(link_plot_x), 2):
        plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2], color='lightgray', linewidth=1, alpha=0.2)
    plt.scatter(data["x"], data["y"], marker='o', color='gray', s=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Show Node ids
    # for label, x, y in zip(data["id"], data["x"], data["y"]):
    # plt.annotate(label, xy=(x, y), fontsize=20)
    ''''
    region_division = 4
    grid_size = 10
    cell_size = grid_size / region_division
    count = 0
    all_cell_bounds = []
    while count < region_division:
        count_1 = 0
        while count_1 < region_division:
            x_lower_bound = count * cell_size
            x_upper_bound = (count + 1) * cell_size
            y_lower_bound = count_1 * cell_size
            y_upper_bound = (count_1 + 1) * cell_size
            all_cell_bounds.append(((x_lower_bound, x_upper_bound), (y_lower_bound, y_upper_bound)))
            count_1 += 1
        count += 1
    # print(all_cell_bounds)
    overall_region_reachability = dict()
    count_region = 1
    for region in all_cell_bounds:
        overall_region_reachability[count_region] = list()
        count_region += 1
    
    # add lines to divide into cells
    draw_cell_bounds = []
    count = 0
    while count <= region_division:
        bound = count * cell_size
        draw_cell_bounds.append(bound)
        count += 1
    for bound in draw_cell_bounds:
        plt.axvline(x=bound, color='black')
        plt.axhline(y=bound, color='black')

    # Write Average Degree (A.D.) in the cell
    cell_num = 1
    for cell_bound in all_cell_bounds:
        ad_text_x = (cell_bound[0][0] + cell_bound[0][1]) / 2 - 1.1
        ad_text_y = (cell_bound[1][0] + cell_bound[1][1]) / 2 - 0.2
        cell_node_count = 0
        cell_degree_count = 0
        for node in all_node_list:
            if cell_bound[0][0] <= node.x <= cell_bound[0][1]:
                if cell_bound[1][0] <= node.y <= cell_bound[1][1]:
                    cell_node_count += 1
                    cell_degree_count += node.degree()
        cell_average_degree = round(cell_degree_count / cell_node_count, 2)
        plt.text(ad_text_x, ad_text_y, 'AD=' + str(cell_average_degree), fontsize=20, color='black')
        # print(cell_bound, cell_num)
        cell_num += 1
    '''

    # Single Trajectory
    if len(output) != 0:
        trajectory_plot_x = []
        trajectory_plot_y = []
        for node_id in output:
            node = topology.get_node(node_id)
            trajectory_plot_x.append(node.x)
            trajectory_plot_y.append(node.y)
        init_node = topology.get_node(output[0])
        end_node = topology.get_node(output[-1])
        plt.scatter(init_node.x, init_node.y, marker='o', linewidth=15, color='red')
        plt.scatter(end_node.x, end_node.y, marker='o', linewidth=15, color='blue')
        for i in range(0, len(trajectory_plot_x), 1):
            plt.plot(trajectory_plot_x[i:i + 2], trajectory_plot_y[i:i + 2], color='black', linewidth=3)

    plt.show()


def trajectory_search(topology: TOPOLOGY, init_node_id: str, dest_node_id: str, init_angle: float, speed_x: float,
                      gravity_vector: tuple):
    # Setup and Initialization
    trajectory_dict = {}

    def trajectory_comparison(trajectory_1: list, trajectory_2: list):
        trajectory_1_fixed = []
        for node_id in trajectory_1:
            node_id_index = trajectory_1.index(node_id)
            if node_id_index > 2:
                if node_id not in trajectory_1_fixed:
                    trajectory_1_fixed.append(node_id)
            else:
                trajectory_1_fixed.append(node_id)

        trajectory_2_fixed = []
        for node_id in trajectory_2:
            node_id_index = trajectory_2.index(node_id)
            if node_id_index > 2:
                if node_id not in trajectory_2_fixed:
                    trajectory_2_fixed.append(node_id)
            else:
                trajectory_2_fixed.append(node_id)

        return trajectory_1_fixed == trajectory_2_fixed

    # Speed and Trajectory Exponential Search
    def trajectory_exponential_bounds():
        exponent_k = 0
        intermediate_speed = 0
        intermediate_trajectory = []
        lower_speed = math.pow(2, exponent_k)
        lower_speed_xy = speed_xy_calculation(lower_speed, init_angle)
        lower_speed_x = lower_speed_xy[0]
        lower_speed_y = lower_speed_xy[1]
        lower_trajectory = trajectory_algorithm(topology, init_node_id, dest_node_id, speed_x, lower_speed_y,
                                                init_angle, gravity_vector, 1, 0, [])
        previous_speed = lower_speed
        previous_trajectory = lower_trajectory

        while True:
            exponent_k += 1
            # print(exponent_k)
            intermediate_speed = math.pow(2, exponent_k)
            intermediate_speed_xy = speed_xy_calculation(intermediate_speed, init_angle)
            intermediate_speed_x = intermediate_speed_xy[0]
            intermediate_speed_y = intermediate_speed_xy[1]
            intermediate_trajectory = trajectory_algorithm(topology, init_node_id, dest_node_id, speed_x,
                                                           intermediate_speed_y, init_angle, gravity_vector, 1, 0, [])
            # print(lower_trajectory, intermediate_trajectory)
            if trajectory_comparison(previous_trajectory, intermediate_trajectory):
            # if previous_trajectory == intermediate_trajectory:
                if exponent_k > 3:
                    higher_speed = intermediate_speed
                    higher_trajectory = intermediate_trajectory
                    break
                else:
                    previous_speed = intermediate_speed
                    previous_trajectory = intermediate_trajectory
                    continue
            else:
                previous_speed = intermediate_speed
                previous_trajectory = intermediate_trajectory
                continue

        return lower_speed, lower_trajectory, higher_speed, higher_trajectory

    lower_speed, lower_trajectory, higher_speed, higher_trajectory = trajectory_exponential_bounds()

    # Binary Search
    def trajectory_binary_search(speed_lower: float, speed_higher: float, trajectory_lower: list,
                                 trajectory_higher: list):
        mid_speed = (speed_lower + speed_higher) / 2
        mid_speed_xy = speed_xy_calculation(mid_speed, init_angle)
        mid_speed_x = mid_speed_xy[0]
        mid_speed_y = mid_speed_xy[1]
        mid_trajectory = trajectory_algorithm(topology, init_node_id, dest_node_id, speed_x, mid_speed_y, init_angle,
                                              gravity_vector, 1, 0, [])
        # print()
        # print(mid_trajectory, trajectory_lower, trajectory_higher)
        if (mid_speed != speed_lower) and (mid_speed != speed_higher):
            if (trajectory_comparison(mid_trajectory, trajectory_lower)) and (trajectory_comparison(trajectory_lower, trajectory_higher) == False):
            # if (mid_trajectory == trajectory_lower) and (trajectory_lower != trajectory_higher):
                trajectory_dict[mid_speed] = mid_trajectory
                # print('lower', init_node, init_angle, speed_lower, trajectory_lower)
                trajectory_binary_search(mid_speed, speed_higher, mid_trajectory, trajectory_higher)
            elif (trajectory_comparison(mid_trajectory, trajectory_higher)) and (trajectory_comparison(trajectory_higher, trajectory_lower) == False):
            # elif (mid_trajectory == trajectory_higher) and (trajectory_higher != trajectory_lower):
                trajectory_dict[mid_speed] = mid_trajectory
                # print('higher', init_node, init_angle, speed_higher, trajectory_higher)
                trajectory_binary_search(speed_lower, mid_speed, trajectory_lower, mid_trajectory)
            elif (trajectory_comparison(mid_trajectory, trajectory_higher) == False) and (trajectory_comparison(mid_trajectory, trajectory_lower) == False):
            # elif (mid_trajectory != trajectory_higher) and (mid_trajectory != trajectory_lower):
                trajectory_binary_search(mid_speed, speed_higher, mid_trajectory, trajectory_higher)
                trajectory_binary_search(speed_lower, mid_speed, trajectory_lower, mid_trajectory)
            elif (trajectory_comparison(mid_trajectory, trajectory_higher) and (trajectory_comparison(mid_trajectory, trajectory_lower))):
                trajectory_dict[speed_lower] = trajectory_lower

    trajectory_binary_search(lower_speed, higher_speed, lower_trajectory, higher_trajectory)
    return trajectory_dict


def multiple_trajectories_plot(topology: TOPOLOGY, output: dict):
    # Initialization
    all_node_list = topology.get_all_nodes()
    link_plot_x = []
    link_plot_y = []

    data = {"x": [], "y": [], "id": []}
    for node in all_node_list:
        data["x"].append(node.x)
        data["y"].append(node.y)
        data["id"].append(node.id)

        node_neighbors = topology.get_neighbors(node)[1]
        for neighbor_id in node_neighbors:
            neighbor = topology.get_node(neighbor_id)
            link_plot_x.append(node.x)
            link_plot_y.append(node.y)
            link_plot_x.append(neighbor.x)
            link_plot_y.append(neighbor.y)

    # Show Nodes
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title('Gravity Direction of 0-Degree (Rightward)', fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    # Show Connections
    for i in range(0, len(link_plot_x), 2):
        plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2], color='lightgray', linewidth=1, alpha=0.2)
    plt.scatter(data["x"], data["y"], marker='o', color='gray', s=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Multiple Trajectory
    # colors = ['blue', 'darkorange', 'magenta', 'sienna', 'olive'] * 100
    # colors = ['blue', 'black', 'red']
    colors = ['black', 'red', 'darkorange']
    for index, speed in enumerate(output, 0):
        color_index = index
        trajectory = output[speed]
        trajectory_plot_x = []
        trajectory_plot_y = []
        for node_id in trajectory:
            node = topology.get_node(node_id)
            trajectory_plot_x.append(node.x)
            trajectory_plot_y.append(node.y)
        init_node = topology.get_node(trajectory[0])
        end_node = topology.get_node(trajectory[-1])
        plt.scatter(init_node.x, init_node.y, marker='o', color='green')
        plt.scatter(end_node.x, end_node.y, marker='o', color=colors[color_index])
        for i in range(0, len(trajectory_plot_x), 1):
            plt.plot(trajectory_plot_x[i:i + 2], trajectory_plot_y[i:i + 2], color=colors[color_index])

    plt.show()


def reverse_trajectory_plot(topology: TOPOLOGY, output: list):
    # Initialization
    all_node_list = topology.get_all_nodes()
    link_plot_x = []
    link_plot_y = []

    data = {"x": [], "y": [], "id": []}
    for node in all_node_list:
        data["x"].append(node.x)
        data["y"].append(node.y)
        data["id"].append(node.id)

        node_neighbors = topology.get_neighbors(node)[1]
        for neighbor_id in node_neighbors:
            neighbor = topology.get_node(neighbor_id)
            link_plot_x.append(node.x)
            link_plot_y.append(node.y)
            link_plot_x.append(neighbor.x)
            link_plot_y.append(neighbor.y)

    # Show Nodes
    plt.figure()
    plt.title('Reverse Trajectory Test', fontsize=20)
    plt.xlabel('x-axis', fontsize=20)
    plt.ylabel('y-axis', fontsize=20)
    plt.scatter(data["x"], data["y"], marker='o', color='cornflowerblue', linewidth=5)
    # Show Node ids
    # for label, x, y in zip(data["id"], data["x"], data["y"]):
    # plt.annotate(label, xy=(x, y), fontsize=20)

    # Show Connections
    for i in range(0, len(link_plot_x), 2):
        plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2], color='lightsteelblue')

    # Multiple Trajectory
    colors = ['blue', 'darkorange']
    both_reverse_trajectories = []
    for trajectory in output:
        trajectory_nodes = []
        color_index = output.index(trajectory)
        trajectory_plot_x = []
        trajectory_plot_y = []
        try:
            trajectory[-1].append('1')
            trajectory = trajectory[:-1]
        except:
            trajectory = trajectory

        for node_id in trajectory:
            node = topology.get_node(node_id)
            trajectory_nodes.append(node_id)
            trajectory_plot_x.append(node.x)
            trajectory_plot_y.append(node.y)
        both_reverse_trajectories.append(trajectory_nodes)
        init_node = topology.get_node(trajectory[0])
        end_node = topology.get_node(trajectory[-1])
        plt.scatter(init_node.x, init_node.y, marker='o', linewidth=15, color='green')
        plt.scatter(end_node.x, end_node.y, marker='o', linewidth=15, color=colors[color_index])
        for i in range(0, len(trajectory_plot_x), 1):
            plt.plot(trajectory_plot_x[i:i + 2], trajectory_plot_y[i:i + 2], color=colors[color_index], linewidth=3)

    # Show All the Common Nodes
    reverse_trajectory_plot_x = []
    reverse_trajectory_plot_y = []
    for node_id_1 in both_reverse_trajectories[0][1:]:
        for node_id_2 in both_reverse_trajectories[1][1:]:
            if node_id_1 == node_id_2:
                common_node = topology.get_node(node_id_1)
                reverse_trajectory_plot_x.append(common_node.x)
                reverse_trajectory_plot_y.append(common_node.y)
    # plt.scatter(reverse_trajectory_plot_x, reverse_trajectory_plot_y, marker='o', linewidth=15, color='yellow')

    plt.show()


def speed_xy_calculation(speed: float, angle: float):
    speed_x = speed * math.cos(angle)
    speed_y = speed * math.sin(angle)

    if 0 <= angle < (math.pi / 2):
        speed_x = abs(speed * math.cos(angle))
        speed_y = abs(speed * math.sin(angle))
    elif (math.pi / 2) <= angle < math.pi:
        speed_x = -1 * abs(speed * math.cos(angle))
        speed_y = abs(speed * math.sin(angle))
    elif math.pi <= angle < (3 * math.pi / 2):
        speed_x = -1 * abs(speed * math.cos(angle))
        speed_y = -1 * abs(speed * math.sin(angle))
    elif (3 * math.pi / 2) <= angle < (2 * math.pi):
        speed_x = abs(speed * math.cos(angle))
        speed_y = -1 * abs(speed * math.sin(angle))

    return speed_x, speed_y


def dijkstra_sp(topology: TOPOLOGY):
    all_nodes = topology.get_all_nodes()
    all_dijkstra_sp = dict()
    for start_node in all_nodes:
        for dest_node in all_nodes:
            if start_node.id != dest_node.id:
                all_dijkstra_sp[(start_node.id, dest_node.id)] = list()

                distances = {node.id: math.inf for node in all_nodes}
                previous_nodes = {node.id: None for node in all_nodes}
                distances[start_node.id] = 0

                priority_queue = [(0, start_node.id)]  # Priority queue of (distance, node_id)

                while priority_queue:
                    current_distance, current_node_id = heapq.heappop(priority_queue)
                    current_node = topology.get_node(current_node_id)

                    if current_node_id == dest_node.id:
                        break

                    for neighbor_id in current_node.neighbors:
                        neighbor = topology.get_node(neighbor_id)
                        distance = 1

                        new_distance = current_distance + distance
                        if new_distance < distances[neighbor_id]:
                            distances[neighbor_id] = new_distance
                            previous_nodes[neighbor_id] = current_node_id
                            heapq.heappush(priority_queue, (new_distance, neighbor_id))

                path = []
                current_id = dest_node.id
                while current_id is not None:
                    path.insert(0, current_id)
                    current_id = previous_nodes[current_id]

                all_dijkstra_sp[(start_node.id, dest_node.id)] = path
                # print(path, distances[dest_node.id])
    # print(all_dijkstra_sp)
    return all_dijkstra_sp



def heat_map(topology: TOPOLOGY, grid_size: int, region_division: int, non_reachable_output: dict):
    '''
    def heat_map_plotting(reachability_region: dict, start_region: int):
        data = {"x": [], "y": [], "id": []}
        for node in all_node_list:
            data["x"].append(node.x)
            data["y"].append(node.y)
            data["id"].append(node.id)

        # add lines to divide into cells
        draw_cell_bounds = []
        count = 0
        while count <= region_division:
            bound = count * cell_size
            draw_cell_bounds.append(bound)
            count += 1
        for bound in draw_cell_bounds:
            plt.axvline(x=bound, color='black')
            plt.axhline(y=bound, color='black')

        plt.title('Degree 5 Topology #1, Union of 8 g-direction Regional Reachability in Region-' + str(start_region))
        # Write numbers in the cell
        cell_num = 1
        region_bound_dict = dict()
        for bound in all_cell_bounds:
            text_x = (bound[0][0] + bound[0][1]) / 2
            text_y = (bound[1][0] + bound[1][1]) / 2
            region_x_bound = (bound[0][0], bound[0][1])
            region_y_bound = (bound[1][0], bound[1][1])
            region_bound_dict[cell_num] = (region_x_bound, region_y_bound)
            if start_region == cell_num:
                plt.text(text_x, text_y, cell_num, fontsize=20, color='red')
            else:
                plt.text(text_x, text_y, cell_num, fontsize=20, color='black')
            cell_num += 1

        # Write Average Degree (A.D.) in the cell
        cell_num = 1
        for cell_bound in all_cell_bounds:
            ad_text_x = (cell_bound[0][0] + cell_bound[0][1]) / 2 - 1
            ad_text_y = (cell_bound[1][0] + cell_bound[1][1]) / 2 - 1
            cell_node_count = 0
            cell_degree_count = 0
            for node in all_node_list:
                if cell_bound[0][0] <= node.x <= cell_bound[0][1]:
                    if cell_bound[1][0] <= node.y <= cell_bound[1][1]:
                        cell_node_count += 1
                        cell_degree_count += node.degree()
            cell_average_degree = round(cell_degree_count / cell_node_count, 2)
            if start_region == cell_num:
                plt.text(ad_text_x, ad_text_y, 'AD=' + str(cell_average_degree), fontsize=12, color='red')
            else:
                plt.text(ad_text_x, ad_text_y, 'AD=' + str(cell_average_degree), fontsize=12, color='black')
            cell_num += 1

        # Region Colorcode
        color_list = ['green', 'forestgreen', 'limegreen', 'springgreen', 'mediumspringgreen', 'aquamarine',
                      'lightseagreen', 'aqua', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'slateblue',
                      'blueviolet', 'darkviolet', 'magenta', 'deeppink', 'hotpink', 'pink']
        color_code_dict = dict()
        maximum_percentage = 100
        index_count = 0
        percentage_bounds = []
        for color in color_list:
            percentage_bounds.append(maximum_percentage)
            lower_maximum_percentage = maximum_percentage - 2
            if index_count == (len(color_list) - 1):
                color_code_dict[(maximum_percentage, 0)] = color_list[index_count]
            else:
                color_code_dict[(maximum_percentage, lower_maximum_percentage)] = color_list[index_count]
            index_count += 1
            maximum_percentage -= 2

        for dest_region in range(1, int(math.pow(region_division, 2) + 1)):
            reachability_percentage = reachability_region[(start_region, dest_region)][2]
            print((start_region, dest_region), reachability_percentage)
            for percentage in color_code_dict:
                higher_percent = percentage[0]
                lower_percent = percentage[1]
                if lower_percent < reachability_percentage <= higher_percent:
                    region_color = color_code_dict[percentage]
                    dest_region_bound = region_bound_dict[dest_region]
                    dest_region_x_bound = dest_region_bound[0]
                    dest_region_y_bound = dest_region_bound[1]
                    plt.fill_between(dest_region_x_bound, dest_region_y_bound[0], dest_region_y_bound[1],
                                     facecolor=region_color)

        percentage_bounds.sort()
        reversed_color_list = color_list
        reversed_color_list.reverse()
        cmap = mpl.colors.ListedColormap(reversed_color_list)
        cmap.set_under('pink')
        norm = mpl.colors.BoundaryNorm(percentage_bounds, cmap.N)
        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            boundaries=percentage_bounds,
            extend='min',
            extendfrac='auto',
            ticks=percentage_bounds,
            spacing='uniform',
            orientation='vertical',
            label='Reachability(%)',
        )
        plt.show()
    '''
    def heat_map_plotting_overall(reachability_region: dict):
        plt.rcParams["font.family"] = "Times New Roman"
        data = {"x": [], "y": [], "id": []}
        for node in all_node_list:
            data["x"].append(node.x)
            data["y"].append(node.y)
            data["id"].append(node.id)

        # add lines to divide into cells
        draw_cell_bounds = []
        count = 0
        while count <= region_division:
            bound = count * cell_size
            draw_cell_bounds.append(bound)
            count += 1
        for bound in draw_cell_bounds:
            plt.axvline(x=bound, color='black')
            plt.axhline(y=bound, color='black')
        
        # plt.title('Regional Reachability, Union of 8 Gravity Directions', fontsize=15)
        cell_num = 1
        region_bound_dict = dict()
        for bound in all_cell_bounds:
            text_x = (bound[0][0] + bound[0][1]) / 2
            text_y = (bound[1][0] + bound[1][1]) / 2
            region_x_bound = (bound[0][0], bound[0][1])
            region_y_bound = (bound[1][0], bound[1][1])
            region_bound_dict[cell_num] = (bound[0], bound[1])
            # region_bound_dict[cell_num] = (region_x_bound, region_y_bound)
            # plt.text(text_x, text_y, cell_num, fontsize=15, color='black')
            # print(bound)
            cell_num += 1

        # Write Average Degree (A.D.) in the cell
        cell_num = 1
        for cell_bound in all_cell_bounds:
            ad_text_x = (cell_bound[0][0] + cell_bound[0][1]) / 2 - 1
            ad_text_y = (cell_bound[1][0] + cell_bound[1][1]) / 2 - 1
            cell_node_count = 0
            cell_degree_count = 0
            for node in all_node_list:
                if cell_bound[0][0] <= node.x <= cell_bound[0][1]:
                    if cell_bound[1][0] <= node.y <= cell_bound[1][1]:
                        cell_node_count += 1
                        cell_degree_count += node.degree()
            cell_average_degree = round(cell_degree_count / cell_node_count, 2)
            # plt.text(ad_text_x, ad_text_y, 'AD=' + str(cell_average_degree), fontsize=12, color='black')
            # print(cell_bound, cell_num)
            cell_num += 1

        # Region Colorcode
        # 17 COLORS
        color_list = ['green', 'forestgreen', 'limegreen', 'springgreen', 'lime',
                      'paleturquoise', 'aqua', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'slateblue',
                      'blueviolet', 'darkviolet', 'violet', 'pink', 'lightpink']
        color_code_dict = dict()
        maximum_percentage = 100
        index_count = 0
        percentage_bounds = []
        for color in color_list:
            percentage_bounds.append(maximum_percentage)
            lower_maximum_percentage = maximum_percentage - 2
            if index_count == (len(color_list) - 1):
                color_code_dict[(maximum_percentage, 0)] = color_list[index_count]
            else:
                color_code_dict[(maximum_percentage, lower_maximum_percentage)] = color_list[index_count]
            index_count += 1
            maximum_percentage -= 2

        for region in range(1, int(math.pow(region_division, 2) + 1)):
            reachability_list = reachability_region[region]
            reachability_percentage = 100 * sum(reachability_list) / (len(reachability_list) * (len(all_node_list)-1))
            for percentage in color_code_dict:
                higher_percent = percentage[0]
                lower_percent = percentage[1]
                if lower_percent < reachability_percentage <= higher_percent:
                    region_color = color_code_dict[percentage]
                    region_bound = region_bound_dict[region]
                    region_x_bound = region_bound[0]
                    region_y_bound = region_bound[1]
                    # plt.fill_between(region_x_bound, region_y_bound[0], region_y_bound[1],
                                     # facecolor=region_color)


        for start_region in range(1, int(math.pow(region_division, 2) + 1)):
            overall_regional_count = 0
            overall_regional_reachable_count = 0
            for dest_region in range(1, int(math.pow(region_division, 2) + 1)):
                overall_regional_count += reachability_region[(start_region, dest_region)][0]
                overall_regional_reachable_count += reachability_region[(start_region, dest_region)][1]
            reachability_percentage = (overall_regional_reachable_count / overall_regional_count) * 100
            for percentage in color_code_dict:
                higher_percent = percentage[0]
                lower_percent = percentage[1]
                if lower_percent < reachability_percentage <= higher_percent:
                    region_color = color_code_dict[percentage]
                    start_region_bound = region_bound_dict[start_region]
                    start_region_x_bound = start_region_bound[0]
                    start_region_y_bound = start_region_bound[1]
                    plt.fill_between(start_region_x_bound, start_region_y_bound[0], start_region_y_bound[1],
                                     facecolor=region_color)

        percentage_bounds.sort()
        reversed_color_list = color_list
        reversed_color_list.reverse()
        cmap = mpl.colors.ListedColormap(reversed_color_list)
        cmap.set_under('lightpink')
        norm = mpl.colors.BoundaryNorm(percentage_bounds, cmap.N)

        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            boundaries=percentage_bounds,
            extend='min',
            extendfrac='auto',
            ticks=percentage_bounds,
            spacing='uniform',
            orientation='vertical',
            label='Reachability(%)',
        )

        plt.show()

    plt.rcParams["font.family"] = "Times New Roman"
    all_node_list = topology.get_all_nodes()

    reachability_output = dict()
    for start_node in all_node_list:
        reachability_output[start_node.id] = []
        for dest_node in all_node_list:
            if start_node != dest_node:
                if start_node.id in non_reachable_output:
                    if dest_node.id not in non_reachable_output[start_node.id]:
                        reachability_output[start_node.id].append(dest_node.id)
                # else:
                    # reachability_output[start_node.id].append(dest_node.id)

    cell_size = grid_size / region_division
    count = 0
    all_cell_bounds = []
    while count < region_division:
        count_1 = 0
        while count_1 < region_division:
            x_lower_bound = count * cell_size
            x_upper_bound = (count + 1) * cell_size
            y_lower_bound = count_1 * cell_size
            y_upper_bound = (count_1 + 1) * cell_size
            all_cell_bounds.append(((x_lower_bound, x_upper_bound), (y_lower_bound, y_upper_bound)))
            count_1 += 1
        count += 1
    # print(all_cell_bounds)
    overall_region_reachability = dict()
    count_region = 1
    for region in all_cell_bounds:
        overall_region_reachability[count_region] = list()
        count_region += 1
    for dest_node_id in reachability_output:
        dest_node = topology.get_node(dest_node_id)
        count_region = 1
        reachable_count = 0
        for region in all_cell_bounds:
            x_lower_bound = region[0][0]
            x_upper_bound = region[0][1]
            y_lower_bound = region[1][0]
            y_upper_bound = region[1][1]
            # print(region)
            if (x_lower_bound <= dest_node.x < x_upper_bound) and (y_lower_bound <= dest_node.y < y_upper_bound):
                for start_node in all_node_list:
                    if dest_node_id != start_node.id:
                        if dest_node_id in reachability_output[start_node.id]:
                            reachable_count += 1
                overall_region_reachability[count_region].append(reachable_count)
            count_region += 1

    '''
    count_1 = 1
    for region_1 in all_cell_bounds:
        x1_lower_bound = region_1[0][0]
        x1_upper_bound = region_1[0][1]
        y1_lower_bound = region_1[1][0]
        y1_upper_bound = region_1[1][1]
        count_2 = 1
        for region_2 in all_cell_bounds:
            overall_region_reachability[(count_1, count_2)] = []
            total_count = 0
            reachable_count = 0
            x2_lower_bound = region_2[0][0]
            x2_upper_bound = region_2[0][1]
            y2_lower_bound = region_2[1][0]
            y2_upper_bound = region_2[1][1]
        
            for start_node_id in reachability_output:
                start_node = topology.get_node(start_node_id)
                if (x1_lower_bound <= start_node.x < x1_upper_bound) and (y1_lower_bound <= start_node.y < y1_upper_bound):
                    for dest_node in all_node_list:
                        if dest_node != start_node:
                            if (x2_lower_bound <= dest_node.x < x2_upper_bound) and (y2_lower_bound <= dest_node.y < y2_upper_bound):
                                total_count += 1
                                reachable_node_list = reachability_output[start_node.id]
                                if dest_node.id in reachable_node_list:
                                    reachable_count += 1
            overall_region_reachability[(count_1, count_2)].append(total_count)
            overall_region_reachability[(count_1, count_2)].append(reachable_count)
            reachable_percentage = round((reachable_count / total_count) * 100, 3)
            overall_region_reachability[(count_1, count_2)].append(reachable_percentage)
            
            count_2 += 1
        count_1 += 1
    '''
    # for start_region in range(1, int(math.pow(region_division, 2) + 1)):
        # heat_map_plotting(overall_region_reachability, start_region)

    heat_map_plotting_overall(overall_region_reachability)

    return overall_region_reachability


def nonreachable_node_plot(topology: TOPOLOGY, init_node_id: str, non_reachable_list: list):
    # Initialization
    all_node_list = topology.get_all_nodes()
    link_plot_x = []
    link_plot_y = []

    data = {"x": [], "y": [], "id": []}
    for node in all_node_list:
        data["x"].append(node.x)
        data["y"].append(node.y)
        data["id"].append(node.id)

        node_neighbors = topology.get_neighbors(node)[1]
        for neighbor_id in node_neighbors:
            neighbor = topology.get_node(neighbor_id)
            link_plot_x.append(node.x)
            link_plot_y.append(node.y)
            link_plot_x.append(neighbor.x)
            link_plot_y.append(neighbor.y)

    # Show Nodes
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title('Non-Reachable Nodes with Eight Gravity Directions', fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    # Show Connections
    for i in range(0, len(link_plot_x), 2):
        plt.plot(link_plot_x[i:i + 2], link_plot_y[i:i + 2], color='lightgray', linewidth=1, alpha=0.2)
    plt.scatter(data["x"], data["y"], marker='o', color='gray', s=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Single Trajectory
    # if len(non_reachable_list) != 0:
    trajectory_plot_x = []
    trajectory_plot_y = []
    for node_id in non_reachable_list:
        node = topology.get_node(node_id)
        trajectory_plot_x.append(node.x)
        trajectory_plot_y.append(node.y)
        plt.scatter(node.x, node.y, marker='o', color='black')
    init_node = topology.get_node(init_node_id)
    plt.scatter(init_node.x, init_node.y, marker='o', color='red')

    plt.show()

'''
# Trajectory Search (One Node, One Angle)
    # Initial Node Plot
plt.scatter(node_coordinates[output[1][0]][0], node_coordinates[output[1][0]][1], marker='o', linewidth=15,
            color='red')
    # Setup
trajectory_max_plot_x = []
trajectory_max_plot_y = []
trajectory_min_plot_x = []
trajectory_min_plot_y = []
trajectory_max = output[1]
trajectory_min = output[-1]

    # Other Trajectories
all_trajectories = []
index = 0
while index < len(output):
    if index % 2 == 1:
        all_trajectories.append(output[index])
    index += 1
# color_list = ['sienna', 'darkorange', 'gold', 'lawngreen', 'darkgreen', 'aqua', 'blue', 'darkviolet', 'magenta', 'crimson',
              # 'pink', 'rosybrown', 'tomato', 'peru', 'tan', 'olive', 'lightgreen', 'lime', 'turquoise', 'skyblue',
              # 'royalblue', 'slateblue', 'indigo', 'thistle', 'orchid']

color_list = [name for name in mcolors.CSS4_COLORS
               if f'xkcd:{name}' in mcolors.XKCD_COLORS]

color_list.remove('black')
color_list.remove('red')

for trajectory in all_trajectories[1:-1]:

    color_index = all_trajectories.index(trajectory)
    # trajectory_color = color_list[color_index - 1]
    trajectory_plot_x = []
    trajectory_plot_y = []
    for node in trajectory:
        node_x = node_coordinates[node][0]
        node_y = node_coordinates[node][1]
        trajectory_plot_x.append(node_x)
        trajectory_plot_y.append(node_y)
    plt.scatter(node_coordinates[trajectory[0]][0], node_coordinates[trajectory[0]][1], marker='o', linewidth=15,
                color='red')
    # plt.scatter(node_coordinates[trajectory[-1]][0], node_coordinates[trajectory[-1]][1], marker='o', linewidth=15,
                # color=trajectory_color)
    for i in range(0, len(trajectory_plot_x), 1):
        plt.plot(trajectory_plot_x[i:i + 2], trajectory_plot_y[i:i + 2], color='aqua', linewidth=3)

    # Trajectory Upper Bound Plot
for max_node in trajectory_max:
    max_node_x = node_coordinates[max_node][0]
    max_node_y = node_coordinates[max_node][1]
    trajectory_max_plot_x.append(max_node_x)
    trajectory_max_plot_y.append(max_node_y)
plt.scatter(node_coordinates[output[1][-1]][0], node_coordinates[output[1][-1]][1], marker='o', linewidth=15,
            color='darkred')
for i in range(0, len(trajectory_max_plot_x), 1):
    plt.plot(trajectory_max_plot_x[i:i + 2], trajectory_max_plot_y[i:i + 2], color='red', linewidth=3)

    # Trajectory Lower Bound Plot
for min_node in trajectory_min:
    min_node_x = node_coordinates[min_node][0]
    min_node_y = node_coordinates[min_node][1]
    trajectory_min_plot_x.append(min_node_x)
    trajectory_min_plot_y.append(min_node_y)
plt.scatter(node_coordinates[output[-1][-1]][0], node_coordinates[output[-1][-1]][1], marker='o', linewidth=15,
            color='black')
for i in range(0, len(trajectory_min_plot_x), 1):
    plt.plot(trajectory_min_plot_x[i:i + 2], trajectory_min_plot_y[i:i + 2], color='black', linewidth=3)
'''
'''
    # Reachable Nodes in between Bounds
all_reachable_nodes = []
index = 0
while index < (len(output) - 1):
    for trajectory_node in output[index + 1]:
        if trajectory_node not in all_reachable_nodes:
            if trajectory_node not in trajectory_max:
                if trajectory_node not in trajectory_min:
                    all_reachable_nodes.append(trajectory_node)
    index += 2
bound_reachability_plot_x = []
bound_reachability_plot_y = []
for bound_node in all_reachable_nodes:
    bound_node_x = node_coordinates[bound_node][0]
    bound_node_y = node_coordinates[bound_node][1]
    bound_reachability_plot_x.append(bound_node_x)
    bound_reachability_plot_y.append(bound_node_y)
plt.scatter(bound_reachability_plot_x, bound_reachability_plot_y, marker='o', linewidth=15, color='green')
'''
'''
# Reachability
reachability_plot_x = []
reachability_plot_y = []
for node in output:
    node_x = node_coordinates[node][0]
    node_y = node_coordinates[node][1]
    reachability_plot_x.append(node_x)
    reachability_plot_y.append(node_y)
plt.scatter(node_coordinates[output[0]][0], node_coordinates[output[0]][1], marker='o', linewidth=15, color='red')
plt.scatter(node_coordinates[output[-1]][0], node_coordinates[output[-1]][1], marker='o', linewidth=15, color='blue')
for i in range(0, len(reachability_plot_x), 1):
    plt.plot(reachability_plot_x[i:i + 2], reachability_plot_y[i:i + 2], color='black', linewidth=3)
'''
'''
# Reachability Extensive
reachability_plot_x = []
reachability_plot_y = []
for node in output[1:]:
    node_x = node_coordinates[node][0]
    node_y = node_coordinates[node][1]
    reachability_plot_x.append(node_x)
    reachability_plot_y.append(node_y)
plt.scatter(node_coordinates[output[0]][0], node_coordinates[output[0]][1], marker='o', linewidth=15,
            color='red')
plt.scatter(reachability_plot_x, reachability_plot_y, marker='o', linewidth=15, color='green')
non_reachable_plot_x = []
non_reachable_plot_y = []
for node in node_coordinates:
    if node not in output:
        node_x = node_coordinates[node][0]
        node_y = node_coordinates[node][1]
        non_reachable_plot_x.append(node_x)
        non_reachable_plot_y.append(node_y)
plt.scatter(non_reachable_plot_x, non_reachable_plot_y, marker='o', linewidth=15, color='darkorange')

# add grid line
# plt.grid(which='minor', color='gray', linestyle='--', linewidth=1)

# Heat map grid line
cell_size = grid_size / 4
draw_cell_bounds = []
count = 0
while count <= 4:
    bound = count * cell_size
    draw_cell_bounds.append(bound)
    count += 1
for bound in draw_cell_bounds:
    plt.axvline(x=bound, color='black')
    plt.axhline(y=bound, color='black')

plt.show()
'''
"""
def quadrant_check(angle_1: float, angle_2: float):
    if (0 <= angle_1 < (math.pi / 2)) and (0 <= angle_2 < (math.pi / 2)):
        return True
    elif ((math.pi / 2) <= angle_1 < math.pi) and ((math.pi / 2) <= angle_2 < math.pi):
        return True
    elif (math.pi <= angle_1 < (3 * math.pi / 2)) and (math.pi <= angle_2 < (3 * math.pi / 2)):
        return True
    elif ((3 * math.pi / 2) <= angle_1 < (2 * math.pi)) and ((3 * math.pi / 2) <= angle_2 < (2 * math.pi)):
        return True
    else:
        return False


def reachability(topology_connection: dict, node_coordinates: dict, velocity_x: float, start_node: str, dest_node: str, delta_t: float):
    route_output = []
    speed_angle_gravity = []
    compare_num = 1000000
    acc_gravity = 10  # m/s^2
    gravity_direction = [3 * math.pi / 2]

    # Initial Angles
    angle_link_list = []
    for node in topology_connection[start_node]:
        angle = topology_connection[start_node][node]
        angle_link_list.append(angle)

    # Reachability Test
    for g_direction in gravity_direction:
        for init_angle in angle_link_list:
            speed = 0
            while speed < 1000:
                if speed == 0:
                    routes = trajectory(topology_connection, node_coordinates, start_node, dest_node, velocity_x, 0, init_angle,
                                        (acc_gravity, g_direction), delta_t, 1, 0, [])
                else:
                    # Original
                    # x_speed = speed * math.cos(init_angle)
                    # y_speed = speed * math.sin(init_angle)

                    # Quadrant Specified
                    speed_xy = speed_xy_calculation(speed, init_angle)
                    x_speed = speed_xy[0]
                    y_speed = speed_xy[1]
                    routes = trajectory(topology_connection, node_coordinates, start_node, dest_node, velocity_x, y_speed,
                                        init_angle, (acc_gravity, g_direction), delta_t, 1, 0, [])
                if (dest_node in routes) and (len(routes) < compare_num):
                    index = routes.index(dest_node)
                    route_output.append(routes[:(index + 1)])
                    speed_angle_gravity.append(((round(speed, 2), round(math.degrees(init_angle), 2)),
                                                (acc_gravity, round(math.degrees(g_direction)))))
                    compare_num = len(routes)
                    break
                speed += 1
            if len(route_output) != 0:
                break
        if len(route_output) != 0:
            break

    output = dict(zip(speed_angle_gravity, route_output))
    return output


def random_walk(topology: Topology, start_node: str, dest_node: str):
    next_node = start_node
    output = []
    while next_node != dest_node:
        output.append(next_node)
        neighbors = topology.get_connection(next_node)[1]
        num_neighbors = len(neighbors)
        ranges = []
        for i in range(1, num_neighbors + 1):
            ranges.append(((i - 1) / num_neighbors, i / num_neighbors))
        random_num = random.random()
        index = 0
        while True:
            if ranges[index][1] > random_num >= ranges[index][0]:
                break
            else:
                index += 1
        next_node = neighbors[index]
        if next_node == dest_node:
            output.append(next_node)
    return output
"""


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

    topology, all_nodes = planar_topology_creation(init_topo, num_node, grid_size, grid_division_num, radius, min_deg, avg_deg)

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
    '''
    # Get one Node all neighbor's angle
    node_id = '1'
    node_angles = list()
    node = topology.get_node(node_id)
    for neighbor_id in node.neighbors:
        neighbor = topology.get_node(neighbor_id)
        if neighbor != node:
            angle = angle_link_calculation(node, neighbor)
            node_angles.append(angle)
            print(str(node.id) + " - " + str(neighbor.id) + ": " + str(angle) + " rad = " + str(math.degrees(angle)) + " deg")
    '''
    '''
    # Trajectory Test
    start_node_id = '1'

    init_x_speed = 0
    init_y_speed = 0
    init_angle = 1.1631419984385158
    gravity_info = (acc_gravity, 3 * math.pi / 2)
    output = trajectory_algorithm(topology, start_node_id, '0', init_x_speed, init_y_speed, init_angle, gravity_info,
                                  1, 0, [])
    #                   topology, start node, dest node, init-speed-x, init-speed-y, init-angle, gravity,
    #                   scalar-delta-t, runtime, output
    print(output)
    print()
    single_trajectory_plot(topology, output)
    '''
    # single_trajectory_plot(topology, [])
    '''
    # Reverse Trajectory
    start_node_id = '65'
    init_info_tuple = ((2.4133644568462143, 2, 10), (6.00422282331718, 5, 1))  # Angle, speed_x, speed_y
    reverse_trajectory_algorithm(topology, start_node_id, init_info_tuple)
    '''
    '''
    # Trajectory Binary Search Test (One node, one angle)
    gravity_info = (acc_gravity, math.pi)
    start_node_id = '154'
    dest_node_id = '0'  # DNE Node, for testing trajectory
    # for neighbor in node_angle[start_node]:
    # init_angle = node_angle[start_node][neighbor]
    init_angle = 1.1631419984385158
    speed_x = 1
    trajectory_search_output = trajectory_search(topology, start_node_id, dest_node_id, init_angle, speed_x, gravity_info)
    copy_trajectory_search_output = {tuple(v): k for k, v in trajectory_search_output.items()}
    trajectory_search_output = {v: list(k) for k, v in copy_trajectory_search_output.items()}
    sorted_trajectory_search_output = dict(sorted(trajectory_search_output.items(), reverse=True))
    trajectory_search_output_dict = dict()
    for speed in sorted_trajectory_search_output:
        trajectory_search_output_dict[speed] = sorted_trajectory_search_output[speed]
    print(len(sorted_trajectory_search_output))
    for speed in trajectory_search_output_dict:
        print(speed * math.sin(init_angle), trajectory_search_output_dict[speed])
    print(trajectory_search_output_dict)
    # print(trajectory_search_output_list[0], trajectory_search_output_list[1])
    # print(trajectory_search_output_list[-2], trajectory_search_output_list[-1])
    multiple_trajectories_plot(topology, trajectory_search_output_dict)
    '''
    '''
    # Trajectory Binary Search Multiple Angles (One Node, All Angles) For Reachability Test
    start_node_id = '1'
    trajectory_search_list = []
    trajectory_search_angle = []
    gravity_info = (acc_gravity, 0 * math.pi / 4)
    dest_node_id = '0'  # DNE Node, for testing trajectory
    velocity_x_list = [num for num in range(-10, 11)]
    start_node = topology.get_node(start_node_id)
    for speed_x in velocity_x_list:
        trajectory_search_list = []
        for neighbor_id in start_node.neighbors:
            neighbor = topology.get_node(neighbor_id)
            if neighbor != start_node:
                init_angle = angle_link_calculation(start_node, neighbor)
                trajectory_search_angle = []
                trajectory_search_output = trajectory_search(topology, start_node_id, dest_node_id, init_angle, speed_x,
                                                             gravity_info)
                copy_trajectory_search_output = {tuple(v): k for k, v in trajectory_search_output.items()}
                trajectory_search_output = {v: list(k) for k, v in copy_trajectory_search_output.items()}
                sorted_trajectory_search_output = dict(sorted(trajectory_search_output.items(), reverse=True))
                trajectory_search_angle.append(init_angle)
                trajectory_search_angle.append(sorted_trajectory_search_output)
                trajectory_search_list.append(trajectory_search_angle)
        # print(trajectory_search_list)

    # For Reachability
    node_reachable = []
    non_reachable = []
    for angle_list in trajectory_search_list:
        speed_trajectory = angle_list[1:]
        for speed_trajectory_dict in speed_trajectory:
            for speed in speed_trajectory_dict:
                for node_id in speed_trajectory_dict[speed]:
                    if (node_id not in node_reachable) and (node_id != start_node.id):
                        node_reachable.append(node_id)
    for node in all_nodes:
        if (node.id not in node_reachable) and (node != start_node):
            non_reachable.append(node.id)
    non_reachable.sort()
    print("Node #" + start_node.id)
    print(len(all_nodes) - 1, "routes total,", len(node_reachable), "reachable routes,", len(non_reachable),
          "routes not reachable.")
    print("Non-Reachable Nodes:", non_reachable)
    print("Reachability Percentage: " + str(round((((len(node_reachable)) / (len(all_nodes) - 1)) * 100), 3)) + '%')
    # plot_topology(topology, node_coordinates, grid_size, grid_division_num, node_reachable)
    '''
    '''
    one_non_reachable_dict = dict()
    two_non_reachable_dict = dict()
    four_non_reachable_dict = dict()
    eight_non_reachable_dict = dict()
    for key in g0_dict:
        one_non_reachable_dict[key] = list()
        two_non_reachable_dict[key] = list()
        four_non_reachable_dict[key] = list()
        eight_non_reachable_dict[key] = list()

        one_non_reachable_dict[key].append(g270_dict[key])
        two_non_reachable_dict[key].append(g270_dict[key])
        four_non_reachable_dict[key].append(g270_dict[key])
        eight_non_reachable_dict[key].append(g270_dict[key])

        two_non_reachable_dict[key].append(g90_dict[key])
        four_non_reachable_dict[key].append(g90_dict[key])
        eight_non_reachable_dict[key].append(g90_dict[key])

        four_non_reachable_dict[key].append(g0_dict[key])
        four_non_reachable_dict[key].append(g180_dict[key])
        eight_non_reachable_dict[key].append(g0_dict[key])
        eight_non_reachable_dict[key].append(g180_dict[key])

        eight_non_reachable_dict[key].append(g45_dict[key])
        eight_non_reachable_dict[key].append(g135_dict[key])
        eight_non_reachable_dict[key].append(g225_dict[key])
        eight_non_reachable_dict[key].append(g315_dict[key])

    # output_set = set(combined_list[0]).intersection(*combined_list)
    # print(len(output_set), list(output_set))

    for node_id in one_non_reachable_dict:
        one_list = one_non_reachable_dict[node_id]
        one_combine_list = set(one_list[0]).intersection(*one_list)
        two_list = two_non_reachable_dict[node_id]
        two_combine_list = set(two_list[0]).intersection(*two_list)
        four_list = four_non_reachable_dict[node_id]
        four_combine_list = set(four_list[0]).intersection(*four_list)
        eight_list = eight_non_reachable_dict[node_id]
        eight_combine_list = set(eight_list[0]).intersection(*eight_list)

        # if len(four_combine_list) > 1:
        print(node_id, len(one_combine_list), len(two_combine_list), len(four_combine_list), len(eight_combine_list))

        # Non-Reachable Node Plot Node 89
        if node_id == '89':
            print(len(eight_combine_list), eight_combine_list)
            nonreachable_node_plot(topology, '89', eight_combine_list)
    '''
    '''
    # Trajectory Binary Search For All Nodes, Angles, and g-direction For Reachability Test
    all_non_reachable_routes = []
    extensive_reachable_dict = dict()
    extensive_non_reachable_dict = dict()
    reachable_count = 0
    num_g_direction = 8
    gravity_info_list = []
    for index in range(0, num_g_direction):
        g_direction = index * 2 * math.pi / num_g_direction
        gravity_info_list.append((acc_gravity, g_direction))
    print(gravity_info_list)
    for start_node in all_nodes:
        trajectory_search_list = []
        dest_node_id = '0'  # DNE Node, for testing trajectory
        speed_x_list = [num for num in range(-10, 11)]
        for speed_x in speed_x_list:
            for gravity_info in gravity_info_list:
                for neighbor_id in start_node.neighbors:
                    neighbor = topology.get_node(neighbor_id)
                    if neighbor != start_node:
                        init_angle = angle_link_calculation(start_node, neighbor)
                        trajectory_search_angle = []
                        trajectory_search_output = trajectory_search(topology, start_node.id, dest_node_id, init_angle,
                                                                     speed_x, gravity_info)
                        copy_trajectory_search_output = {tuple(v): k for k, v in trajectory_search_output.items()}
                        trajectory_search_output = {v: list(k) for k, v in copy_trajectory_search_output.items()}
                        sorted_trajectory_search_output = dict(sorted(trajectory_search_output.items(), reverse=True))
                        trajectory_search_angle.append(init_angle)
                        trajectory_search_angle.append(sorted_trajectory_search_output)
                        trajectory_search_list.append(trajectory_search_angle)
        # print(trajectory_search_list)
        # print()

        # For Reachability
        node_reachable = []
        non_reachable = []
        for angle_list in trajectory_search_list:
            speed_trajectory = angle_list[1:]
            for speed_trajectory_dict in speed_trajectory:
                for speed in speed_trajectory_dict:
                    for node_id in speed_trajectory_dict[speed]:
                        if (node_id not in node_reachable) and (node_id != start_node.id):
                            node_reachable.append(node_id)
        for node in all_nodes:
            if (node.id not in node_reachable) and (node != start_node):
                non_reachable.append(node.id)
                all_non_reachable_routes.append((start_node.id, node.id))

        reachable_count += len(node_reachable)
        non_reachable.sort()
        print("Node #" + start_node.id)
        print(len(all_nodes) - 1, "routes total,", len(node_reachable), "reachable routes,", len(non_reachable), "routes not reachable.")
        print("Non-Reachable Nodes:", non_reachable)
        print("Reachability Percentage: " + str(round((((len(node_reachable)) / (len(all_nodes) - 1)) * 100), 3)) + '%')
        extensive_reachable_dict[start_node.id] = node_reachable
        if len(non_reachable) != 0:
            extensive_non_reachable_dict[start_node.id] = non_reachable

    print()
    print("Total Number of Nodes:", len(all_nodes))
    print("Total Number of Possible Routes:", (len(all_nodes) * (len(all_nodes) - 1)))
    print("Total Number of Reachable Routes:", reachable_count)
    print("Overall Average Reachability Percentage: " + str(round(((reachable_count / (len(all_nodes) * (len(all_nodes) - 1))) * 100), 3)) + '%')
    print("All Non-Reachable Routes:", all_non_reachable_routes)
    print()
    print()
    print(extensive_non_reachable_dict)
    print()

    # Heat Map (Reachability Test in each Region)
    print("Union of all 8 g direction")
    content_output = extensive_non_reachable_dict
    region_division_num = 4
    reachability_heat_map = heat_map(topology, grid_size, region_division_num, content_output)
    sum_total = 0
    sum_reachable = 0
    for regions in reachability_heat_map:
        sum_total += reachability_heat_map[regions][0]
        sum_reachable += reachability_heat_map[regions][1]
    regional_overall_reachability_heat_map = dict()

    for start_region in range(1, int(math.pow(region_division_num, 2) + 1)):
        regional_overall_reachability_heat_map[start_region] = []
        all_combinations = 0
        all_reachable = 0
        for dest_region in range(1, int(math.pow(region_division_num, 2) + 1)):
            region = (start_region, dest_region)
            all_combinations += reachability_heat_map[region][0]
            all_reachable += reachability_heat_map[region][1]
        all_reachability_percentage = (all_reachable / all_combinations) * 100
        regional_overall_reachability_heat_map[start_region].append(all_combinations)
        regional_overall_reachability_heat_map[start_region].append(all_reachable)
        regional_overall_reachability_heat_map[start_region].append(all_reachability_percentage)
        print()
        print("Regional #" + str(start_region) + " Possible Routes:", str(all_combinations))
        print("Regional #" + str(start_region) + " Reachable Routes:", str(all_reachable))
        print("Regional #" + str(start_region) + " Reachability Percentage:", str(all_reachability_percentage))

    print()
    print("Total Possible Routes:", str(sum_total))
    print("Reachable Routes:", str(sum_reachable))
    print("Reachability Percentage:", str((sum_reachable / sum_total) * 100))
    '''
    '''
    # Random Walk
    start_node = '57'
    dest_node = '10'
    count = 0
    total = 0
    for seed in range(100):
        count += 1
        random.seed(seed)
        random_walk_output = random_walk(topology, start_node, dest_node)
        hops = len(random_walk_output) - 1
        total += hops
    average_hops = total / count
    print("Random Walk Average Hops: " + str(average_hops))
    '''
    '''
    # Reachability Test
    start_node = '26'
    dest_node = '82'
    # random_walk_output = random_walk(topology, start_node, dest_node)
    print()
    route = reachability(node_angle, node_coordinates, start_node, dest_node, delta_t)
    print("Initial Node: " + start_node)
    print("Destination Node: " + dest_node)
    # print("Random Walk Route: " + str(random_walk_output) + str(len(random_walk_output)-1))
    # plot_topology(topology, node_coordinates, grid_size, grid_division_num, random_walk_output)
    print(len(route), "Route(s) Available: ")
    for routes in route:
        print("Gravity Direction: " + str(routes[1][1]) + " deg, Gravity Value: " + str(routes[1][0])
              + " m/s^2, Initial Speed: " + str(routes[0][0]) + " m/s, Initial Angle: " + str(routes[0][1])
              + " degree\n" + "Route: " + str(route[routes]) + ", Number of Hops: " + str(len(route[routes]) - 1))
        plot_topology(topology, node_coordinates, grid_size, grid_division_num, route[routes])
    print()
    '''
    '''
    # Reachability and Shortest Path Extensive Test
    print()
    print("Reachability Test")
    count = 0
    count_total = 0
    count_non = 0
    # count_shortest = 0
    non_reachable = []
    start_node = '69'
    node_reachable = [start_node]
    # for start_node in node_list:
    for dest_node in node_list:
        if (start_node != dest_node) and (node_coordinates[start_node][1] > 0.5):
            # path_shortest = shortest_path(grid_dict, start_node, dest_node)
            count_total += 1
            print("Initial Node:", start_node)
            print("Destination Node:", dest_node)
            route = reachability(node_angle, node_coordinates, start_node, dest_node, delta_t)
            print("Number of routes available:", len(route))
            if len(route) > 0:
                node_reachable.append(dest_node)
                count += 1
                # for r in route.items():
                # if len(r[1]) == len(path_shortest):
                # count_shortest += 1
            else:
                non_reachable.append((start_node, dest_node))
                count_non += 1
            for routes in route:
                print("Gravity Direction: " + str(routes[1][1]) + " deg, Gravity Value: " + str(routes[1][0])
                      + " m/s^2, Initial Speed: " + str(routes[0][0]) + " m/s, Initial Angle: " + str(routes[0][1])
                      + " degree\n" + "Route: " + str(route[routes]) + ", Number of Hops: " + str(
                    len(route[routes]) - 1))
            print()
    print()

    print()
    print("Total number of nodes:", len(node_list))
    print(count_total, "routes total,", count, "reachable routes,", count_non, "routes not reachable.")
    # print(count_shortest, "performs shortest routes.")
    print("Non-Reachable routes:", non_reachable)
    plot_topology(topology, node_coordinates, grid_size, grid_division_num, node_reachable)
    '''

    # Dijkstra Shortest Path
    all_sp = dijkstra_sp(topology)


    # Runtime Check
    print()
    print("Execution Time: " + str(time.process_time() - start_time) + " seconds")


if __name__ == '__main__':
    main()
