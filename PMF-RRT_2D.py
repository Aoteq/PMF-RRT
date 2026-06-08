import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time

CIRCULAR_OBSTACLES = [
    [3, 3, 1.0], [2.5, 6, 0.8], [5, 8, 1.2], [7, 2, 0.9], [8, 6, 1.1],
    [6, 12, 1.0], [10, 4, 0.7], [11, 9, 1.3], [9, 15, 0.9], [13, 1, 1.0],
    [14, 6, 0.8], [12, 13, 1.1], [16, 3, 1.2], [17, 10, 1.0], [15, 16, 0.7],
    [1, 12, 0.6], [4, 16, 1.0], [8, 18, 0.8], [13, 18, 1.0], [18, 14, 0.9],
    [19, 5, 1.1], [1.5, 1.5, 0.5], [4.5, 1.0, 0.7],
    [10.5, 1.5, 0.8], [13.5, 4.0, 0.5], [16.5, 1.0, 0.7], [1.0, 15.0, 0.8],
    [6.5, 17.0, 0.6], [11.5, 16.5, 0.9], [15.5, 13.5, 0.5], [18.5, 17.5, 0.7],
    [19, 10, 0.6], [5, 14, 0.7], [9, 11, 0.5], [13, 8, 0.6]
]
START_POS = np.array([18.0, 0.0])
END_POS = np.array([5, 13.0])

SPACE_X_RANGE = (0.0, 20.0)
SPACE_Y_RANGE = (0.0, 20.0)

MAX_ITER = 1500
STEP_SIZE = 0.5
GOAL_REACH_THRESHOLD = 0.7

INITIAL_GOAL_BIAS = 0.01
TARGET_MAX_GOAL_BIAS_A = 0.2
GOAL_BIAS_RATE_K = 15.0

PDF_NUM_BINS = 4
PDF_UPDATE_FACTOR = 0.5
PDF_INFLUENCE_WIDTH_BINS = 4
PDF_MATURITY_THRESHOLD = 90
PDF_POSITIVE_FACTOR = 1.21


class Node:
    def __init__(self, state, parent=None, pdf_num_bins=PDF_NUM_BINS):
        self.state = np.array(state)
        self.parent = parent
        self.cost = 0.0
        if parent:
            self.cost = parent.cost + np.linalg.norm(self.state - parent.state)
        self._pdf_num_bins = pdf_num_bins
        self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        self.pdf_update_count = 0

    def initialize_pdf(self, method='uniform', parent_pdf_data=None):
        if method == 'uniform' or parent_pdf_data is None:
            self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        elif method == 'inherit_parent' and parent_pdf_data is not None:
            self.local_direction_pdf = 0.7 * np.copy(parent_pdf_data) + 0.3 * (
                        np.ones(self._pdf_num_bins) / self._pdf_num_bins)
            if np.sum(self.local_direction_pdf) > 1e-9:
                self.local_direction_pdf /= np.sum(self.local_direction_pdf)
            else:
                self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        else:
            self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        self.pdf_update_count = 0


def get_distance(state1, state2): return np.linalg.norm(np.array(state1) - np.array(state2))


def is_collision(state, circular_obstacles):
    if not (SPACE_X_RANGE[0] <= state[0] <= SPACE_X_RANGE[1] and SPACE_Y_RANGE[0] <= state[1] <= SPACE_Y_RANGE[
        1]): return True
    for obs_cx, obs_cy, obs_r in circular_obstacles:
        if get_distance(state, [obs_cx, obs_cy]) <= obs_r: return True
    return False


def is_path_collision_free(start_state, end_state, circular_obstacles, step_size_for_check):
    start_state_arr = np.array(start_state);
    end_state_arr = np.array(end_state)
    direction = end_state_arr - start_state_arr;
    dist = np.linalg.norm(direction)
    if dist < 1e-6: return not is_collision(start_state_arr, circular_obstacles)
    unit_direction = direction / dist
    num_steps = max(1, int(dist / (min(step_size_for_check, 0.1) / 2)))
    for i in range(num_steps + 1):
        current_pos = start_state_arr + unit_direction * (i * dist / num_steps)
        if is_collision(current_pos, circular_obstacles): return False
    return True


def get_nearest_node(node_list, target_state):
    min_dist = float('inf');
    nearest_node = None
    for node in node_list:
        dist = get_distance(node.state, target_state)
        if dist < min_dist: min_dist = dist; nearest_node = node
    return nearest_node


def sample_from_pdf(pdf_array):
    if np.sum(pdf_array) < 1e-9:
        selected_bin_index = random.randint(0, len(pdf_array) - 1)
    else:
        probabilities = pdf_array / np.sum(pdf_array)
        selected_bin_index = np.random.choice(len(pdf_array), p=probabilities)
    angle_per_bin = 2 * np.pi / len(pdf_array)
    return selected_bin_index * angle_per_bin + angle_per_bin / 2


def update_pdf_on_collision(node_to_update_pdf, collided_angle_rad, update_factor, influence_width_bins):
    pdf_array = node_to_update_pdf.local_direction_pdf;
    num_total_bins = node_to_update_pdf._pdf_num_bins
    angle_per_bin = 2 * np.pi / num_total_bins
    collided_bin_index = int(round(collided_angle_rad / angle_per_bin)) % num_total_bins
    pdf_array[collided_bin_index] *= update_factor
    for i in range(1, influence_width_bins + 1):
        factor_neighbor = update_factor + (1.0 - update_factor) * 0.5
        left_neighbor_idx = (collided_bin_index - i + num_total_bins) % num_total_bins
        pdf_array[left_neighbor_idx] *= factor_neighbor
        right_neighbor_idx = (collided_bin_index + i) % num_total_bins
        pdf_array[right_neighbor_idx] *= factor_neighbor
    current_sum = np.sum(pdf_array)
    if current_sum > 1e-9:
        pdf_array /= current_sum
    else:
        pdf_array[:] = 1.0 / num_total_bins
    node_to_update_pdf.pdf_update_count += 1


def positive_feedback_to_pdf(node_to_update_pdf, success_angle_rad, positive_factor, influence_width_bins):
    pdf_array = node_to_update_pdf.local_direction_pdf;
    num_total_bins = node_to_update_pdf._pdf_num_bins
    angle_per_bin = 2 * np.pi / num_total_bins
    success_bin_index = int(round(success_angle_rad / angle_per_bin)) % num_total_bins
    pdf_array[success_bin_index] *= positive_factor
    for i in range(1, influence_width_bins + 1):
        factor_neighbor = 1.0 + (positive_factor - 1.0) * 0.5
        left_neighbor_idx = (success_bin_index - i + num_total_bins) % num_total_bins
        pdf_array[left_neighbor_idx] *= factor_neighbor
        right_neighbor_idx = (success_bin_index + i) % num_total_bins
        pdf_array[right_neighbor_idx] *= factor_neighbor
    current_sum = np.sum(pdf_array)
    if current_sum > 1e-9:
        pdf_array /= current_sum
    else:
        pdf_array[:] = 1.0 / num_total_bins
    node_to_update_pdf.pdf_update_count += 1


def calculate_dynamic_goal_bias(current_iter, max_iter, initial_bias, target_max_bias, rate_k):
    if max_iter == 0: return target_max_bias
    if current_iter == 0 and initial_bias > 0: return initial_bias
    effective_max_growth = target_max_bias - initial_bias
    if effective_max_growth <= 0: return initial_bias
    growth_factor = 1 - math.exp(-rate_k * (current_iter / max_iter))
    dynamic_bias = initial_bias + effective_max_growth * growth_factor
    return np.clip(dynamic_bias, initial_bias, target_max_bias)


def pmf_rrt_planning(start_pos, end_pos, obstacles, space_x_range, space_y_range,
                     max_iter, step_size, goal_reach_threshold,
                     pdf_num_bins, pdf_update_factor, pdf_influence_width_bins,
                     pdf_maturity_threshold, pdf_positive_factor,
                     initial_goal_bias_param, target_max_goal_bias_param, goal_bias_rate_k_param,
                     visualize_progress=False):
    start_node = Node(start_pos, pdf_num_bins=pdf_num_bins)
    end_node_state = np.array(end_pos)
    node_list = [start_node]
    path_found = False
    final_goal_node = None

    for i in range(max_iter):
        q_rand_for_selection = np.array([random.uniform(space_x_range[0], space_x_range[1]),
                                         random.uniform(space_y_range[0], space_y_range[1])])
        expand_from_node = get_nearest_node(node_list, q_rand_for_selection)

        new_state_candidate = None
        actual_direction_angle_rad = None
        is_goal_bias_expansion = False
        is_pdf_driven_expansion = False

        current_goal_bias = calculate_dynamic_goal_bias(i, max_iter,
                                                        initial_goal_bias_param,
                                                        target_max_goal_bias_param,
                                                        goal_bias_rate_k_param)

        if random.random() < current_goal_bias:
            is_goal_bias_expansion = True
            direction_to_goal = end_node_state - expand_from_node.state
            dist_to_goal = np.linalg.norm(direction_to_goal)
            if dist_to_goal > 1e-6:
                unit_direction_to_goal = direction_to_goal / dist_to_goal
                new_state_candidate = expand_from_node.state + unit_direction_to_goal * step_size
                if dist_to_goal <= step_size: new_state_candidate = end_node_state
                actual_direction_angle_rad = math.atan2(unit_direction_to_goal[1], unit_direction_to_goal[0])
                if actual_direction_angle_rad < 0: actual_direction_angle_rad += 2 * np.pi
            else:
                new_state_candidate = expand_from_node.state
        elif expand_from_node.pdf_update_count < pdf_maturity_threshold:
            is_pdf_driven_expansion = False
            q_rand_external = np.array([random.uniform(space_x_range[0], space_x_range[1]),
                                        random.uniform(space_y_range[0], space_y_range[1])])
            direction_to_external_rand = q_rand_external - expand_from_node.state
            dist_to_external_rand = np.linalg.norm(direction_to_external_rand)
            if dist_to_external_rand > 1e-6:
                unit_direction_to_external_rand = direction_to_external_rand / dist_to_external_rand
                new_state_candidate = expand_from_node.state + unit_direction_to_external_rand * step_size
                actual_direction_angle_rad = math.atan2(unit_direction_to_external_rand[1],
                                                        unit_direction_to_external_rand[0])
                if actual_direction_angle_rad < 0: actual_direction_angle_rad += 2 * np.pi
            else:
                new_state_candidate = expand_from_node.state
        else:
            is_pdf_driven_expansion = True
            sampled_angle_rad = sample_from_pdf(expand_from_node.local_direction_pdf)
            actual_direction_angle_rad = sampled_angle_rad
            direction_vector = np.array([math.cos(sampled_angle_rad), math.sin(sampled_angle_rad)])
            new_state_candidate = expand_from_node.state + step_size * direction_vector

        if new_state_candidate is None: continue

        path_is_free = is_path_collision_free(expand_from_node.state, new_state_candidate, obstacles, step_size)

        current_parent_for_plot = expand_from_node.state
        current_new_for_plot, collision_start_for_plot, collision_end_for_plot = None, None, None

        if path_is_free:
            new_node = Node(new_state_candidate, parent=expand_from_node, pdf_num_bins=pdf_num_bins)
            new_node.initialize_pdf(method='inherit_parent', parent_pdf_data=expand_from_node.local_direction_pdf)
            node_list.append(new_node)
            current_new_for_plot = new_node.state

            if is_pdf_driven_expansion and actual_direction_angle_rad is not None:
                positive_feedback_to_pdf(expand_from_node,
                                         actual_direction_angle_rad,
                                         pdf_positive_factor,
                                         pdf_influence_width_bins)

            if get_distance(new_node.state, end_node_state) <= goal_reach_threshold:
                if is_path_collision_free(new_node.state, end_node_state, obstacles, STEP_SIZE):
                    final_goal_node = Node(end_node_state, parent=new_node, pdf_num_bins=pdf_num_bins)
                    path_found = True
                    print(f"PMF-RRT DynGB (NoDC) (Iter: {i + 1}): Goal region reached and final connection successful!")
                else:
                    print(
                        f"PMF-RRT DynGB (NoDC) (Iter: {i + 1}): Near goal, but final connection to exact goal failed.")

        else:
            if actual_direction_angle_rad is not None:
                update_pdf_on_collision(expand_from_node,
                                        actual_direction_angle_rad,
                                        pdf_update_factor,
                                        pdf_influence_width_bins)
            collision_start_for_plot = expand_from_node.state
            collision_end_for_plot = new_state_candidate
            if expand_from_node.pdf_update_count > 2 * pdf_maturity_threshold + 10:
                print(f"Resetting PMF for node at {expand_from_node.state} due to high collisions.")
                expand_from_node.initialize_pdf(method='uniform')

        if visualize_progress and (i % 50 == 0 or path_found or (not path_is_free and i % 10 == 0)):
            title_str = f"PMF-RRT DynGB (NoDC) (Iter: {i + 1}, GB:{current_goal_bias:.2f}) "
            if not path_is_free: title_str += "Col "
            if is_goal_bias_expansion:
                title_str += "GBexp "
            elif not is_pdf_driven_expansion:
                title_str += "ExtRand "
            else:
                title_str += "PDFdrv "
            plot_environment(node_list, start_pos, end_pos, obstacles, space_x_range, space_y_range,
                             title=title_str,
                             final_path=(reconstruct_path(final_goal_node) if path_found and final_goal_node else None),
                             current_new_node_state=current_new_for_plot, current_parent_state=current_parent_for_plot,
                             collision_attempt_start=collision_start_for_plot,
                             collision_attempt_end=collision_end_for_plot,
                             is_final_plot=False)
        if path_found: break

    if path_found and final_goal_node:
        if final_goal_node.parent in node_list and final_goal_node not in node_list:
            node_list.append(final_goal_node)

        path = reconstruct_path(final_goal_node)
        return path, node_list
    else:
        print("PMF-RRT DynGB (NoDC): Failed to find a path after max iterations.")
        if visualize_progress:
            plot_environment(node_list, start_pos, end_pos, obstacles, space_x_range, space_y_range,
                             title=f"PMF-RRT DynGB (NoDC) - Failed (Iter: {max_iter})")
        return None, node_list


def reconstruct_path(goal_node):
    path = [];
    current_node = goal_node
    while current_node is not None: path.append(current_node.state); current_node = current_node.parent
    return path[::-1]


# ==============================================================================
# ==============================================================================
def plot_environment(node_list, start_pos, end_pos, circular_obstacles, x_range, y_range,
                     title="RRT Environment", final_path=None,
                     current_new_node_state=None, current_parent_state=None,
                     collision_attempt_start=None, collision_attempt_end=None,
                     is_final_plot=False):
    """
    绘制环境、RRT树和路径。
    is_final_plot=True时，应用SCI论文风格。
    """
    plt.clf()
    ax = plt.gca()

    if is_final_plot:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
        })

    for cx, cy, r in circular_obstacles:
        circle = patches.Circle((cx, cy), r, linewidth=1.5, edgecolor='black', facecolor='gray')
        ax.add_patch(circle)

    tree_color = 'cornflowerblue' if is_final_plot else 'skyblue'
    tree_linewidth = 0.8 if is_final_plot else 0.7
    for node in node_list:
        if node.parent:
            plt.plot([node.state[0], node.parent.state[0]], [node.state[1], node.parent.state[1]],
                     color=tree_color, linestyle='-', linewidth=tree_linewidth, alpha=0.9)

    if not is_final_plot:
        if current_new_node_state is not None and current_parent_state is not None:
            plt.plot([current_new_node_state[0], current_parent_state[0]],
                     [current_new_node_state[1], current_parent_state[1]],
                     color='blue', linestyle='-', linewidth=1.2, label='Current Extension')
        if collision_attempt_start is not None and collision_attempt_end is not None:
            plt.plot([collision_attempt_start[0], collision_attempt_end[0]],
                     [collision_attempt_start[1], collision_attempt_end[1]],
                     color='orangered', linestyle='--', linewidth=1.0, label='Collision Attempt')
            plt.plot(collision_attempt_start[0], collision_attempt_start[1], 'o', color='yellow', markersize=4)

    plt.plot(start_pos[0], start_pos[1], marker='o', color='blue', markersize=10, label='Start', zorder=10)
    plt.plot(end_pos[0], end_pos[1], marker='*', color='#2ca02c', markersize=15, label='Goal', zorder=10)

    if final_path:
        path_states = np.array(final_path)
        plt.plot(path_states[:, 0], path_states[:, 1], color='red', linestyle='-', linewidth=2.0, label='Final Path',
                 zorder=5)

    plt.xlim(x_range)
    plt.ylim(y_range)
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)

    if final_path:
        plt.legend(loc='best')

    plt.grid(True, linestyle='--', alpha=0.6)
    if not is_final_plot:
        plt.pause(0.001)


# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    plt.ion()
    fig_animation = plt.figure(figsize=(10, 10))

    print("\nRunning PMF-RRT Algorithm with Dynamic Goal Bias (No Direct Connect Opt)...")
    start_time = time.time()

    pmf_rrt_path, pmf_rrt_node_list = pmf_rrt_planning(
        START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
        MAX_ITER, STEP_SIZE, GOAL_REACH_THRESHOLD,
        PDF_NUM_BINS, PDF_UPDATE_FACTOR, PDF_INFLUENCE_WIDTH_BINS,
        PDF_MATURITY_THRESHOLD, PDF_POSITIVE_FACTOR,
        INITIAL_GOAL_BIAS, TARGET_MAX_GOAL_BIAS_A, GOAL_BIAS_RATE_K,
        visualize_progress=True
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Planning finished in: {elapsed_time:.2f} seconds")

    fig_final = plt.figure(figsize=(8, 8))

    if pmf_rrt_path:
        path_length = sum(np.linalg.norm(pmf_rrt_path[i] - pmf_rrt_path[i - 1]) for i in range(1, len(pmf_rrt_path)))
        print(f"PMF-RRT (DynGB, NoDC) Path found with {len(pmf_rrt_path)} points. Length: {path_length:.2f} m")
        final_title = "Path Planning Result using PMF-RRT"
        plot_environment(pmf_rrt_node_list, START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
                         title=final_title,
                         final_path=pmf_rrt_path,
                         is_final_plot=True)
    else:
        print("PMF-RRT (DynGB, NoDC) failed to find a path.")
        final_title = "PMF-RRT Failed to Find a Path"
        plot_environment(pmf_rrt_node_list, START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
                         title=final_title,
                         is_final_plot=True)

    plt.savefig("pmf_rrt_path_result.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("pmf_rrt_path_result.png", format='png', dpi=300, bbox_inches='tight')

    print("\nFinal plot saved as 'pmf_rrt_path_result.pdf' and 'pmf_rrt_path_result.png'")

    plt.ioff()
    plt.show()
