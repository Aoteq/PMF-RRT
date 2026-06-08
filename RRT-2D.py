import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time

CIRCULAR_OBSTACLES = [  # [center_x, center_y, radius]
    [3, 3, 1.0], [2.5, 6, 0.8], [5, 8, 1.2],
    [7, 2, 0.9], [8, 6, 1.1], [6, 12, 1.0],
    [10, 4, 0.7], [11, 9, 1.3], [9, 15, 0.9],
    [13, 1, 1.0], [14, 6, 0.8], [12, 13, 1.1],
    [16, 3, 1.2], [17, 10, 1.0], [15, 16, 0.7],
    [1, 12, 0.6], [4, 16, 1.0], [8, 18, 0.8],
    [13, 18, 1.0], [18, 14, 0.9], [19, 5, 1.1],
    [1.5, 1.5, 0.5], [4.5, 1.0, 0.7],
    [10.5, 1.5, 0.8], [13.5, 4.0, 0.5], [16.5, 1.0, 0.7],
    [1.0, 15.0, 0.8], [6.5, 17.0, 0.6], [11.5, 16.5, 0.9],
    [15.5, 13.5, 0.5], [18.5, 17.5, 0.7], [19, 10, 0.6],
    [5, 14, 0.7], [9, 11, 0.5], [13, 8, 0.6]
]

START_POS = np.array([18.0, 0.0])
END_POS = np.array([5, 13.0])

SPACE_X_RANGE = (0.0, 20.0)
SPACE_Y_RANGE = (0.0, 20.0)

MAX_ITER = 1500
STEP_SIZE = 0.5
GOAL_REACH_THRESHOLD = 0.7


class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent


def get_distance(state1, state2):
    return np.linalg.norm(np.array(state1) - np.array(state2))


def is_collision(state, circular_obstacles):
    if not (SPACE_X_RANGE[0] <= state[0] <= SPACE_X_RANGE[1] and \
            SPACE_Y_RANGE[0] <= state[1] <= SPACE_Y_RANGE[1]):
        return True
    for obs_cx, obs_cy, obs_r in circular_obstacles:
        if get_distance(state, [obs_cx, obs_cy]) <= obs_r:
            return True
    return False


def is_path_collision_free(start_state, end_state, circular_obstacles, step_size_for_check):
    start_state_arr = np.array(start_state)
    end_state_arr = np.array(end_state)
    direction = end_state_arr - start_state_arr
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return not is_collision(start_state_arr, circular_obstacles)
    unit_direction = direction / dist
    num_steps = max(1, int(dist / (min(step_size_for_check, 0.1) / 2)))
    for i in range(num_steps + 1):
        current_pos = start_state_arr + unit_direction * (i * dist / num_steps)
        if is_collision(current_pos, circular_obstacles):
            return False
    return True


def get_nearest_node(node_list, target_state):
    min_dist = float('inf');
    nearest_node = None
    for node in node_list:
        dist = get_distance(node.state, target_state)
        if dist < min_dist: min_dist = dist; nearest_node = node
    return nearest_node


def steer(from_node_state, to_state, step_size):
    direction = np.array(to_state) - np.array(from_node_state)
    dist = np.linalg.norm(direction)
    if dist <= step_size:
        return np.array(to_state)
    else:
        return np.array(from_node_state) + (direction / dist) * step_size


def vanilla_rrt_planning(start_pos, end_pos, obstacles, space_x_range, space_y_range,
                         max_iter, step_size, goal_reach_threshold,
                         visualize_progress=False):
    start_node = Node(start_pos)
    end_node_state = np.array(end_pos)

    node_list = [start_node]
    path_found = False
    final_node = None

    for i in range(max_iter):
        random_state = np.array([random.uniform(space_x_range[0], space_x_range[1]),
                                 random.uniform(space_y_range[0], space_y_range[1])])

        expand_from_node = get_nearest_node(node_list, random_state)

        new_state_candidate = steer(expand_from_node.state, random_state, step_size)

        path_is_free = is_path_collision_free(expand_from_node.state, new_state_candidate, obstacles, step_size)

        current_parent_for_plot = expand_from_node.state
        current_new_for_plot, collision_start_for_plot, collision_end_for_plot = None, None, None

        if path_is_free:
            new_node = Node(new_state_candidate, parent=expand_from_node)
            node_list.append(new_node)
            current_new_for_plot = new_node.state

            if get_distance(new_node.state, end_node_state) <= goal_reach_threshold:
                if is_path_collision_free(new_node.state, end_node_state, obstacles, STEP_SIZE):
                    final_node = Node(end_node_state, parent=new_node)
                    node_list.append(final_node)
                    path_found = True
                    print(f"Vanilla RRT: Goal reached at iteration {i + 1}!")
                else:
                    print(f"Vanilla RRT: Near goal at iter {i + 1}, but final connection to exact goal blocked.")

        else:
            collision_start_for_plot = expand_from_node.state
            collision_end_for_plot = new_state_candidate

        if visualize_progress and (i % 100 == 0 or path_found or (not path_is_free and i % 20 == 0)):
            # MODIFIED: Pass is_final_plot=False for animation frames
            plot_environment(node_list, start_pos, end_pos, obstacles, space_x_range, space_y_range,
                             title=f"Vanilla RRT (Iter: {i + 1}) {'Col' if not path_is_free else ''}",
                             final_path=(reconstruct_path(final_node) if path_found else None),
                             current_new_node_state=current_new_for_plot, current_parent_state=current_parent_for_plot,
                             collision_attempt_start=collision_start_for_plot,
                             collision_attempt_end=collision_end_for_plot,
                             is_final_plot=False)
        if path_found: break

    if path_found:
        path = reconstruct_path(final_node)
        return path, node_list
    else:
        print("Vanilla RRT: Failed to find a path after max iterations.")
        if visualize_progress:
            plot_environment(node_list, start_pos, end_pos, obstacles, space_x_range, space_y_range,
                             title=f"Vanilla RRT - Failed (Iter: {max_iter})", is_final_plot=False)
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

    print("\nRunning Vanilla RRT Algorithm...")
    start_time = time.time()

    vanilla_rrt_path, vanilla_rrt_node_list = vanilla_rrt_planning(
        START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
        MAX_ITER, STEP_SIZE, GOAL_REACH_THRESHOLD,
        visualize_progress=True
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Planning finished in: {elapsed_time:.2f} seconds")

    fig_final = plt.figure(figsize=(8, 8))

    if vanilla_rrt_path:
        path_length = sum(
            np.linalg.norm(vanilla_rrt_path[i] - vanilla_rrt_path[i - 1]) for i in range(1, len(vanilla_rrt_path)))
        print(f"Vanilla RRT Path found with {len(vanilla_rrt_path)} points. Length: {path_length:.2f} m")
        final_title = "Path Planning Result using RRT"
        plot_environment(vanilla_rrt_node_list, START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
                         title=final_title,
                         final_path=vanilla_rrt_path,
                         is_final_plot=True)
    else:
        print("Vanilla RRT failed to find a path.")
        final_title = "RRT Failed to Find a Path"
        plot_environment(vanilla_rrt_node_list, START_POS, END_POS, CIRCULAR_OBSTACLES, SPACE_X_RANGE, SPACE_Y_RANGE,
                         title=final_title,
                         is_final_plot=True)

    plt.savefig("rrt_path_result.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("rrt_path_result.png", format='png', dpi=300, bbox_inches='tight')

    print("\nFinal plot saved as 'rrt_path_result.pdf' and 'rrt_path_result.png'")

    plt.ioff()
    plt.show()
