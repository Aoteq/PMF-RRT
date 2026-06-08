import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time
import uuid

collision_check_count = 0
collision_count = 0

class Environment:
    def __init__(self, start, goal, obstacles, bounds):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds          # [xmin, xmax, ymin, ymax, zmin, zmax]
        self.space_xyz_range = ([bounds[0], bounds[1]], [bounds[2], bounds[3]], [bounds[4], bounds[5]])

    def is_in_free_space(self, point):
        """检查点是否在自由空间（不与障碍物碰撞）"""
        global collision_check_count
        collision_check_count += 1
        for obs in self.obstacles:
            center, radius = obs[:3], obs[3]
            if np.linalg.norm(np.array(point) - np.array(center)) < radius:
                return False
        return (self.bounds[0] <= point[0] <= self.bounds[1] and
                self.bounds[2] <= point[1] <= self.bounds[3] and
                self.bounds[4] <= point[2] <= self.bounds[5])

    def is_collision_free(self, q1, q2, step_size_for_check=0.1):
        """检查线段 q1 → q2 是否无碰撞"""
        global collision_check_count, collision_count
        q1, q2 = np.array(q1), np.array(q2)
        direction = q2 - q1
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            collision_check_count += 1
            return self.is_in_free_space(q1)
        unit_direction = direction / dist
        num_steps = max(1, int(dist / step_size_for_check))
        for i in range(num_steps + 1):
            current_pos = q1 + unit_direction * (i * dist / num_steps)
            if not self.is_in_free_space(current_pos):
                collision_count += 1
                return False
        return True

class Node:
    def __init__(self, state, parent=None, pdf_num_bins=50):
        self.state = np.array(state)  # (x, y, z)
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

class PMFRRTPlanner:
    def __init__(self, env, step_size=0.9, max_iterations=5000, goal_reach_threshold=1.5,
                 pdf_num_bins=50, pdf_update_factor=0.5, pdf_influence_width_bins=5,
                 pdf_maturity_threshold=30, pdf_positive_factor=1.11,
                 initial_goal_bias=0.01, target_max_goal_bias=0.2, goal_bias_rate_k=39.0):
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_reach_threshold = goal_reach_threshold
        self.pdf_num_bins = pdf_num_bins
        self.pdf_update_factor = pdf_update_factor
        self.pdf_influence_width_bins = pdf_influence_width_bins
        self.pdf_maturity_threshold = pdf_maturity_threshold
        self.pdf_positive_factor = pdf_positive_factor
        self.initial_goal_bias = initial_goal_bias
        self.target_max_goal_bias = target_max_goal_bias
        self.goal_bias_rate_k = goal_bias_rate_k
        self.nodes = [Node(env.start, pdf_num_bins=pdf_num_bins)]
        self.all_spheres = []

    def get_distance(self, state1, state2):
        return np.linalg.norm(np.array(state1) - np.array(state2))

    def get_nearest_node(self, target_state):
        min_dist = float('inf')
        nearest_node = None
        for node in self.nodes:
            dist = self.get_distance(node.state, target_state)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def sample_from_pdf(self, pdf_array):
        if np.sum(pdf_array) < 1e-9:
            selected_bin_index = random.randint(0, len(pdf_array) - 1)
        else:
            probabilities = pdf_array / np.sum(pdf_array)
            selected_bin_index = np.random.choice(len(pdf_array), p=probabilities)
        angle_per_bin = 2 * np.pi / len(pdf_array)
        phi = selected_bin_index * angle_per_bin + angle_per_bin / 2
        theta = np.pi / 2
        return phi, theta

    def update_pdf_on_collision(self, node, collided_angle_rad):
        pdf_array = node.local_direction_pdf
        num_total_bins = node._pdf_num_bins
        angle_per_bin = 2 * np.pi / num_total_bins
        collided_bin_index = int(round(collided_angle_rad / angle_per_bin)) % num_total_bins
        pdf_array[collided_bin_index] *= self.pdf_update_factor
        for i in range(1, self.pdf_influence_width_bins + 1):
            factor_neighbor = self.pdf_update_factor + (1.0 - self.pdf_update_factor) * 0.5
            left_neighbor_idx = (collided_bin_index - i + num_total_bins) % num_total_bins
            pdf_array[left_neighbor_idx] *= factor_neighbor
            right_neighbor_idx = (collided_bin_index + i) % num_total_bins
            pdf_array[right_neighbor_idx] *= factor_neighbor
        current_sum = np.sum(pdf_array)
        if current_sum > 1e-9:
            pdf_array /= current_sum
        else:
            pdf_array[:] = 1.0 / num_total_bins
        node.pdf_update_count += 1

    def positive_feedback_to_pdf(self, node, success_angle_rad):
        pdf_array = node.local_direction_pdf
        num_total_bins = node._pdf_num_bins
        angle_per_bin = 2 * np.pi / num_total_bins
        success_bin_index = int(round(success_angle_rad / angle_per_bin)) % num_total_bins
        pdf_array[success_bin_index] *= self.pdf_positive_factor
        for i in range(1, self.pdf_influence_width_bins + 1):
            factor_neighbor = 1.0 + (self.pdf_positive_factor - 1.0) * 0.5
            left_neighbor_idx = (success_bin_index - i + num_total_bins) % num_total_bins
            pdf_array[left_neighbor_idx] *= factor_neighbor
            right_neighbor_idx = (success_bin_index + i) % num_total_bins
            pdf_array[right_neighbor_idx] *= factor_neighbor
        current_sum = np.sum(pdf_array)
        if current_sum > 1e-9:
            pdf_array /= current_sum
        else:
            pdf_array[:] = 1.0 / num_total_bins
        node.pdf_update_count += 1

    def calculate_dynamic_goal_bias(self, current_iter):
        if self.max_iterations == 0:
            return self.target_max_goal_bias
        if current_iter == 0 and self.initial_goal_bias > 0:
            return self.initial_goal_bias
        effective_max_growth = self.target_max_goal_bias - self.initial_goal_bias
        if effective_max_growth <= 0:
            return self.initial_goal_bias
        growth_factor = 1 - math.exp(-self.goal_bias_rate_k * (current_iter / self.max_iterations))
        dynamic_bias = self.initial_goal_bias + effective_max_growth * growth_factor
        return np.clip(dynamic_bias, self.initial_goal_bias, self.target_max_goal_bias)

    def plan(self, visualize_progress=True):
        global collision_check_count, collision_count
        collision_check_count = 0
        collision_count = 0
        start_time = time.time()
        path_found = False
        final_goal_node = None
        final_iteration = 0

        for i in range(self.max_iterations):
            q_rand = np.array([
                random.uniform(self.env.bounds[0], self.env.bounds[1]),
                random.uniform(self.env.bounds[2], self.env.bounds[3]),
                random.uniform(self.env.bounds[4], self.env.bounds[5])
            ])
            expand_from_node = self.get_nearest_node(q_rand)

            new_state_candidate = None
            actual_direction_angle_rad = None
            is_goal_bias_expansion = False
            is_pdf_driven_expansion = False

            current_goal_bias = self.calculate_dynamic_goal_bias(i)
            if random.random() < current_goal_bias:
                is_goal_bias_expansion = True
                direction_to_goal = self.env.goal - expand_from_node.state
                dist_to_goal = np.linalg.norm(direction_to_goal)
                if dist_to_goal > 1e-6:
                    unit_direction = direction_to_goal / dist_to_goal
                    new_state_candidate = expand_from_node.state + self.step_size * unit_direction
                    if dist_to_goal <= self.step_size:
                        new_state_candidate = self.env.goal
                    actual_direction_angle_rad = math.atan2(unit_direction[1], unit_direction[0])
                    if actual_direction_angle_rad < 0:
                        actual_direction_angle_rad += 2 * np.pi
                else:
                    new_state_candidate = expand_from_node.state
            elif expand_from_node.pdf_update_count < self.pdf_maturity_threshold:
                is_pdf_driven_expansion = False
                direction_to_rand = q_rand - expand_from_node.state
                dist_to_rand = np.linalg.norm(direction_to_rand)
                if dist_to_rand > 1e-6:
                    unit_direction = direction_to_rand / dist_to_rand
                    new_state_candidate = expand_from_node.state + self.step_size * unit_direction
                    actual_direction_angle_rad = math.atan2(unit_direction[1], unit_direction[0])
                    if actual_direction_angle_rad < 0:
                        actual_direction_angle_rad += 2 * np.pi
                else:
                    new_state_candidate = expand_from_node.state
            else:
                is_pdf_driven_expansion = True
                phi, theta = self.sample_from_pdf(expand_from_node.local_direction_pdf)
                direction_vector = np.array([
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta)
                ])
                new_state_candidate = expand_from_node.state + self.step_size * direction_vector
                actual_direction_angle_rad = phi

            if new_state_candidate is None:
                continue

            path_is_free = self.env.is_collision_free(expand_from_node.state, new_state_candidate, self.step_size)

            current_parent_for_plot = expand_from_node.state
            current_new_for_plot = None
            collision_start_for_plot = None
            collision_end_for_plot = None

            if path_is_free:
                new_node = Node(new_state_candidate, parent=expand_from_node, pdf_num_bins=self.pdf_num_bins)
                new_node.initialize_pdf(method='inherit_parent', parent_pdf_data=expand_from_node.local_direction_pdf)
                self.nodes.append(new_node)
                current_new_for_plot = new_node.state

                if is_pdf_driven_expansion and actual_direction_angle_rad is not None:
                    self.positive_feedback_to_pdf(expand_from_node, actual_direction_angle_rad)

                dist_to_goal = self.get_distance(new_node.state, self.env.goal)
                if dist_to_goal <= self.goal_reach_threshold:
                    if self.env.is_collision_free(new_node.state, self.env.goal, self.step_size):
                        final_goal_node = Node(self.env.goal, parent=new_node, pdf_num_bins=self.pdf_num_bins)
                        path_found = True
                        final_iteration = i + 1
                        print(f"PMF-RRT DynGB (Iter: {i+1}): Goal region reached and final connection successful!")
                    else:
                        print(f"PMF-RRT DynGB (Iter: {i+1}): Near goal, but final connection to exact goal failed.")

            else:
                if actual_direction_angle_rad is not None:
                    self.update_pdf_on_collision(expand_from_node, actual_direction_angle_rad)
                collision_start_for_plot = expand_from_node.state
                collision_end_for_plot = new_state_candidate
                if expand_from_node.pdf_update_count > 2 * self.pdf_maturity_threshold + 10:
                    print(f"Resetting PDF for node at {expand_from_node.state} due to high collisions.")
                    expand_from_node.initialize_pdf(method='uniform')

            if visualize_progress and (i % 100 == 0 or path_found or (not path_is_free and i % 10 == 0)):
                title_str = f"PMF-RRT DynGB (Iter: {i+1}, GB:{current_goal_bias:.2f})"
                if not path_is_free:
                    title_str += " Col"
                if is_goal_bias_expansion:
                    title_str += " GBexp"
                elif not is_pdf_driven_expansion:
                    title_str += " ExtRand"
                else:
                    title_str += " PDFdrv"
                self.plot_environment(i + 1,
                                     final_path=(self.reconstruct_path(final_goal_node) if path_found else None),
                                     current_new_node_state=current_new_for_plot,
                                     current_parent_state=current_parent_for_plot,
                                     collision_attempt_start=collision_start_for_plot,
                                     collision_attempt_end=collision_end_for_plot)

            if path_found:
                break

        end_time = time.time()
        elapsed_time = end_time - start_time

        if path_found and final_goal_node:
            if final_goal_node.parent in self.nodes and final_goal_node not in self.nodes:
                self.nodes.append(final_goal_node)
            path = self.reconstruct_path(final_goal_node)
            path_length = sum(self.get_distance(path[i], path[i+1]) for i in range(len(path)-1))
            if visualize_progress:
                self.plot_environment(final_iteration, final_path=path)
            print(f"找到路径的迭代次数: {final_iteration}")
            print(f"找到路径耗费的时间: {elapsed_time:.2f} 秒")
            print(f"路径长度: {path_length:.2f} 单位")
            print(f"总碰撞检测次数: {collision_check_count}")
            print(f"总碰撞次数: {collision_count}")
            print(f"PMF-RRT DynGB: Path found with {len(path)} points.")
            return path, self.all_spheres, collision_check_count, collision_count
        else:
            print("PMF-RRT DynGB: Failed to find a path after max iterations.")
            print(f"找到路径的迭代次数: {self.max_iterations}")
            print(f"找到路径耗费的时间: {elapsed_time:.2f} 秒")
            print(f"总碰撞检测次数: {collision_check_count}")
            print(f"总碰撞次数: {collision_count}")
            print(f"PMF-RRT DynGB: Path found with 0 points.")
            if visualize_progress:
                self.plot_environment(self.max_iterations, final_path=None)
            return None, self.all_spheres, collision_check_count, collision_count

    def reconstruct_path(self, goal_node):
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        return path[::-1]

    def plot_environment(self, iteration, final_path=None, current_new_node_state=None,
                        current_parent_state=None, collision_attempt_start=None, collision_attempt_end=None):
        fig = plt.gcf()
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')

        for obs in self.env.obstacles:
            center, radius = obs[:3], obs[3]
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='gray', alpha=0.3)

        for node in self.nodes:
            if node.parent:
                ax.plot([node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]],
                        [node.state[2], node.parent.state[2]],
                        color='skyblue', linestyle='-', linewidth=0.7, alpha=0.6)

        if current_new_node_state is not None and current_parent_state is not None:
            ax.plot([current_new_node_state[0], current_parent_state[0]],
                    [current_new_node_state[1], current_parent_state[1]],
                    [current_new_node_state[2], current_parent_state[2]],
                    color='blue', linestyle='-', linewidth=1.2, label='Current Extension')

        if collision_attempt_start is not None and collision_attempt_end is not None:
            ax.plot([collision_attempt_start[0], collision_attempt_end[0]],
                    [collision_attempt_start[1], collision_attempt_end[1]],
                    [collision_attempt_start[2], collision_attempt_end[2]],
                    color='orangered', linestyle='--', linewidth=1.0, label='Collision Attempt')
            ax.scatter(collision_attempt_start[0], collision_attempt_start[1], collision_attempt_start[2],
                       s=20, color='yellow', marker='o')

        ax.scatter([self.env.start[0]], [self.env.start[1]], [self.env.start[2]],
                   color='red', s=100, label='Start')
        ax.scatter([self.env.goal[0]], [self.env.goal[1]], [self.env.goal[2]],
                   color='blue', s=100, label='Goal')

        if final_path:
            path_states = np.array(final_path)
            ax.plot(path_states[:, 0], path_states[:, 1], path_states[:, 2],
                    color='green', linestyle='-', linewidth=2.5, label='Final Path')

        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[2], self.env.bounds[3])
        ax.set_zlim(self.env.bounds[4], self.env.bounds[5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'PMF-RRT DynGB (Iteration: {iteration})')
        ax.legend()
        plt.pause(0.001)

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure(figsize=(10, 8))

    start = [10, 0, 0]
    goal = [0, 10, 10]
    obstacles = [
        (2.3, 2.3, 2.3, 1.3),
        (5, 5, 5, 1.3),
        (7.5, 7, 7, 1.3),
        (7.0, 3, 2.7, 1.8),
        (2.3, 7.7, 7.7, 2.3)
    ]
    bounds = [0, 10, 0, 10, 0, 10]

    env = Environment(start, goal, obstacles, bounds)
    planner = PMFRRTPlanner(
        env,
        step_size=1.2,
        max_iterations=1500,
        goal_reach_threshold=1.5,
        pdf_num_bins=6,
        pdf_update_factor=0.45,
        pdf_influence_width_bins=6,
        pdf_maturity_threshold=30,
        pdf_positive_factor=1.18,
        initial_goal_bias=0.01,
        target_max_goal_bias=0.5,
        goal_bias_rate_k=15.0
    )
    path, spheres, collision_checks, collision_count = planner.plan(visualize_progress=True)

    plt.ioff()
    plt.show()
