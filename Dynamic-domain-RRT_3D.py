import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time
import uuid

collision_check_count = 0
collision_count = 0

GOAL_BIAS = 0.00
DD_INITIAL_DOMAIN_RADIUS = 20.0
DD_MIN_DOMAIN_RADIUS = 1.15
DD_SHRINK_FACTOR = 0.50
DD_SUCCESS_RECOVERY_FACTOR = 1.03


class Environment:
    def __init__(self, start, goal, obstacles, bounds):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
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
    def __init__(self, state, parent=None, domain_radius=DD_INITIAL_DOMAIN_RADIUS):
        self.state = np.array(state)  # (x, y, z)
        self.parent = parent
        self.domain_radius = domain_radius
        self.cost = 0.0
        if parent:
            self.cost = parent.cost + np.linalg.norm(self.state - parent.state)


class DynamicDomainRRTPlanner:
    def __init__(self, env, step_size=0.9, max_iterations=5000, goal_reach_threshold=1.5):
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_reach_threshold = goal_reach_threshold
        self.nodes = [Node(env.start)]
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

    def get_nearest_node_in_dynamic_domain(self, target_state):
        """
        Dynamic-domain RRT的节点选择：
        先找到全局最近节点；只有当采样点位于该节点动态域内时才允许扩展。
        若不在动态域内，则本轮采样被拒绝。
        """
        nearest_node = self.get_nearest_node(target_state)
        if nearest_node is None:
            return None
        if self.get_distance(nearest_node.state, target_state) <= nearest_node.domain_radius:
            return nearest_node
        return None

    def steer(self, from_state, to_state):
        direction = np.array(to_state) - np.array(from_state)
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return np.array(to_state)
        else:
            return np.array(from_state) + (direction / dist) * self.step_size

    def sample_uniform_state(self):
        return np.array([
            random.uniform(self.env.bounds[0], self.env.bounds[1]),
            random.uniform(self.env.bounds[2], self.env.bounds[3]),
            random.uniform(self.env.bounds[4], self.env.bounds[5])
        ])

    def plan(self, visualize_progress=True):
        global collision_check_count, collision_count
        collision_check_count = 0
        collision_count = 0
        start_time = time.time()
        path_found = False
        final_goal_node = None
        final_iteration = 0

        for i in range(self.max_iterations):
            rand_val = random.random()

            if rand_val < GOAL_BIAS:
                q_rand = self.env.goal
            else:
                q_rand = self.sample_uniform_state()

            expand_from_node = self.get_nearest_node_in_dynamic_domain(q_rand)

            if expand_from_node is None:
                continue

            new_state_candidate = self.steer(expand_from_node.state, q_rand)

            path_is_free = self.env.is_collision_free(expand_from_node.state, new_state_candidate, self.step_size)

            current_parent_for_plot = expand_from_node.state
            current_new_for_plot = None
            collision_start_for_plot = None
            collision_end_for_plot = None

            if path_is_free:
                expand_from_node.domain_radius = min(
                    DD_INITIAL_DOMAIN_RADIUS,
                    expand_from_node.domain_radius * DD_SUCCESS_RECOVERY_FACTOR
                )
                new_node = Node(new_state_candidate, parent=expand_from_node)
                self.nodes.append(new_node)
                current_new_for_plot = new_node.state

                dist_to_goal = self.get_distance(new_node.state, self.env.goal)
                if dist_to_goal <= self.goal_reach_threshold:
                    if self.env.is_collision_free(new_node.state, self.env.goal, self.step_size):
                        final_goal_node = Node(self.env.goal, parent=new_node)
                        path_found = True
                        final_iteration = i + 1
                        print(f"Dynamic-domain RRT (Iter: {i + 1}): Goal region reached and final connection successful!")
                    else:
                        print(f"Dynamic-domain RRT (Iter: {i + 1}): Near goal, but final connection to exact goal failed.")
            else:
                expand_from_node.domain_radius = max(
                    DD_MIN_DOMAIN_RADIUS,
                    expand_from_node.domain_radius * DD_SHRINK_FACTOR
                )
                collision_start_for_plot = expand_from_node.state
                collision_end_for_plot = new_state_candidate

            if visualize_progress and (i % 100 == 0 or path_found or (not path_is_free and i % 10 == 0)):
                title_str = f"Dynamic-domain RRT (Iter: {i + 1})"
                if not path_is_free:
                    title_str += " Col"
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
            path_length = sum(self.get_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
            if visualize_progress:
                self.plot_environment(final_iteration, final_path=path)
            print(f"找到路径的迭代次数: {final_iteration}")
            print(f"找到路径耗费的时间: {elapsed_time:.2f} 秒")
            print(f"路径长度: {path_length:.2f} 单位")
            print(f"总碰撞检测次数: {collision_check_count}")
            print(f"总碰撞次数: {collision_count}")
            print(f"Dynamic-domain RRT: Path found with {len(path)} points.")
            return path, self.all_spheres, path_length, collision_check_count, collision_count
        else:
            print("Dynamic-domain RRT: Failed to find a path after max iterations.")
            print(f"找到路径的迭代次数: {self.max_iterations}")
            print(f"找到路径耗费的时间: {elapsed_time:.2f} 秒")
            print(f"路径长度: 0.00 单位")
            print(f"总碰撞检测次数: {collision_check_count}")
            print(f"总碰撞次数: {collision_count}")
            print(f"Dynamic-domain RRT: Path found with 0 points.")
            if visualize_progress:
                self.plot_environment(self.max_iterations, final_path=None)
            return None, self.all_spheres, 0.0, collision_check_count, collision_count

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

            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(0, np.pi, 25)

            for phi in u:
                x = center[0] + radius * np.sin(v) * np.cos(phi)
                y = center[1] + radius * np.sin(v) * np.sin(phi)
                z = center[2] + radius * np.cos(v)
                ax.plot(x, y, z, color='gray', alpha=0.5, linewidth=0.5)

            for theta in v:
                x = center[0] + radius * np.sin(theta) * np.cos(u)
                y = center[1] + radius * np.sin(theta) * np.sin(u)
                z = center[2] + radius * np.cos(theta) * np.ones_like(u)
                ax.plot(x, y, z, color='gray', alpha=0.5, linewidth=0.5)

        for node in self.nodes:
            if node.parent:
                ax.plot([node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]],
                        [node.state[2], node.parent.state[2]],
                        color='blue', linestyle='-', linewidth=1.5, alpha=0.7)

        if current_new_node_state is not None and current_parent_state is not None:
            ax.plot([current_new_node_state[0], current_parent_state[0]],
                    [current_new_node_state[1], current_parent_state[1]],
                    [current_new_node_state[2], current_parent_state[2]],
                    color='blue', linestyle='-', linewidth=2.0, label='Current Extension')

        if collision_attempt_start is not None and collision_attempt_end is not None:
            ax.plot([collision_attempt_start[0], collision_attempt_end[0]],
                    [collision_attempt_start[1], collision_attempt_end[1]],
                    [collision_attempt_start[2], collision_attempt_end[2]],
                    color='orangered', linestyle='--', linewidth=1.2, label='Collision Attempt')
            ax.scatter(collision_attempt_start[0], collision_attempt_start[1], collision_attempt_start[2],
                       s=20, color='yellow', marker='o')

        ax.scatter([self.env.start[0]], [self.env.start[1]], [self.env.start[2]],
                   color='red', s=100, label='Start', marker='o')
        ax.scatter([self.env.goal[0]], [self.env.goal[1]], [self.env.goal[2]],
                   color='green', s=100, label='Goal', marker='*')

        if final_path:
            path_states = np.array(final_path)
            ax.plot(path_states[:, 0], path_states[:, 1], path_states[:, 2],
                    color='red', linestyle='-', linewidth=3.0, label='Final Path')

        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[2], self.env.bounds[3])
        ax.set_zlim(self.env.bounds[4], self.env.bounds[5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Dynamic-domain RRT')
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
    planner = DynamicDomainRRTPlanner(
        env,
        step_size=1.0,
        max_iterations=2500,
        goal_reach_threshold=1.5
    )
    path, spheres, path_length, collision_checks, collision_count = planner.plan(visualize_progress=True)

    plt.ioff()
    plt.show()
