import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time
import uuid

collision_check_count = 0
collision_count = 0

GVP_INFLUENCE = 0.3
GOAL_BIAS = 0.00
GVP_GRID_RESOLUTION = 10
GVP_GOAL_WEIGHT = 1.8
GVP_CLEARANCE_WEIGHT = 0.7
GVP_COLLISION_DECAY = 0.55
GVP_SUCCESS_GAIN = 1.05
GVP_MIN_WEIGHT = 1e-6


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
    def __init__(self, state, parent=None):
        self.state = np.array(state)  # (x, y, z)
        self.parent = parent
        self.cost = 0.0
        if parent:
            self.cost = parent.cost + np.linalg.norm(self.state - parent.state)


def point_in_free_space_without_count(point, obstacles, bounds):
    """不改变碰撞计数器的自由空间检查，用于初始化GVP网格权重。"""
    point = np.array(point)
    for obs in obstacles:
        center, radius = np.array(obs[:3]), obs[3]
        if np.linalg.norm(point - center) < radius:
            return False
    return (bounds[0] <= point[0] <= bounds[1] and
            bounds[2] <= point[1] <= bounds[3] and
            bounds[4] <= point[2] <= bounds[5])


def compute_clearance_without_count(point, obstacles):
    """计算点到最近球形障碍物表面的距离，不改变碰撞计数器。"""
    point = np.array(point)
    if not obstacles:
        return float('inf')
    clearance_values = []
    for obs in obstacles:
        center, radius = np.array(obs[:3]), obs[3]
        clearance_values.append(np.linalg.norm(point - center) - radius)
    return min(clearance_values)


def normalize_weights(weights):
    total = np.sum(weights)
    if total <= 1e-12:
        return np.ones_like(weights) / len(weights)
    return weights / total


class GVPRRTPlanner:
    def __init__(self, env, step_size=0.9, max_iterations=5000, goal_reach_threshold=1.5):
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_reach_threshold = goal_reach_threshold
        self.nodes = [Node(env.start)]
        self.all_spheres = []

        self.grid_centers = []
        self.grid_weights = []
        self.grid_cell_size = np.array([
            (env.bounds[1] - env.bounds[0]) / GVP_GRID_RESOLUTION,
            (env.bounds[3] - env.bounds[2]) / GVP_GRID_RESOLUTION,
            (env.bounds[5] - env.bounds[4]) / GVP_GRID_RESOLUTION
        ])
        self.initialize_gvp_grid()

    def initialize_gvp_grid(self):
        """初始化三维网格变量概率分布。"""
        centers = []
        raw_weights = []

        xmin, xmax, ymin, ymax, zmin, zmax = self.env.bounds
        xs = np.linspace(xmin + self.grid_cell_size[0] / 2, xmax - self.grid_cell_size[0] / 2, GVP_GRID_RESOLUTION)
        ys = np.linspace(ymin + self.grid_cell_size[1] / 2, ymax - self.grid_cell_size[1] / 2, GVP_GRID_RESOLUTION)
        zs = np.linspace(zmin + self.grid_cell_size[2] / 2, zmax - self.grid_cell_size[2] / 2, GVP_GRID_RESOLUTION)

        max_goal_dist = np.linalg.norm(np.array([xmax - xmin, ymax - ymin, zmax - zmin]))
        for x in xs:
            for y in ys:
                for z in zs:
                    center = np.array([x, y, z])
                    centers.append(center)
                    if not point_in_free_space_without_count(center, self.env.obstacles, self.env.bounds):
                        raw_weights.append(GVP_MIN_WEIGHT)
                        continue

                    dist_to_goal = np.linalg.norm(center - self.env.goal)
                    goal_score = 1.0 - min(dist_to_goal / max_goal_dist, 1.0)

                    clearance = compute_clearance_without_count(center, self.env.obstacles)
                    clearance_score = min(max(clearance / (2.0 * self.step_size), 0.0), 1.0)

                    weight = GVP_MIN_WEIGHT + GVP_GOAL_WEIGHT * goal_score + GVP_CLEARANCE_WEIGHT * clearance_score
                    raw_weights.append(weight)

        self.grid_centers = np.array(centers)
        self.grid_weights = normalize_weights(np.array(raw_weights, dtype=float))

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

    def get_grid_index_from_state(self, state):
        state = np.array(state)
        ix = int((state[0] - self.env.bounds[0]) / self.grid_cell_size[0])
        iy = int((state[1] - self.env.bounds[2]) / self.grid_cell_size[1])
        iz = int((state[2] - self.env.bounds[4]) / self.grid_cell_size[2])
        ix = min(max(ix, 0), GVP_GRID_RESOLUTION - 1)
        iy = min(max(iy, 0), GVP_GRID_RESOLUTION - 1)
        iz = min(max(iz, 0), GVP_GRID_RESOLUTION - 1)
        return ix * GVP_GRID_RESOLUTION * GVP_GRID_RESOLUTION + iy * GVP_GRID_RESOLUTION + iz

    def sample_from_gvp_grid(self):
        """按照三维变量概率网格采样一个随机状态。"""
        weights = normalize_weights(self.grid_weights)
        selected_index = np.random.choice(len(self.grid_centers), p=weights)
        center = self.grid_centers[selected_index]
        jitter = np.array([
            random.uniform(-0.5, 0.5) * self.grid_cell_size[0],
            random.uniform(-0.5, 0.5) * self.grid_cell_size[1],
            random.uniform(-0.5, 0.5) * self.grid_cell_size[2]
        ])
        q_rand = center + jitter
        q_rand[0] = np.clip(q_rand[0], self.env.bounds[0], self.env.bounds[1])
        q_rand[1] = np.clip(q_rand[1], self.env.bounds[2], self.env.bounds[3])
        q_rand[2] = np.clip(q_rand[2], self.env.bounds[4], self.env.bounds[5])
        return q_rand

    def update_gvp_weight(self, state, success=True):
        """根据扩展反馈更新对应网格的变量概率。"""
        idx = self.get_grid_index_from_state(state)
        if success:
            self.grid_weights[idx] *= GVP_SUCCESS_GAIN
        else:
            self.grid_weights[idx] *= GVP_COLLISION_DECAY
        self.grid_weights[idx] = max(self.grid_weights[idx], GVP_MIN_WEIGHT)
        self.grid_weights = normalize_weights(self.grid_weights)

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
            elif rand_val < GOAL_BIAS + GVP_INFLUENCE:
                q_rand = self.sample_from_gvp_grid()
            else:
                q_rand = self.sample_uniform_state()

            expand_from_node = self.get_nearest_node(q_rand)
            new_state_candidate = self.steer(expand_from_node.state, q_rand)

            path_is_free = self.env.is_collision_free(expand_from_node.state, new_state_candidate, self.step_size)

            current_parent_for_plot = expand_from_node.state
            current_new_for_plot = None
            collision_start_for_plot = None
            collision_end_for_plot = None

            if path_is_free:
                new_node = Node(new_state_candidate, parent=expand_from_node)
                self.nodes.append(new_node)
                current_new_for_plot = new_node.state
                self.update_gvp_weight(new_node.state, success=True)

                dist_to_goal = self.get_distance(new_node.state, self.env.goal)
                if dist_to_goal <= self.goal_reach_threshold:
                    if self.env.is_collision_free(new_node.state, self.env.goal, self.step_size):
                        final_goal_node = Node(self.env.goal, parent=new_node)
                        path_found = True
                        final_iteration = i + 1
                        print(f"GVP-RRT (Iter: {i + 1}): Goal region reached and final connection successful!")
                    else:
                        print(f"GVP-RRT (Iter: {i + 1}): Near goal, but final connection to exact goal failed.")
            else:
                collision_start_for_plot = expand_from_node.state
                collision_end_for_plot = new_state_candidate
                self.update_gvp_weight(q_rand, success=False)

            if visualize_progress and (i % 100 == 0 or path_found or (not path_is_free and i % 10 == 0)):
                title_str = f"GVP-RRT (Iter: {i + 1})"
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
            print(f"GVP-RRT: Path found with {len(path)} points.")
            return path, self.all_spheres, path_length, collision_check_count, collision_count
        else:
            print("GVP-RRT: Failed to find a path after max iterations.")
            print(f"找到路径的迭代次数: {self.max_iterations}")
            print(f"找到路径耗费的时间: {elapsed_time:.2f} 秒")
            print(f"路径长度: 0.00 单位")
            print(f"总碰撞检测次数: {collision_check_count}")
            print(f"总碰撞次数: {collision_count}")
            print(f"GVP-RRT: Path found with 0 points.")
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
        plt.title(f'GVP-RRT')
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
    planner = GVPRRTPlanner(
        env,
        step_size=1.0,
        max_iterations=2500,
        goal_reach_threshold=1.5
    )
    path, spheres, path_length, collision_checks, collision_count = planner.plan(visualize_progress=True)

    plt.ioff()
    plt.show()
