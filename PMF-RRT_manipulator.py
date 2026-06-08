import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973]
])
NUM_JOINTS = 7

MAX_ITER = 10000
STEP_SIZE = 0.2
GOAL_TOLERANCE = 0.2
INITIAL_GOAL_BIAS = 0.15
TARGET_MAX_GOAL_BIAS_A = 0.2
GOAL_BIAS_RATE_K = 39.0

OBSTACLE_POSITION = [0.5, 0.0, 5.5]
OBSTACLE_RADIUS = 0.25
RECT_OBSTACLE_POS = [0.7, -0.1, 0.45]
RECT_OBSTACLE_DIM = [0.7, 0.1, 0.9]

PLAYBACK_SPEED = 0.1

PDF_NUM_BINS = 36
PDF_UPDATE_FACTOR = 0.7
PDF_INFLUENCE_WIDTH_BINS = 1
PDF_POSITIVE_FACTOR = 1.2
PDF_MATURITY_THRESHOLD = 3

def calculate_dynamic_goal_bias(current_iter: int, max_iter: int, initial_bias: float,
                               target_max_bias: float, rate_k: float) -> float:
    if max_iter == 0:
        return target_max_bias
    if current_iter == 0 and initial_bias > 0:
        return initial_bias
    effective_max_growth = target_max_bias - initial_bias
    if effective_max_growth <= 0:
        return initial_bias
    growth_factor = 1 - math.exp(-rate_k * (current_iter / max_iter))
    dynamic_bias = initial_bias + effective_max_growth * growth_factor
    return np.clip(dynamic_bias, initial_bias, target_max_bias)

class PMFRRTNode:
    def __init__(self, config: np.ndarray, parent: Optional['PMFRRTNode'] = None,
                 pdf_num_bins: int = PDF_NUM_BINS):
        self.config = config
        self.parent = parent
        self._pdf_num_bins = pdf_num_bins
        self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        self.pdf_update_count = 0

    def initialize_pdf(self, method: str = 'uniform', parent_pdf_data: Optional[np.ndarray] = None):
        if method == 'uniform' or parent_pdf_data is None:
            self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        elif method == 'inherit_parent' and parent_pdf_data is not None:
            self.local_direction_pdf = 0.7 * np.copy(parent_pdf_data) + 0.3 * (
                np.ones(self._pdf_num_bins) / self._pdf_num_bins)
            current_sum = np.sum(self.local_direction_pdf)
            if current_sum > 1e-9:
                self.local_direction_pdf /= current_sum
            else:
                self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        else:
            self.local_direction_pdf = np.ones(self._pdf_num_bins) / self._pdf_num_bins
        self.pdf_update_count = 0

def sample_abstract_direction_bin_from_pdf(pdf_array: np.ndarray) -> int:
    if np.sum(pdf_array) < 1e-9:
        return np.random.randint(0, len(pdf_array) - 1)
    else:
        probabilities = pdf_array / np.sum(pdf_array)
        return np.random.choice(len(pdf_array), p=probabilities)

def abstract_bin_to_joint_space_direction(selected_bin_index: int, num_bins: int,
                                         num_joints: int) -> np.ndarray:
    direction_vector = np.zeros(num_joints)
    angle = (selected_bin_index / num_bins) * 2 * np.pi
    if num_joints >= 1: direction_vector[0] = math.cos(angle)
    if num_joints >= 2: direction_vector[1] = math.sin(angle)
    if num_joints >= 3: direction_vector[2] = math.cos(angle + np.pi / 4)
    for i in range(3, num_joints):
        direction_vector[i] = np.random.uniform(-0.3, 0.3)
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm if norm > 1e-6 else np.zeros(num_joints)

def get_closest_abstract_direction_bin(actual_joint_direction: np.ndarray, num_bins: int,
                                      num_joints: int) -> int:
    if np.linalg.norm(actual_joint_direction) < 1e-6:
        return np.random.randint(0, num_bins - 1)
    dx, dy = actual_joint_direction[0], actual_joint_direction[1]
    angle = math.atan2(dy, dx)
    if angle < 0: angle += 2 * np.pi
    angle_per_bin = 2 * np.pi / num_bins
    return int(round(angle / angle_per_bin)) % num_bins

def update_pdf(node_to_update: PMFRRTNode, direction_bin_index: int, factor: float,
               is_positive_feedback: bool):
    pdf_array = node_to_update.local_direction_pdf
    pdf_array[direction_bin_index] *= factor
    current_sum = np.sum(pdf_array)
    if current_sum > 1e-9:
        pdf_array /= current_sum
    else:
        pdf_array[:] = 1.0 / len(pdf_array)
    node_to_update.pdf_update_count += 1

class PMFRRTPlanner:
    def __init__(self, robot_id: int, max_iter: int = MAX_ITER, step_size: float = STEP_SIZE,
                 initial_goal_bias: float = INITIAL_GOAL_BIAS,
                 target_max_goal_bias: float = TARGET_MAX_GOAL_BIAS_A,
                 goal_bias_rate_k: float = GOAL_BIAS_RATE_K,
                 goal_tolerance: float = GOAL_TOLERANCE):
        self.robot_id = robot_id
        self.max_iter = max_iter
        self.step_size = step_size
        self.initial_goal_bias = initial_goal_bias
        self.target_max_goal_bias = target_max_goal_bias
        self.goal_bias_rate_k = goal_bias_rate_k
        self.goal_tolerance = goal_tolerance
        self.tree: List[PMFRRTNode] = []
        self.collision_checks = 0
        self.collision_count = 0
        self.iterations = 0

    def get_random_config(self, q_goal: np.ndarray) -> np.ndarray:
        config = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            config[i] = np.random.uniform(JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
        return config

    def nearest_node(self, q_rand: np.ndarray) -> PMFRRTNode:
        distances = [np.linalg.norm(node.config - q_rand) for node in self.tree]
        return self.tree[np.argmin(distances)]

    def steer_in_direction(self, q_near: np.ndarray, unit_direction_vec: np.ndarray) -> np.ndarray:
        return q_near + self.step_size * unit_direction_vec

    def is_collision_free_path(self, q_near: np.ndarray, q_new: np.ndarray) -> bool:
        num_steps = int(np.linalg.norm(q_new - q_near) / (self.step_size / 5.0)) + 2
        for t in np.linspace(0, 1, num_steps):
            q_intermediate = (1 - t) * q_near + t * q_new
            for i in range(NUM_JOINTS):
                if not (JOINT_LIMITS[i][0] <= q_intermediate[i] <= JOINT_LIMITS[i][1]):
                    return False
                p.resetJointState(self.robot_id, i, q_intermediate[i])
            self.collision_checks += 1
            p.performCollisionDetection()
            if p.getContactPoints(bodyA=self.robot_id):
                self.collision_count += 1
                return False
        return True

    def is_goal_reached(self, q_new: np.ndarray, q_goal: np.ndarray) -> bool:
        return np.linalg.norm(q_new - q_goal) < self.goal_tolerance

    def plan(self, q_init: np.ndarray, q_goal: np.ndarray) -> List[np.ndarray]:
        self.tree = [PMFRRTNode(q_init)]
        for self.iterations in range(self.max_iter):
            current_goal_bias = calculate_dynamic_goal_bias(
                self.iterations, self.max_iter, self.initial_goal_bias,
                self.target_max_goal_bias, self.goal_bias_rate_k
            )
            expand_from_node = self.nearest_node(self.get_random_config(q_goal))

            new_config_candidate = None
            actual_direction_bin_for_update = None
            is_goal_bias_expansion = False
            is_pdf_driven_expansion = False

            if np.random.rand() < current_goal_bias:
                is_goal_bias_expansion = True
                direction_to_goal_vec = q_goal - expand_from_node.config
                dist_to_goal = np.linalg.norm(direction_to_goal_vec)
                if dist_to_goal > 1e-6:
                    unit_dir_to_goal = direction_to_goal_vec / dist_to_goal
                    new_config_candidate = expand_from_node.config + unit_dir_to_goal * self.step_size
                    if dist_to_goal <= self.step_size: new_config_candidate = q_goal
                    actual_direction_bin_for_update = get_closest_abstract_direction_bin(
                        unit_dir_to_goal, PDF_NUM_BINS, NUM_JOINTS
                    )
                else:
                    new_config_candidate = expand_from_node.config
            elif expand_from_node.pdf_update_count < PDF_MATURITY_THRESHOLD:
                is_pdf_driven_expansion = False
                q_rand_external = self.get_random_config(q_goal)
                direction_to_external_vec = q_rand_external - expand_from_node.config
                dist_to_external = np.linalg.norm(direction_to_external_vec)
                if dist_to_external > 1e-6:
                    unit_dir_to_external = direction_to_external_vec / dist_to_external
                    new_config_candidate = expand_from_node.config + unit_dir_to_external * self.step_size
                    actual_direction_bin_for_update = get_closest_abstract_direction_bin(
                        unit_dir_to_external, PDF_NUM_BINS, NUM_JOINTS
                    )
                else:
                    new_config_candidate = expand_from_node.config
            else:
                is_pdf_driven_expansion = True
                selected_abstract_bin = sample_abstract_direction_bin_from_pdf(
                    expand_from_node.local_direction_pdf
                )
                actual_direction_bin_for_update = selected_abstract_bin
                joint_space_dir_vec = abstract_bin_to_joint_space_direction(
                    selected_abstract_bin, PDF_NUM_BINS, NUM_JOINTS
                )
                if np.linalg.norm(joint_space_dir_vec) < 1e-6:
                    new_config_candidate = expand_from_node.config
                else:
                    new_config_candidate = expand_from_node.config + joint_space_dir_vec * self.step_size

            if new_config_candidate is None:
                continue
            new_config_candidate = np.clip(new_config_candidate, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

            if self.is_collision_free_path(expand_from_node.config, new_config_candidate):
                new_node = PMFRRTNode(new_config_candidate, parent=expand_from_node)
                new_node.initialize_pdf(method='inherit_parent',
                                       parent_pdf_data=expand_from_node.local_direction_pdf)
                self.tree.append(new_node)

                if is_pdf_driven_expansion and actual_direction_bin_for_update is not None:
                    update_pdf(expand_from_node, actual_direction_bin_for_update, PDF_POSITIVE_FACTOR, True)

                if self.is_goal_reached(new_node.config, q_goal):
                    if self.is_collision_free_path(new_node.config, q_goal):
                        final_node = PMFRRTNode(q_goal, parent=new_node)
                        self.tree.append(final_node)
                        path = []
                        curr = final_node
                        while curr is not None:
                            path.append(curr.config)
                            curr = curr.parent
                        print(f"PMF-RRT (Panda): Goal reached at iter {self.iterations + 1} via direct connect!")
                        return path[::-1]
                    else:
                        print(f"PMF-RRT (Panda): Near goal at iter {self.iterations + 1}, but direct connection blocked.")
            else:
                if actual_direction_bin_for_update is not None:
                    update_pdf(expand_from_node, actual_direction_bin_for_update, PDF_UPDATE_FACTOR, False)
                if expand_from_node.pdf_update_count > 2 * PDF_MATURITY_THRESHOLD + 10:
                    print(f"Resetting PDF for node at {expand_from_node.config} due to high collisions.")
                    expand_from_node.initialize_pdf(method='uniform')
            if self.iterations % 500 == 0:
                print(f"Iteration: {self.iterations}, Tree size: {len(self.tree)}")
        print("PMF-RRT (Panda): Failed to find a path after max iterations.")
        return []

def get_end_effector_position(robot_id: int) -> np.ndarray:
    state = p.getLinkState(robot_id, 11)
    return np.array(state[0])

def setup_pybullet() -> Tuple[int, int, int]:
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    initial_joint_positions = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]
    for i in range(NUM_JOINTS):
        p.resetJointState(robot_id, i, initial_joint_positions[i])

    obstacle_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=OBSTACLE_RADIUS,
                                               rgbaColor=[1, 0, 0, 0.5])
    obstacle_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=OBSTACLE_RADIUS)
    obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_collision_shape,
                                    baseVisualShapeIndex=obstacle_visual_shape,
                                    basePosition=OBSTACLE_POSITION)

    rect_obstacle_visual = p.createVisualShape(p.GEOM_BOX,
                                              halfExtents=[x / 2 for x in RECT_OBSTACLE_DIM],
                                              rgbaColor=[0, 0, 1, 0.5])
    rect_obstacle_collision = p.createCollisionShape(p.GEOM_BOX,
                                                    halfExtents=[x / 2 for x in RECT_OBSTACLE_DIM])
    rect_obstacle_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=rect_obstacle_collision,
                                        baseVisualShapeIndex=rect_obstacle_visual,
                                        basePosition=RECT_OBSTACLE_POS)

    return robot_id, obstacle_id, rect_obstacle_id

def plot_joint_angles(path: List[np.ndarray], playback_speed: float,
                     filename="joint_angles_pmfrrt_raw.png"):
    if not path:
        return
    num_points = len(path)
    times = np.arange(num_points) * playback_speed
    joint_angles_array = np.array(path)
    plt.figure(figsize=(12, 7))
    for i in range(NUM_JOINTS):
        plt.plot(times, joint_angles_array[:, i], label=f'关节 {i + 1}')
    plt.title('PMF-RRT 关节角度随时间变化 (原始路径)')
    plt.xlabel('模拟时间 (秒)')
    plt.ylabel('角度 (弧度)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"关节角度曲线已保存为 '{filename}'")

def plot_trajectory(trajectory: List[np.ndarray], filename="end_effector_trajectory.png"):
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', label='末端执行器轨迹')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('末端执行器轨迹')
    ax.legend()
    plt.savefig(filename)
    plt.close()
    print(f"末端执行器轨迹图已保存为 '{filename}'")

def calculate_trajectory_length(trajectory: List[np.ndarray]) -> float:
    total_length = 0.0
    for i in range(1, len(trajectory)):
        total_length += np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
    return total_length

def main():
    robot_id, _, rect_obstacle_id = setup_pybullet()
    q_init = np.array([0.0, 0.0, -1.1, -1.8, 0.0, 2.0, 1.5])
    q_goal = np.array([0.0, 0.0, 0.9, -1.8, 0.0, 2.0, 1.5])
    for i in range(NUM_JOINTS):
        q_goal[i] = np.clip(q_goal[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])

    planner = PMFRRTPlanner(
        robot_id=robot_id,
        max_iter=MAX_ITER,
        step_size=STEP_SIZE,
        initial_goal_bias=INITIAL_GOAL_BIAS,
        target_max_goal_bias=TARGET_MAX_GOAL_BIAS_A,
        goal_bias_rate_k=GOAL_BIAS_RATE_K,
        goal_tolerance=GOAL_TOLERANCE
    )
    print("开始 PMF-RRT 规划 (Panda - 无平滑, 动态目标偏置)...")
    start_time = time.time()
    raw_path = planner.plan(q_init, q_goal)
    planning_time = time.time() - start_time

    if raw_path:
        print(f"找到路径！耗时：{planning_time:.2f}秒, 路径节点数: {len(raw_path)}")
        print(f"总迭代次数: {planner.iterations + 1}")
        print(f"总碰撞检测次数: {planner.collision_checks}")
        print(f"总碰撞次数: {planner.collision_count}")
        plot_joint_angles(raw_path, PLAYBACK_SPEED)

        print("开始路径回放并记录末端执行器轨迹...")
        trajectory = []
        for i_joint in range(NUM_JOINTS):
            p.resetJointState(robot_id, i_joint, q_init[i_joint])
        trajectory.append(get_end_effector_position(robot_id))
        time.sleep(1)

        for config in raw_path:
            for i_joint in range(NUM_JOINTS):
                p.resetJointState(robot_id, i_joint, config[i_joint])
            trajectory.append(get_end_effector_position(robot_id))
            p.stepSimulation()
            p.addUserDebugLine(trajectory[-2], trajectory[-1], [1, 0, 0], 2.0)
            time.sleep(PLAYBACK_SPEED)
        print("路径回放结束。")

        trajectory_length = calculate_trajectory_length(trajectory)
        print(f"末端执行器轨迹总长度: {trajectory_length:.3f} 米")
        plot_trajectory(trajectory)
    else:
        print(f"未找到路径！耗时：{planning_time:.2f}秒")
        print(f"总迭代次数: {planner.iterations}")
        print(f"总碰撞检测次数: {planner.collision_checks}")
        print(f"总碰撞次数: {planner.collision_count}")

    input("按 Enter 键退出...")
    if p.isConnected():
        p.disconnect()

if __name__ == "__main__":
    main()
