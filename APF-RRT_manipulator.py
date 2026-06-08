import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import matplotlib.pyplot as plt

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

KEY_LINKS = [3, 4, 5, 6, 7, 8, 9, 11]

MAX_ITER = 20000
STEP_SIZE = 0.18
GOAL_SAMPLE_RATE = 0.05
GOAL_TOLERANCE = 0.15
OBSTACLE_POSITION = [0.5, 0.0, 5.5]
OBSTACLE_RADIUS = 0.25
RECT_OBSTACLE_POS = [0.7, -0.1, 0.45]
RECT_OBSTACLE_DIM = [0.7, 0.1, 0.9]
PLAYBACK_SPEED = 0.1

K_ATT = 1.8
K_REP = 0.8
RHO_0 = 0.2
APF_INFLUENCE = 0.65


class Obstacle:
    """障碍物基类"""

    def distance_to_point(self, point):
        raise NotImplementedError

    def get_repulsion_direction(self, point):
        raise NotImplementedError


class SphereObstacle(Obstacle):
    """球形障碍物"""

    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def distance_to_point(self, point):
        point = np.array(point)
        dist_to_center = np.linalg.norm(point - self.center)
        return max(dist_to_center - self.radius, 1e-6)

    def get_repulsion_direction(self, point):
        point = np.array(point)
        direction = point - self.center
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.random.randn(3)
        return direction / norm


class BoxObstacle(Obstacle):
    """矩形（长方体）障碍物"""

    def __init__(self, center, dimensions):
        self.center = np.array(center)
        self.half_dims = np.array(dimensions) / 2.0

    def distance_to_point(self, point):
        point = np.array(point)
        local_point = point - self.center
        dx = max(abs(local_point[0]) - self.half_dims[0], 0)
        dy = max(abs(local_point[1]) - self.half_dims[1], 0)
        dz = max(abs(local_point[2]) - self.half_dims[2], 0)
        outside_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if outside_dist > 1e-6:
            return outside_dist
        inside_dists = self.half_dims - np.abs(local_point)
        return max(min(inside_dists), 1e-6)

    def get_repulsion_direction(self, point):
        point = np.array(point)
        local_point = point - self.center
        dx = max(abs(local_point[0]) - self.half_dims[0], 0)
        dy = max(abs(local_point[1]) - self.half_dims[1], 0)
        dz = max(abs(local_point[2]) - self.half_dims[2], 0)
        outside_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if outside_dist > 1e-6:
            closest_point = np.zeros(3)
            for i in range(3):
                closest_point[i] = np.clip(local_point[i], -self.half_dims[i], self.half_dims[i])
            direction = local_point - closest_point
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return np.array([0, 0, 1])
            return direction / norm
        else:
            inside_dists = self.half_dims - np.abs(local_point)
            min_axis = np.argmin(inside_dists)
            direction = np.zeros(3)
            direction[min_axis] = np.sign(local_point[min_axis])
            if np.linalg.norm(direction) < 1e-6:
                direction[min_axis] = 1.0
            return direction


class RRTNode:
    def __init__(self, config: np.ndarray, parent=None):
        self.config = config
        self.parent = parent


class APFRRTPlanner:
    def __init__(self, robot_id, max_iter, step_size, goal_bias, goal_tolerance):
        self.robot_id = robot_id
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.tree = []
        self.collision_checks = 0
        self.collision_count = 0
        self.iterations = 0
        self.obstacles = []

    def add_sphere_obstacle(self, center, radius):
        self.obstacles.append(SphereObstacle(center, radius))

    def add_box_obstacle(self, center, dimensions):
        self.obstacles.append(BoxObstacle(center, dimensions))

    def compute_attractive_force_joint(self, current_config, goal_config):
        direction = goal_config - current_config
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros(NUM_JOINTS)
        return K_ATT * direction / dist

    def compute_numerical_jacobian_for_link(self, config, link_id, link_pos):
        jacobian = np.zeros((3, NUM_JOINTS))
        delta = 0.01
        for j in range(NUM_JOINTS):
            config_plus = config.copy()
            config_plus[j] += delta
            config_plus = np.clip(config_plus, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            for i in range(NUM_JOINTS):
                p.resetJointState(self.robot_id, i, config_plus[i])
            state = p.getLinkState(self.robot_id, link_id)
            link_pos_plus = np.array(state[0])
            jacobian[:, j] = (link_pos_plus - link_pos) / delta
        return jacobian

    def compute_repulsive_force_joint(self, current_config, goal_config):
        total_force = np.zeros(NUM_JOINTS)
        if not self.obstacles:
            return total_force
        current_states = [p.getJointState(self.robot_id, i)[0] for i in range(NUM_JOINTS)]
        for i in range(NUM_JOINTS):
            p.resetJointState(self.robot_id, i, current_config[i])
        for link_id in KEY_LINKS:
            state = p.getLinkState(self.robot_id, link_id)
            link_pos = np.array(state[0])
            for obs in self.obstacles:
                dist = obs.distance_to_point(link_pos)
                if dist < RHO_0:
                    repulsion_dir = obs.get_repulsion_direction(link_pos)
                    force_magnitude = K_REP * (1.0 / dist - 1.0 / RHO_0) * (1.0 / (dist ** 2))
                    cartesian_force = force_magnitude * repulsion_dir
                    jacobian = self.compute_numerical_jacobian_for_link(current_config, link_id, link_pos)
                    joint_force = jacobian.T @ cartesian_force
                    total_force += joint_force
        for i in range(NUM_JOINTS):
            p.resetJointState(self.robot_id, i, current_states[i])
        return total_force

    def compute_apf_direction(self, current_config, goal_config):
        f_att = self.compute_attractive_force_joint(current_config, goal_config)
        f_rep = self.compute_repulsive_force_joint(current_config, goal_config)
        total_force = f_att + f_rep
        norm = np.linalg.norm(total_force)
        if norm < 1e-6:
            return None
        return total_force / norm

    def get_random_config(self, q_goal):
        rand_val = np.random.rand()
        if rand_val < self.goal_bias:
            return q_goal
        elif rand_val < self.goal_bias + APF_INFLUENCE:
            if len(self.tree) > 0:
                base_node = self.tree[np.random.randint(len(self.tree))]
                apf_direction = self.compute_apf_direction(base_node.config, q_goal)
                if apf_direction is not None:
                    noise = np.random.uniform(-0.1, 0.1, NUM_JOINTS)
                    perturbed_direction = apf_direction + noise
                    norm = np.linalg.norm(perturbed_direction)
                    if norm > 1e-6:
                        perturbed_direction = perturbed_direction / norm
                    new_config = base_node.config + perturbed_direction * self.step_size * np.random.uniform(1, 2)
                    return np.clip(new_config, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def nearest_node(self, q_rand):
        distances = [np.linalg.norm(node.config - q_rand) for node in self.tree]
        return self.tree[np.argmin(distances)]

    def steer(self, q_near, q_rand):
        direction = q_rand - q_near
        norm = np.linalg.norm(direction)
        if norm == 0:
            return q_near
        step = self.step_size * direction / norm
        return np.clip(q_near + step, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def is_collision_free_path(self, q_near, q_new):
        steps = int(np.linalg.norm(q_new - q_near) / (self.step_size / 5.0)) + 2
        for t in np.linspace(0, 1, steps):
            q_interp = (1 - t) * q_near + t * q_new
            for i in range(NUM_JOINTS):
                if not (JOINT_LIMITS[i][0] <= q_interp[i] <= JOINT_LIMITS[i][1]):
                    return False
                p.resetJointState(self.robot_id, i, q_interp[i])
            self.collision_checks += 1
            p.performCollisionDetection()
            if p.getContactPoints(bodyA=self.robot_id):
                self.collision_count += 1
                return False
        return True

    def is_goal_reached(self, q_new, q_goal):
        return np.linalg.norm(q_new - q_goal) < self.goal_tolerance

    def plan(self, q_init, q_goal):
        self.tree = [RRTNode(q_init)]
        for self.iterations in range(self.max_iter):
            q_rand = self.get_random_config(q_goal)
            nearest = self.nearest_node(q_rand)
            q_new = self.steer(nearest.config, q_rand)

            if self.is_collision_free_path(nearest.config, q_new):
                new_node = RRTNode(q_new, parent=nearest)
                self.tree.append(new_node)

                if self.is_goal_reached(q_new, q_goal):
                    if self.is_collision_free_path(q_new, q_goal):
                        final_node = RRTNode(q_goal, parent=new_node)
                        self.tree.append(final_node)
                        path = []
                        current = final_node
                        while current:
                            path.append(current.config)
                            current = current.parent
                        print(f"APF-RRT: Goal reached in {self.iterations + 1} iterations.")
                        return path[::-1]
            if self.iterations % 500 == 0:
                print(f"Iteration: {self.iterations}, Tree size: {len(self.tree)}")
        print("APF-RRT: Failed to find a path.")
        return []


def get_end_effector_position(robot_id):
    state = p.getLinkState(robot_id, 11)
    return np.array(state[0])


def setup_pybullet():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    initial_positions = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]
    for i in range(NUM_JOINTS):
        p.resetJointState(robot_id, i, initial_positions[i])

    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=OBSTACLE_RADIUS, rgbaColor=[1, 0, 0, 0.5])
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=OBSTACLE_RADIUS)
    sphere_id = p.createMultiBody(0, sphere_col, sphere_vis, OBSTACLE_POSITION)

    rect_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[x / 2 for x in RECT_OBSTACLE_DIM], rgbaColor=[0, 0, 1, 0.5])
    rect_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x / 2 for x in RECT_OBSTACLE_DIM])
    rect_id = p.createMultiBody(0, rect_col, rect_vis, RECT_OBSTACLE_POS)

    return robot_id, sphere_id, rect_id


def plot_joint_angles(path, speed, filename="joint_angles_apf_rrt.png"):
    if not path:
        return
    times = np.arange(len(path)) * speed
    angles = np.array(path)
    plt.figure(figsize=(12, 7), dpi=300)
    for i in range(NUM_JOINTS):
        plt.plot(times, angles[:, i], label=f'关节 {i + 1}')
    plt.title("APF-RRT 关节角变化曲线")
    plt.xlabel("时间（秒）")
    plt.ylabel("角度（弧度）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"关节角度图已保存为 '{filename}'")


def plot_trajectory(trajectory, filename="end_effector_trajectory_apf_rrt.png"):
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', label='末端执行器轨迹')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('末端执行器轨迹')
    ax.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"末端执行器轨迹图已保存为 '{filename}'")


def calculate_trajectory_length(trajectory):
    total_length = 0.0
    for i in range(1, len(trajectory)):
        total_length += np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i - 1]))
    return total_length


def main():
    robot_id, _, _ = setup_pybullet()
    q_init = np.array([0.0, 0.0, -1.1, -1.8, 0.0, 2.0, 1.5])
    q_goal = np.array([0.0, 0.0, 0.9, -1.8, 0.0, 2.0, 1.5])
    q_goal = np.clip(q_goal, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    planner = APFRRTPlanner(robot_id, MAX_ITER, STEP_SIZE, GOAL_SAMPLE_RATE, GOAL_TOLERANCE)

    planner.add_sphere_obstacle(OBSTACLE_POSITION, OBSTACLE_RADIUS)
    planner.add_box_obstacle(RECT_OBSTACLE_POS, RECT_OBSTACLE_DIM)

    print("开始 APF-RRT 路径规划...")
    start_time = time.time()
    path = planner.plan(q_init, q_goal)
    duration = time.time() - start_time

    if path:
        print(f"路径规划成功！耗时 {duration:.2f} 秒，路径节点数: {len(path)}")
        print(f"总迭代次数: {planner.iterations + 1}")
        print(f"总碰撞检测次数: {planner.collision_checks}")
        print(f"总碰撞次数: {planner.collision_count}")
        plot_joint_angles(path, PLAYBACK_SPEED)

        print("开始路径回放并记录末端执行器轨迹...")
        trajectory = []
        for i in range(NUM_JOINTS):
            p.resetJointState(robot_id, i, q_init[i])
        trajectory.append(get_end_effector_position(robot_id))
        time.sleep(1)
        for config in path:
            for i in range(NUM_JOINTS):
                p.resetJointState(robot_id, i, config[i])
            trajectory.append(get_end_effector_position(robot_id))
            p.stepSimulation()
            p.addUserDebugLine(trajectory[-2], trajectory[-1], [1, 0, 0], 2.0)
            time.sleep(PLAYBACK_SPEED)
        print("路径回放结束。")

        trajectory_length = calculate_trajectory_length(trajectory)
        print(f"末端执行器轨迹总长度: {trajectory_length:.3f} 米")
        plot_trajectory(trajectory)
    else:
        print(f"路径规划失败！耗时 {duration:.2f} 秒")
        print(f"总迭代次数: {planner.iterations}")
        print(f"总碰撞检测次数: {planner.collision_checks}")
        print(f"总碰撞次数: {planner.collision_count}")

    input("按 Enter 键退出...")
    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()
