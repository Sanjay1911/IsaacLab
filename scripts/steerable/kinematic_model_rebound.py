import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew(v):
    """Skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def twist_to_matrix(v, w):
    """Convert a twist (v, w) to a 4x4 matrix."""
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(w)
    mat[:3, 3] = v
    return mat

def generate_needle_path(
    u1=0.01, u2=0.1, phi_deg=30, r=0.2, reb=0.001,
    dt=1.0, num_steps=5, start=np.eye(4), goal=np.array([0.015, -0.005, 0.05])
):
    phi = np.deg2rad(phi_deg)
    poses = [start.copy()]
    g_current = start.copy()

    last_dir = np.array([0, 0, 1])
    for k in range(num_steps):
        # Compute direction to goal
        tip_pos = g_current[:3, 3]
        to_goal = goal - tip_pos
        to_goal = to_goal / np.linalg.norm(to_goal)

        # Check if curvature direction has changed significantly
        current_dir = g_current[:3, 2]
        angle_diff = np.arccos(np.clip(np.dot(current_dir, to_goal), -1, 1))
        direction_changed = angle_diff > np.deg2rad(10)  # 10 degree threshold

        # Twist components
        v = np.array([
            0,
            -u1 * np.sin(phi),
            u1 * np.cos(phi)
        ])
        w = np.array([
            u1 / r,
            0,
            u2
        ])

        xi_hat = twist_to_matrix(v, w)

        if direction_changed:
            # Step 1: apply reduced step (T - t)
            t_mod = reb / u1
            g_partial = g_current @ expm(xi_hat * (dt - t_mod))

            # Step 2: translate forward by reb along z
            z_axis = g_partial[:3, 2]
            trans = np.eye(4)
            trans[:3, 3] = z_axis * reb

            # Step 3: rotate around z to align curvature direction with goal
            axis = np.array([0, 0, 1])
            theta = np.arctan2(to_goal[1], to_goal[0]) - np.arctan2(z_axis[1], z_axis[0])
            rot_z = np.eye(4)
            rot_z[:3, :3] = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            g_current = g_partial @ trans @ rot_z
        else:
            # Normal exponential step
            g_current = g_current @ expm(xi_hat * dt)

        poses.append(g_current.copy())
        last_dir = current_dir

    return poses

def visualize_needle_path(poses, start, goal):
    positions = np.array([g[:3, 3] for g in poses])
    directions = np.array([g[:3, 2] for g in poses])  # +Z axis of needle tip

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'g-', label="Needle Tip Path", linewidth=2)

    for i in range(0, len(positions), 5):
        tip = positions[i]
        dir_vec = directions[i] * 0.01
        base = tip - dir_vec
        ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], 'r-', linewidth=3)

    ax.quiver(*positions[0], *directions[0], length=0.01, color='blue', label='Start Direction')
    ax.quiver(*positions[-1], *directions[-1], length=0.01, color='black', label='End Direction')

    ax.scatter(*start[:3, 3], color='blue', s=50, label='Start Point')
    ax.scatter(*goal, color='black', s=50, label='Goal Point')

    ax.set_xlim([-0.01, 0.03])
    ax.set_ylim([-0.02, 0.02])
    ax.set_zlim([0.0, 0.06])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Needle Path from Start to Goal")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Define start pose and goal point
start_pose = np.eye(4)
goal_point = np.array([0.015, -0.005, 0.05])

# Simulate and visualize
needle_poses = generate_needle_path(start=start_pose, goal=goal_point)
visualize_needle_path(needle_poses, start_pose, goal_point)

