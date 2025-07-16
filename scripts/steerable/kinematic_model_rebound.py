import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime
# Utility functions
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def twist_to_matrix(v, w):
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(w)
    mat[:3, 3] = v
    return mat

# Needle simulation with adjustable parameters
def generate_needle_path(u1=0.01, u2=0.1, phi_deg=30, r=0.2, reb=0.001,
                         dt=0.5, num_steps=100, start=np.eye(4), goal=np.array([0.015, -0.005, 0.05])):
    phi = np.deg2rad(phi_deg)
    poses = [start.copy()]
    g_current = start.copy()

    for k in range(num_steps):
        tip_pos = g_current[:3, 3]
        to_goal = goal - tip_pos
        to_goal = to_goal / np.linalg.norm(to_goal)
        current_dir = g_current[:3, 2]
        angle_diff = np.arccos(np.clip(np.dot(current_dir, to_goal), -1, 1))
        direction_changed = angle_diff > np.deg2rad(10)

        v = np.array([0, -u1 * np.sin(phi), u1 * np.cos(phi)])
        w = np.array([u1 / r, 0, u2])
        xi_hat = twist_to_matrix(v, w)

        if direction_changed:
            t_mod = reb / u1
            g_partial = g_current @ expm(xi_hat * (dt - t_mod))
            z_axis = g_partial[:3, 2]
            trans = np.eye(4)
            trans[:3, 3] = z_axis * reb
            axis = np.array([0, 0, 1])
            theta = np.arctan2(to_goal[1], to_goal[0]) - np.arctan2(z_axis[1], z_axis[0])
            rot_z = np.eye(4)
            rot_z[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta),  np.cos(theta), 0],
                                      [0, 0, 1]])
            g_current = g_partial @ trans @ rot_z
        else:
            g_current = g_current @ expm(xi_hat * dt)

        poses.append(g_current.copy())
    return poses


# Visualization and saving
def visualize_and_save(poses, start, goal, title, filename):
    positions = np.array([g[:3, 3] for g in poses])
    directions = np.array([g[:3, 2] for g in poses])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'g-', linewidth=2, label="Needle Tip Path")

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
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(f"/home/sanjay/thesis_replications/forked/IsaacLab/scripts/steerable/needle_plots/{date}", exist_ok=True)
    plt.savefig(f"/home/sanjay/thesis_replications/forked/IsaacLab/scripts/steerable/needle_plots/{date}/{filename}.png")
    plt.close()


# Run for different phi and r
start_pose = np.eye(4)
goal_point = np.array([0.015, -0.005, 0.05])
phi_values = [10, 20, 30, 45]
radius_values = [0.1, 0.2, 0.4]


# Generate plots for varying phi
for phi in phi_values:
    poses = generate_needle_path(phi_deg=phi, r=0.2, start=start_pose, goal=goal_point)
    visualize_and_save(poses, start_pose, goal_point, f"Needle Path (phi={phi}°)", f"phi_{phi}")


# Generate plots for varying r
for r in radius_values:
    poses = generate_needle_path(phi_deg=30, r=r, start=start_pose, goal=goal_point)
    visualize_and_save(poses, start_pose, goal_point, f"Needle Path (r={r}m)", f"radius_{r}")


