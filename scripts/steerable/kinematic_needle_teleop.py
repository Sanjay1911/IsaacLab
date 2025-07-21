import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.linalg import expm
import numpy as np
import os

# Re-define functions to avoid kernel reset issues
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def twist_to_matrix(v, w):
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew(w)
    mat[:3, 3] = v
    return mat

def generate_needle_path(u1, u2, phi_deg=30, r=0.2, reb=0.001,
                         dt=1.0, num_steps=5, start=np.eye(4), goal=np.array([0.015, -0.005, 0.05])):
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

# Interactive plot
start_pose = np.eye(4)
goal_point = np.array([0.015, -0.005, 0.05])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Default values
u1_init = 0.01
u2_init = 0.0

poses = generate_needle_path(u1=u1_init, u2=u2_init)
path_line, = ax.plot([], [], [], 'g-', linewidth=2)
start_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='blue')
end_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='black')
start_dot = ax.scatter(*start_pose[:3, 3], color='blue', s=50)
goal_dot = ax.scatter(*goal_point, color='black', s=50)

def update_plot(u1, u2):
    poses = generate_needle_path(u1=u1, u2=u2)
    positions = np.array([g[:3, 3] for g in poses])
    directions = np.array([g[:3, 2] for g in poses])

    path_line.set_data(positions[:, 0], positions[:, 1])
    path_line.set_3d_properties(positions[:, 2])

    # Start direction
    s_pos, s_dir = positions[0], directions[0] * 0.01
    e_pos, e_dir = positions[-1], directions[-1] * 0.01
    ax.quiver(*s_pos, *s_dir, color='blue')
    ax.quiver(*e_pos, *e_dir, color='black')
    ax.scatter(*start_pose[:3, 3], color='blue', s=50)
    ax.scatter(*goal_point, color='black', s=50)

    fig.canvas.draw_idle()

# Sliders for u1 and u2
ax_u1 = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_u2 = plt.axes([0.1, 0.1, 0.8, 0.03])
s_u1 = Slider(ax_u1, 'u1 (feed)', 0.001, 0.05, valinit=u1_init)
s_u2 = Slider(ax_u2, 'u2 (twist)', -0.5, 0.5, valinit=u2_init)

def on_change(val):
    ax.cla()
    update_plot(s_u1.val, s_u2.val)

s_u1.on_changed(on_change)
s_u2.on_changed(on_change)

# Initial update
update_plot(u1_init, u2_init)

plt.show()
