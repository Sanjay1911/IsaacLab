import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_needle_path(u1=0.01, u2=0.1, phi_deg=20, r=0.1, dt=1.0, num_steps=50):
    # Convert bevel angle to radians
    phi = np.deg2rad(phi_deg)

    # Compute body-frame linear and angular velocities
    v = np.array([
        0,
        -u1 * np.sin(phi),
        u1 * np.cos(phi)
    ])

    omega = np.array([
        u1 / r,
        0,
        u2
    ])

    # Build skew-symmetric omega matrix
    omega_hat = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

    # Build twist matrix
    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = omega_hat
    xi_hat[:3, 3] = v

    # Start from identity pose
    g_current = np.eye(4)
    poses = [g_current.copy()]

    for _ in range(num_steps):
        g_next = g_current @ expm(xi_hat * dt)
        poses.append(g_next.copy())
        g_current = g_next

    return poses

def plot_needle_trajectory(poses):
    positions = np.array([g[:3, 3] for g in poses])
    directions = np.array([g[:3, 2] for g in poses])  # local +Z axis

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.01, 0.02)
    ax.set_ylim(-0.01, 0.01)
    ax.set_zlim(0, 0.06)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bevel-Tip Needle Motion (Insertion + Twist)')

    # Plot full path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'g-', label="Needle Tip Path", linewidth=2)

    # Draw needle body orientation at intervals
    for i in range(0, len(positions), 5):
        tip = positions[i]
        dir_vec = directions[i] * 0.01
        base = tip - dir_vec
        ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], 'r-', linewidth=3)

    # Mark initial and final directions
    ax.quiver(*positions[0], *directions[0], length=0.01, color='blue', label='Initial Direction')
    ax.quiver(*positions[-1], *directions[-1], length=0.01, color='black', label='Final Direction')

    plt.legend()
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    poses = simulate_needle_path(
        u1=0.01,    # insertion speed (m/s)
        u2=0.1,     # twist speed (rad/s)
        phi_deg=20, # bevel angle in degrees
        r=0.1,      # curvature radius (m)
        dt=1.0,     # time step (s)
        num_steps=50
    )
    plot_needle_trajectory(poses)
