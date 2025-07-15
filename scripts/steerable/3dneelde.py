import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# Load a simple 3D needle mesh (substitute with your actual path if available)
needle_mesh = trimesh.creation.cylinder(radius=0.0002, height=0.01, sections=16)

# Define needle parameters
u1 = 0.009  # insertion speed
u2 = 0.0    # rotation speed
phi = np.deg2rad(30)  # bevel angle in radians
r = 0.12    # radius of curvature
dt = 0.2    # time step
steps = 40  # number of steps

# Initial pose: Identity
g_current = np.eye(4)
poses = [g_current.copy()]

# Compute twist
v = np.array([0,
              -u1 * np.sin(phi),
              u1 * np.cos(phi)])
w = np.array([u1 / r, 0, u2])
xi = np.zeros((4, 4))
xi[:3, :3] = np.array([[0, -w[2], w[1]],
                       [w[2], 0, -w[0]],
                       [-w[1], w[0], 0]])
xi[:3, 3] = v

# Integrate over time
for _ in range(steps):
    g_next = g_current @ expm(xi * dt)
    poses.append(g_next.copy())
    g_current = g_next

# Apply transformations to the needle mesh and collect for visualization
transformed_meshes = []
for pose in poses:
    transformed = needle_mesh.copy()
    transformed.apply_transform(pose)
    transformed_meshes.append(transformed)

# Combine into a single scene
scene = trimesh.Scene()
for mesh in transformed_meshes:
    scene.add_geometry(mesh)

# Display the result
scene.show()