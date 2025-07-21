import trimesh
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
original_mesh_path = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/objs/deformed_env0_entry0_step0.obj"    # <- path to original vessel mesh (before deformation)
deformed_mesh_path = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/objs/deformed_env0_entry0_step10_new.obj"     # <- path to deformed vessel mesh (after deformation)

# === Load meshes ===
original_mesh = trimesh.load_mesh(original_mesh_path, process=False)
deformed_mesh = trimesh.load_mesh(deformed_mesh_path, process=False)

# === Sanity check ===
assert original_mesh.vertices.shape == deformed_mesh.vertices.shape, "Mismatch in vertex count!"

# === Compute displacement ===
displacement = np.linalg.norm(deformed_mesh.vertices - original_mesh.vertices, axis=1)

print(f"Max displacement: {displacement.max():.6f} m")
print(f"Mean displacement: {displacement.mean():.6f} m")

# === Color-map by displacement ===
colors = plt.cm.viridis((displacement - displacement.min()) / (displacement.max() - displacement.min()))
colors = (colors[:, :3] * 255).astype(np.uint8)  # remove alpha and scale to RGB

# === Apply colors to mesh ===
colored_mesh = deformed_mesh.copy()
colored_mesh.visual.vertex_colors = colors

# === Export colored mesh ===
colored_mesh.export("/home/sanjay/thesis_replications/forked/IsaacLab/custom/objs/deformed_colored.obj")
print("✅ Colored mesh saved to: /path/to/deformed_colored.obj")

# Optional: show with matplotlib
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    original_mesh.vertices[:, 0],
    original_mesh.vertices[:, 1],
    original_mesh.vertices[:, 2],
    c=displacement,
    cmap='viridis',
    s=1,
)
plt.colorbar(sc, label="Displacement (m)")
plt.title("Vertex-wise Displacement (Original → Deformed)")
plt.tight_layout()
plt.show()
