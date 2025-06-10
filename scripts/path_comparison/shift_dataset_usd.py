import numpy as np
np.random.seed(42)  # For reproducibility
import torch
from pxr import Usd, UsdGeom
from isaaclab.utils.io import load_pickle, dump_pickle
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import trimesh
import omni

# === Settings ===
USD_STAGE_PATH = "/home/sanjay/thesis_replications/forked/Vessels.usd"
TUMOR_DATA_PATH = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/pickle_finale/rl_dataset_10envs.pkl"
OUTPUT_PATH = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/path_comparison/pickle_finale/precomputed_brain_deformations_10envs.pkl"
PRIM_PATH = "/Root/Vessels/Vessels"
# === Save OBJ files? ===
SAVE_OBJ = False  # Set to True to save OBJ files
# === Constants ===
MAX_STEPS = 10
INFLUENCE_RADIUS = 0.04
ANCHOR_RADIUS = 1.5
MAX_SHIFT = 0.015
NUM_ENVS = 10              
NUM_TOP_PATHS = 10          
OBJ_EXPORT_DIR = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/objs"

# === Load USD mesh ===
print("[INFO] Opening USD stage...")
stage = Usd.Stage.Open(USD_STAGE_PATH)
mesh_prim = stage.GetPrimAtPath(PRIM_PATH)
usd_mesh = UsdGeom.Mesh(mesh_prim)

points = np.asarray(usd_mesh.GetPointsAttr().Get())
print(f"[INFO] Loaded {len(points)} vertices from USD mesh.")
transform_matrix = np.array(usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
points = np.matmul(points, transform_matrix[:3, :3].T)
points += transform_matrix[:3, 3]

faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
if not np.all(counts == 3):
    raise ValueError("Mesh must be triangulated.")
faces = faces.reshape((-1, 3))
mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

# === Load tumor data ===
tumor_pickle = load_pickle(TUMOR_DATA_PATH)
tumor_top_entry_points = []
tumor_centroids = []
print(f"[INFO] Processing tumor data for {len(tumor_pickle)} entries")
for i in range(len(tumor_pickle)):
    try:
        entry = tumor_pickle[i]
        tumor_top_entry_points.append(entry["top_entry_points"])
        tumor_centroids.append(entry["tumor_centroid"])
    except KeyError:
        print(f"[WARN] Missing data for env {i}")

# === Build KDTree on original mesh vertices ===
tree = cKDTree(points)

# === Deformation function ===
def compute_deformation_steps(vertices, center, entry_point, tumor_centroid, env_id=None, entry_id=None):
    num_samples = 3000
    surface_samples, _ = trimesh.sample.sample_surface(mesh, num_samples)
    dists = np.linalg.norm(surface_samples - center, axis=1)
    control_points = surface_samples[dists < INFLUENCE_RADIUS]
    print(f"[INFO] Found {len(control_points)} control points near center.")

    if len(control_points) == 0:
        print("[WARN] No control points found. Skipping.")
        return None

    if len(control_points) > 1000:
        idx = np.random.choice(len(control_points), 1000, replace=False)
        control_points = control_points[idx]

    original_vertices = vertices.copy()
    all_steps = []

    for step in range(MAX_STEPS + 1):
        shift_ratio = step / MAX_STEPS
        displacements = np.zeros_like(control_points)

        for i, p in enumerate(control_points):
            dist = np.linalg.norm(p - center)
            falloff = np.exp(-dist**2 / (2 * (INFLUENCE_RADIUS / 2)**2))
            path_vector = (tumor_centroid - entry_point)
            path_vector /= np.linalg.norm(path_vector)  # Normalize

            # Add small random noise to the direction
            noise_strength = 0.05  # You can tune this
            noise = np.random.normal(scale=noise_strength, size=3)

            # Combine and normalize again
            noisy_vector = path_vector + noise
            noisy_vector /= np.linalg.norm(noisy_vector)

            # Apply displacement in noisy direction
            displacements[i] += noisy_vector * (MAX_SHIFT * falloff * shift_ratio)
            #displacements[i, 2] -= MAX_SHIFT * falloff * shift_ratio

        try:
            rbf = RBFInterpolator(control_points, displacements, kernel="thin_plate_spline", smoothing=1e-5)
            deformation = rbf(original_vertices)
            deformed_vertices = original_vertices + deformation
            print(f"[DEBUG] Step {step}: Deformed {len(deformed_vertices)} vertices")
            assert deformed_vertices.shape == original_vertices.shape, "Deformed vertices shape mismatch"
            if step == 5 and env_id is not None and entry_id is not None:
                all_steps.append(deformed_vertices.astype(np.float32))
                print(f"[DEBUG] Step {step} deformation saved for env {env_id}, entry {entry_id}")
                break

            if step == 5 and env_id is not None and entry_id is not None and SAVE_OBJ is True:    # irrespectuive of max step, the final deformation will be same because of shift ratio. for max step 10 and 50, it still saves 100% deformation. So to save intermediate step, change this condition
                filename = f"{OBJ_EXPORT_DIR}/deformed_env{env_id}_entry{entry_id}_step{step}_noisy.obj"
                mesh_deformed =  trimesh.Trimesh(vertices=deformed_vertices, faces=faces, process=False)
                mesh_deformed.apply_translation(-mesh.centroid)
                mesh_deformed.export(filename)
                print(f"[EXPORT] Saved {filename}")
                break

            disp = np.linalg.norm(deformation, axis=1)
            print(f"[DEBUG] Step {step}: max = {disp.max():.6f}, mean = {disp.mean():.6f}")
        except Exception as e:
            print(f"[ERROR] Step {step} RBF failed: {e}")
            return None

    return all_steps

# === Main deformation loop ===
deformation_data = {}

for env_id, top_entries in enumerate(tumor_top_entry_points[:NUM_ENVS]):
    print(f"\n[ENV {env_id}] Processing {len(top_entries)} top entry points")
    deformation_data[env_id] = []

    for entry_id in range(min(NUM_TOP_PATHS, len(top_entries))):
        entry_info = top_entries[entry_id]
        tumor_centroid = tumor_centroids[env_id]
        print(f"[Entry {entry_id}]")
        entry_point = entry_info["entry_point"]
        closest_idx = tree.query(entry_point, k=1)[1]
        center = points[closest_idx]

        step_deformations = compute_deformation_steps(points, center, entry_point, tumor_centroid, env_id=env_id, entry_id=entry_id)
        if step_deformations is not None:
            deformation_data[env_id].append(step_deformations)

# === Save data in [[[]]] format for simulation (optional) ===
final_data = [[steps] for steps in deformation_data.values()]
dump_pickle(OUTPUT_PATH, final_data)  # Uncomment if needed
print(f"\n✅ Finished deformation computation. Data saved to {OUTPUT_PATH}")
