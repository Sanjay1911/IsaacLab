import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# === CONFIG ===
paths_pickle = "/home/sanjay/thesis_replications/forked/IsaacLab/logs/skrl/Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0/2025-08-08_17-13-56_ppo_torch/metrics/agent_paths_20250809_195930.pkl"
preop_pickle = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/pickle_finale/rl_dataset_10envs.pkl"

env_id    = 0
max_paths = 20

# Grid / offsets
ENV_SPACING      = 0.2              # meters
GRID_MODE        = "sqrt"           # "row" | "sqrt" | "cols"
GRID_COLS        = None             # e.g., 5 if GRID_MODE == "cols"
OFFSET_TO_SKULL  = np.array([0.0, 0.049, 0.0])  # meters
PREOP_UNIT_SCALE = 1.0              # 0.001 if pre-op is in mm

# Diagnostics / alignment
APPLY_MEAN_TRANSLATION = True      # set True to re-anchor pre-op to agent by mean delta

# Spheres
SPHERE_R_START = 0.00
SPHERE_R_END   = 0.00
SPHERE_RES     = 20

def draw_sphere(ax, center, radius, color=None, alpha=0.9, res=SPHERE_RES):
    c = np.asarray(center, dtype=float)
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = c[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = c[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = c[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, edgecolor="none", alpha=alpha)

def compute_env_offset(env_id: int, num_envs: int, spacing: float, mode="sqrt", cols: int | None = None):
    if mode == "row":
        row, col = 0, env_id
    elif mode == "sqrt":
        c = int(np.ceil(np.sqrt(num_envs)))
        row, col = divmod(env_id, c)
    elif mode == "cols":
        if not isinstance(cols, int) or cols <= 0:
            raise ValueError("When GRID_MODE='cols', set GRID_COLS to a positive int.")
        row, col = divmod(env_id, cols)
    else:
        raise ValueError("GRID_MODE must be 'row', 'sqrt', or 'cols'")
    return np.array([col * spacing, row * spacing, 0.0], dtype=float)

def load_paths_one_env(paths_pickle, env_id, max_paths):
    with open(paths_pickle, "rb") as f:
        data = pickle.load(f)
    trajs = []
    for item in data:
        if int(item["env_id"]) != env_id:
            continue
        pid = int(item["path_idx"])
        steps = item.get("steps", [])
        if not steps:
            continue
        xyz = []
        for s in steps:
            if "tooltip_pos" in s:
                p = np.asarray(s["tooltip_pos"]).reshape(-1)
            elif "new_pos" in s:
                p = np.asarray(s["new_pos"]).reshape(-1)
            else:
                continue
            if p.shape[0] == 3:
                xyz.append(p)
        if xyz:
            trajs.append({"path_idx": pid, "xyz": np.stack(xyz, axis=0)})
    trajs = sorted(trajs, key=lambda t: t["path_idx"])[:max_paths]
    return trajs

def load_preop_one_env_world(preop_pickle, env_id, env_offset, offset_to_skull, scale=1.0):
    with open(preop_pickle, "rb") as f:
        ds = pickle.load(f)
    num_envs = len(ds) if not isinstance(ds, dict) else len(ds.keys())
    entry = ds[env_id]
    starts_local = np.array([p["position"] for p in entry["start_pose"]], dtype=float) * scale
    tumor_local  = np.array(entry["tumor_centroid"], dtype=float) * scale
    starts_world = starts_local + env_offset + offset_to_skull
    tumor_world  = tumor_local  + env_offset + offset_to_skull
    return starts_world, tumor_world, num_envs

# --- Load, compute offsets, and diagnose ---
# get num_envs to compute env offset consistently with preop
dummy_offset = np.zeros(3)
_, _, num_envs = load_preop_one_env_world(preop_pickle, env_id, dummy_offset, np.zeros(3), PREOP_UNIT_SCALE)
ENV_OFFSET = compute_env_offset(env_id, num_envs, ENV_SPACING, GRID_MODE, GRID_COLS)

start_positions_w, tumor_centroid_w, _ = load_preop_one_env_world(
    preop_pickle, env_id, ENV_OFFSET, OFFSET_TO_SKULL, PREOP_UNIT_SCALE
)
trajs = load_paths_one_env(paths_pickle, env_id, max_paths)

# Compute difference vectors: agent_start - preop_start_world
deltas = []
pairs  = []
for t in trajs:
    pid = t["path_idx"]
    if pid >= start_positions_w.shape[0]:
        continue
    agent_start = t["xyz"][0]
    preop_start = start_positions_w[pid]
    delta = agent_start - preop_start
    deltas.append(delta)
    pairs.append((pid, agent_start, preop_start))

deltas = np.array(deltas) if deltas else np.zeros((0,3))
if len(deltas):
    norms = np.linalg.norm(deltas, axis=1)
    mean_delta = deltas.mean(axis=0)
    std_delta  = deltas.std(axis=0)
    print("\n--- Alignment diagnostics ---")
    for (pid, a, s), d, n in zip(pairs, deltas, norms):
        print(f"path_idx {pid:2d}: delta = {d},  ||delta|| = {n:.6f} m")
    print(f"\nmean delta: {mean_delta}  (||mean|| = {np.linalg.norm(mean_delta):.6f} m)")
    print(f"std  delta: {std_delta}    (component-wise)")
else:
    mean_delta = np.zeros(3)

# Optionally apply the mean translation to pre-op (so dashed lines start exactly where agent starts)
if APPLY_MEAN_TRANSLATION and len(deltas):
    start_positions_w = start_positions_w + mean_delta
    tumor_centroid_w  = tumor_centroid_w + mean_delta
    print("\n[info] Applied mean translation to pre-op data for visualization.")

# --- Plot ---
# --- Get a consistent color map for path indices ---
unique_pids = sorted({t["path_idx"] for t in trajs})
color_cycle = plt.cm.tab10(np.linspace(0, 1, len(unique_pids)))  # or any colormap you like
pid_to_color = {pid: color_cycle[i] for i, pid in enumerate(unique_pids)}

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(f"Env {env_id} — first {max_paths} paths")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

# Track legend handles
legend_handles = []

for t in trajs:
    pid = t["path_idx"]
    xyz = t["xyz"]
    color = pid_to_color[pid]

    # Draw curved path
    curved_line, = ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, label=f"path_idx={pid}")
    draw_sphere(ax, xyz[0],  SPHERE_R_START, color=color, alpha=0.95)  # start
    draw_sphere(ax, xyz[-1], SPHERE_R_END,   color=color, alpha=0.95)  # end

    # Draw straight pre-op line
    if pid < start_positions_w.shape[0]:
        s = start_positions_w[pid]
        ax.plot([s[0], tumor_centroid_w[0]],
                [s[1], tumor_centroid_w[1]],
                [s[2], tumor_centroid_w[2]],
                linestyle="--", color=color, alpha=0.6)

    # Add to legend only once per path_idx
    if pid not in [h.get_label().replace("path_idx=", "") for h in legend_handles]:
        legend_handles.append(curved_line)

# One clean legend
ax.legend(handles=legend_handles, title="Paths", loc="best")
plt.tight_layout()
plt.show()
