import argparse
import pickle
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_trajectories(path_pickle):
    with open(path_pickle, "rb") as f:
        data = pickle.load(f)
    return data["trajectories"] if isinstance(data, dict) and "trajectories" in data else data


def load_preop(preop_pickle):
    """Return: { env_id:int -> { 'start_positions': [N,3], 'start_quaternions':[N,4], 'tumor_centroid':[3] } }"""
    if preop_pickle is None:
        return {}
    with open(preop_pickle, "rb") as f:
        ds = pickle.load(f)

    # support list-like or dict-like datasets
    it = ds.items() if isinstance(ds, dict) else enumerate(ds)

    out = {}
    for k, env_data in it:
        env_id = int(k)

        # required keys for this dataset
        if "start_pose" not in env_data or "tumor_centroid" not in env_data:
            raise KeyError(f"Entry {env_id} missing 'start_pose' or 'tumor_centroid'")

        starts = env_data["start_pose"]  # list of dicts
        # robustly extract positions/quats
        pos_list, quat_list = [], []
        for p in starts:
            # Accept dicts {position, quaternion}; ignore malformed entries
            if isinstance(p, dict):
                if "position" in p:
                    pos = np.asarray(p["position"], dtype=float).reshape(-1)
                    if pos.shape[0] == 3:
                        pos_list.append(pos)
                if "quaternion" in p:
                    q = np.asarray(p["quaternion"], dtype=float).reshape(-1)
                    if q.shape[0] == 4:
                        quat_list.append(q)
            # You could also accept raw [x,y,z] if ever present:
            elif isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
                pos_list.append(np.asarray(p, dtype=float))

        start_positions = np.array(pos_list, dtype=float) if pos_list else np.empty((0, 3))
        start_quaternions = np.array(quat_list, dtype=float) if quat_list else np.empty((0, 4))

        tumor_centroid = np.asarray(env_data["tumor_centroid"], dtype=float).reshape(-1)
        if tumor_centroid.shape[0] != 3:
            raise ValueError(f"Entry {env_id} 'tumor_centroid' must be length 3")

        out[env_id] = {
            "start_positions": start_positions,
            "start_quaternions": start_quaternions,
            "tumor_centroid": tumor_centroid,
        }

    return out


def extract_curve(traj):
    steps = traj["steps"]
    if len(steps) == 0:
        return None, None, None

    first_step = steps[0]
    path_idx = int(first_step["active_path_index"])
    start_positions = np.asarray(first_step["start_positions"])
    tumor_centroid = np.asarray(first_step["tumor_centroids"])
    chosen_start = start_positions[path_idx]
    curve = np.stack([np.asarray(s["tooltip_pos"]) for s in steps])

    return chosen_start, tumor_centroid, curve


def load_tumor_mesh(obj_path, tumor_centroid, n_samples=5000):
    """Load OBJ tumor mesh, sample surface points, shift centroid to tumor_centroid."""
    mesh = trimesh.load(obj_path, process=True)
    points, _ = trimesh.sample.sample_surface(mesh, n_samples)    
    # compute current centroid
    centroid = points.mean(axis=0)
    shift = tumor_centroid - centroid
    shifted_points = points + shift
    return shifted_points


def plot_scene(trajs, tumor_points, env_id=None, preop_start=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # plot tumor sampled cloud
    # ax.scatter(tumor_points[:, 0], tumor_points[:, 1], tumor_points[:, 2],
    #            c="magenta", alpha=0.3, s=2, label="Tumor mesh samples")
    traj_count = 10
    for traj in trajs:
        print(f"Trajectory Start: {traj['steps'][0]['tooltip_pos']}, End: {traj['steps'][-1]['tooltip_pos']}")
        print(f"Preop Start: {preop_start}")
        if traj_count >= 30:
            break
        if env_id is not None and traj["env_id"] != env_id:
            continue

        start, tumor, curve = extract_curve(traj)
        if start is None:
            continue

        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=f"Env {traj['env_id']} Path {traj['path_idx']}")
        ax.scatter(start[0], start[1], start[2], c="g", marker="o", s=40)
        ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], c="r", marker="^", s=50)
        ax.plot([start[0], tumor[0]], [start[1], tumor[1]], [start[2], tumor[2]],
                linestyle="--", color="orange", alpha=0.7)
        ax.scatter(tumor[0], tumor[1], tumor[2], c="black", marker="x", s=60)
        traj_count += 1

    #ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_pickle", required=True, help="Trajectories pickle (from play script)")
    ap.add_argument("--tumor_obj", required=True, help="Tumor OBJ file path")
    ap.add_argument("--env_id", type=int, default=None, help="Filter by env_id (optional)")
    ap.add_argument("--preop_pickle", default=None, help="original dataset pickle to overlay straight line (optional)")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj_pickle)
    preop = load_preop(args.preop_pickle)
    path_idx = 0  # specify the path index you want to use
    preop_start = preop[env_id]["start_positions"][path_idx]
    print(f"[INFO] Loaded {len(trajs)} trajectories")

    # take tumor centroid from first trajectory (or env_id if provided)
    if args.env_id is not None:
        trajs = [t for t in trajs if t["env_id"] == args.env_id]
    if len(trajs) == 0:
        print("[WARN] No trajectories to plot")
        return

    _, tumor_centroid, _ = extract_curve(trajs[0])
    tumor_points = load_tumor_mesh(args.tumor_obj, tumor_centroid)

    plot_scene(trajs, tumor_points, env_id=args.env_id, preop_start)


if __name__ == "__main__":
    main()
