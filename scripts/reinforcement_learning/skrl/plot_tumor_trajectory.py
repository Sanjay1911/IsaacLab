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


def plot_scene(trajs, tumor_points, env_id=None):
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
    args = ap.parse_args()

    trajs = load_trajectories(args.traj_pickle)
    print(f"[INFO] Loaded {len(trajs)} trajectories")

    # take tumor centroid from first trajectory (or env_id if provided)
    if args.env_id is not None:
        trajs = [t for t in trajs if t["env_id"] == args.env_id]
    if len(trajs) == 0:
        print("[WARN] No trajectories to plot")
        return

    _, tumor_centroid, _ = extract_curve(trajs[0])
    tumor_points = load_tumor_mesh(args.tumor_obj, tumor_centroid)

    plot_scene(trajs, tumor_points, env_id=args.env_id)


if __name__ == "__main__":
    main()
