import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_paths(paths_pickle):
    with open(paths_pickle, "rb") as f:
        data = pickle.load(f)
    # data: list of {env_id, path_idx, steps:[{...}]}
    trajs = []
    for item in data:
        env_id = int(item["env_id"])
        path_idx = int(item["path_idx"])
        steps = item["steps"]
        if not steps:
            continue
        # prefer tooltip_pos; fallback to new_pos
        xyz = []
        for s in steps:
            if "tooltip_pos" in s:
                p = np.asarray(s["tooltip_pos"]).reshape(-1)  # [3]
            elif "new_pos" in s:
                p = np.asarray(s["new_pos"]).reshape(-1)
            else:
                continue
            if p.shape[0] == 3:
                xyz.append(p)
        if len(xyz) > 0:
            trajs.append({
                "env_id": env_id,
                "path_idx": path_idx,
                "xyz": np.stack(xyz, axis=0)  # [T,3]
            })
    return trajs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_pickle", required=True, help="agent_paths_*.pkl produced by play.py")
    ap.add_argument("--preop_pickle", default=None, help="original dataset pickle to overlay straight line (optional)")
    ap.add_argument("--out", default=None, help="save figure path (optional)")
    ap.add_argument("--per_env", action="store_true", help="make one figure per env instead of a single figure")
    args = ap.parse_args()

    trajs = load_paths(args.paths_pickle)
    preop = load_preop(args.preop_pickle)
    env_id = 0      # choose your env
    path_idx = 0    # choose your path index

    # Find that trajectory
    traj = next(
        (t for t in trajs if t["env_id"] == env_id and t["path_idx"] == path_idx),
        None
    )

    if traj is None:
        print(f"No trajectory for env {env_id}, path_idx {path_idx}")
    else:
        traj_start = traj["xyz"][0]  # [x, y, z] of first step
        preop_start = preop[env_id]["start_positions"][path_idx]

        print("Trajectory start:", traj_start)
        print("Pre-op start    :", preop_start)

        diff = traj_start - preop_start
        dist = np.linalg.norm(diff)

        print("Difference vector:", diff)
        print("Euclidean distance:", dist)

        if np.allclose(traj_start, preop_start, atol=1e-6):
            print("✅ Starts match (within tolerance).")
        else:
            print("❌ Starts differ.")

        if len(trajs) == 0:
            print("No trajectories found.")
            return

    # group by env
    env_ids = sorted(set(t["env_id"] for t in trajs))

    def plot_env(ax, env_id, env_trajs):
        ax.set_title(f"Env {env_id}")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        # overlay pre-op lines if available
        if env_id in preop:
            starts = preop[env_id]["start_positions"]
            tumor = preop[env_id]["tumor_centroid"]
            # draw a thin line for all start->tumor pairs (faint)
            for i, s in enumerate(starts):
                ax.plot([s[0], tumor[0]], [s[1], tumor[1]], [s[2], tumor[2]], linestyle="--", alpha=0.2)
        # plot agent trajectories
        for t in env_trajs:
            xyz = t["xyz"]
            pid = t["path_idx"]
            ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label=f"path_idx={pid}")
            # start/end markers
            ax.scatter(xyz[0,0], xyz[0,1], xyz[0,2], marker="o", s=20)
            ax.scatter(xyz[-1,0], xyz[-1,1], xyz[-1,2], marker="^", s=30)
            # if we know the matching pre-op line, draw the specific one bolder
            if env_id in preop and pid < preop[env_id]["start_positions"].shape[0]:
                s = preop[env_id]["start_positions"][pid]
                tumor = preop[env_id]["tumor_centroid"]
                ax.plot([s[0], tumor[0]], [s[1], tumor[1]], [s[2], tumor[2]], linestyle="--", linewidth=2, alpha=0.6)
        ax.legend(loc="best")

    if args.per_env:
        for env_id in env_ids:
            env_trajs = [t for t in trajs if t["env_id"] == env_id]
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection="3d")
            plot_env(ax, env_id, env_trajs)
            plt.tight_layout()
            if args.out:
                base, ext = (args.out.rsplit(".", 1) + ["png"])[:2]
                out_path = f"{base}_env{env_id}.{ext}"
                plt.savefig(out_path, dpi=200)
                print(f"saved {out_path}")
            else:
                plt.show()
            plt.close(fig)
    else:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d")
        for env_id in env_ids:
            env_trajs = [t for t in trajs if t["env_id"] == env_id]
            plot_env(ax, env_id, env_trajs)
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=200)
            print(f"saved {args.out}")
        else:
            plt.show()

if __name__ == "__main__":
    main()
