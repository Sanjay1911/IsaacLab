import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

R_n = 0.001  # needle outer radius in meters

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
        if "start_pose" not in env_data or "tumor_centroid" not in env_data or "top_entry_points" not in env_data:
            raise KeyError(f"Entry {env_id} missing 'start_pose' or 'tumor_centroid' or 'top_entry_points'")

        starts = env_data["start_pose"]  # list of dicts
        collision_score = env_data["top_entry_points"]  # list of floats
        print(f"ENV {env_id} | Collision scores for top entry points: and length {len(collision_score[:10])}")
        print(collision_score[:20][6]["score"])
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
        return None, None, None, None

    # extract mean, min and max deviation over the trajectory
    deviations = [float(np.asarray(s.get("deviation_t", 0.0))) for s in steps]
    for i, d in enumerate(deviations):
        d += R_n  # account for needle OD
        deviations[i] = d
    mean_dev = np.mean(deviations)
    min_dev = np.min(deviations)
    max_dev = np.max(deviations)
    print(f"Mean deviation: {mean_dev:.6f}, Min deviation: {min_dev:.6f}, Max deviation: {max_dev:.6f}")

    first_step = steps[0]
    #print(f"First step keys: {list(first_step.keys())}")
    path_idx = int(first_step["active_path_index"])
    start_positions = np.asarray(first_step["start_positions"])
    tumor_centroid = np.asarray(first_step["tumor_centroids"])
    deviation = first_step.get("deviation_t", 0.0)
    #print(f"Path index: {path_idx}, Deviation: {deviation:.6f}")
    chosen_start = start_positions[path_idx]

    curve = np.stack([np.asarray(s["tooltip_pos"]) for s in steps])

    # also collect collision indices (reward_collision == -50)
    collisions = []
    for s in steps:
        val = float(np.asarray(s.get("reward_collision", 0.0)))
        if abs(val + 50.0) < 1e-3:  # ~ -50
            collisions.append(np.asarray(s["tooltip_pos"]))
    collisions = np.array(collisions) if len(collisions) > 0 else None

    return chosen_start, tumor_centroid, curve, collisions


def plot_scene(trajs, preop_start, env_id=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    pure_success = 0
    collided_success = 0
    traj_counter = 0
    one_step_counter = 0

    #print(f"Keys: {list(trajs[0].keys())}")
    # for i in range(len(trajs)):
    #     if len(trajs[i]["steps"]) <= 1:
    #         one_step_counter += 1
    #         trajs[i] = None
    # trajs = [t for t in trajs if t is not None]

    print(f"[INFO] Found {one_step_counter} trajectories with only 1 step each (out of {len(trajs)})")
    for traj in trajs:
        # if traj_counter >= 20:
        #     break
        traj_counter += 1
        if env_id is not None and traj["env_id"] != env_id:
            continue

        had_collision = traj.get("collision", False)
        #print(f"Trajectory {traj_counter} | Had collision: {had_collision}")
        success = traj.get("success", False)
        #print(f"Trajectory {traj_counter} | Success: {success}")
        if success and not had_collision:
            pure_success += 1
        elif success and had_collision:
            collided_success += 1

        start, tumor, curve, collisions = extract_curve(traj)
        if start is None:
            continue
        
        # compare start to preop start
        path_idx = traj["path_idx"]
        diff = start - preop_start[path_idx]
        adjusted = preop_start[path_idx].copy()
        adjusted[1] += 0.0595
        dist = np.linalg.norm(diff)

        # print(f"Trajectory {traj_counter} | Start: {start}, Pre-op start: {preop_start[path_idx]}")
        # print(f"Adjusted pre-op start: {adjusted}")
        # print(f"Difference vector: {diff}, Euclidean distance: {dist:.6f}")

        # if not np.allclose(start, preop_start[path_idx], atol=1e-6):
        #     print(f"❌ Traj start {start} differs from pre-op start {preop_start[path_idx]} (path_idx {path_idx})")
        # else:
        #     print(f"✅ Traj start matches pre-op start (path_idx {path_idx})")
        # --- plotting ---
        if had_collision:
            # gray trajectory
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="gray",
                    label=f"Env {traj['env_id']} Path {traj['path_idx']} (collision)")
            # mark collision points
            if collisions is not None:
                ax.scatter(collisions[:, 0], collisions[:, 1], collisions[:, 2],
                           c="red", marker="x", s=50, label="collision points")
        else:
            # clean trajectory
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                    label=f"Env {traj['env_id']} Path {traj['path_idx']}")
            ax.scatter(start[0], start[1], start[2], c="g", marker="o", s=40)
            ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], c="r", marker="^", s=50)

        # tumor and line
        ax.plot([start[0], tumor[0]], [start[1], tumor[1]], [start[2], tumor[2]],
                linestyle="--", color="orange", alpha=0.7)
        ax.scatter(tumor[0], tumor[1], tumor[2], c="black", marker="x", s=60)

        # print per-trajectory stats
        coll_steps = traj.get("collision_steps", 0)
        
        print(f"Env {traj['env_id']} Path {traj['path_idx']} | "
                f"Collision steps: {coll_steps} / {len(traj['steps'])}")

    total = traj_counter
    success_rate = (pure_success / total) * 100 if total > 0 else 0

    print("\n[SUMMARY]")
    print(f"Total trajectories: {total}")
    print(f"  Collision-free successes: {pure_success}")
    print(f"  Successes with ≥1 collision: {collided_success}")
    print(f"  Failures: {total - pure_success - collided_success}")
    print(f"Success rate (pure collision-free): {success_rate:.2f}%")

    ax.set_title(f"Tumor Trajectories (Success Rate: {success_rate:.2f}%)")
    plt.tight_layout()
    #plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_pickle", required=True, help="Trajectories pickle (from play script)")
    ap.add_argument("--env_id", type=int, default=None, help="Filter by env_id (optional)")
    ap.add_argument("--preop_pickle", default=None, help="original dataset pickle to overlay straight line (optional)")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj_pickle)
    preop = load_preop(args.preop_pickle)
    preop_start = preop[args.env_id]["start_positions"]  # [path_idx]
    #print(f"[INFO] Loaded pre-op start position for Env {args.env_id}: {preop_start}")
    print(f"[INFO] Loaded {len(trajs)} trajectories")

    if args.env_id is not None:
        trajs = [t for t in trajs if t["env_id"] == args.env_id]
    if len(trajs) == 0:
        print("[WARN] No trajectories to plot")
        return

    plot_scene(trajs, preop_start=preop_start, env_id=args.env_id)


if __name__ == "__main__":
    main()
