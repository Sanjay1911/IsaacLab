import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

R_n = 0.001  # needle radius (meters) for OD-aware deviation

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

    it = ds.items() if isinstance(ds, dict) else enumerate(ds)
    out = {}
    for k, env_data in it:
        env_id = int(k)

        if "start_pose" not in env_data or "tumor_centroid" not in env_data:
            raise KeyError(f"Entry {env_id} missing 'start_pose' or 'tumor_centroid'")

        starts = env_data["start_pose"]  # list of dicts
        pos_list, quat_list = [], []
        for p in starts:
            if isinstance(p, dict):
                if "position" in p:
                    pos = np.asarray(p["position"], dtype=float).reshape(-1)
                    if pos.shape[0] == 3:
                        pos_list.append(pos)
                if "quaternion" in p:
                    q = np.asarray(p["quaternion"], dtype=float).reshape(-1)
                    if q.shape[0] == 4:
                        quat_list.append(q)
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
    """Return (chosen_start, tumor_centroid, tip_curve, collision_points, per_episode_stats)"""
    steps = traj.get("steps", [])
    if len(steps) == 0:
        return None, None, None, None, None

    # centerline deviations from pickle
    dev_center = np.asarray([float(np.asarray(s.get("deviation_t", 0.0))) for s in steps], dtype=float)
    # OD-aware: outer-envelope (centerline + radius). If you prefer surface clearance, use np.maximum(0, dev_center - R_n).
    dev_od = dev_center + R_n

    # per-episode stats on OD-aware deviation (paper-friendly)
    ep_mean = float(np.mean(dev_od))
    ep_min  = float(np.min(dev_od))
    ep_max  = float(np.max(dev_od))
    ep_stats = {"mean": ep_mean, "min": ep_min, "max": ep_max}

    first = steps[0]
    path_idx = int(first["active_path_index"])
    start_positions = np.asarray(first["start_positions"])
    tumor_centroid = np.asarray(first["tumor_centroids"])
    print(f"Tumor Centroid: {tumor_centroid}")
    chosen_start = start_positions[path_idx]

    # tip trajectory
    curve = np.stack([np.asarray(s["tooltip_pos"]) for s in steps])

    # collision points (reward_collision == -50)
    coll_pts = []
    for s in steps:
        val = float(np.asarray(s.get("reward_collision", 0.0)))
        if abs(val + 50.0) < 1e-3:  # treat -50 as direct hit
            coll_pts.append(np.asarray(s["tooltip_pos"]))
    coll_pts = np.array(coll_pts) if len(coll_pts) > 0 else None

    return chosen_start, tumor_centroid, curve, coll_pts, ep_stats


def iqr(x):
    if len(x) == 0:
        return 0.0
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def summarize_paths(path_stats):
    """Print a compact per-path table: episodes, successes, collision-free successes, collided successes, failures,
       and median [IQR] of episode mean/min/max OD-aware deviations (mm)."""
    if not path_stats:
        print("[WARN] No per-path stats to summarize.")
        return

    print("\n[PER-PATH DEVIATION SUMMARY] (OD-aware; mm)")
    header = (
        f"{'Path':>4} | {'N':>3} | {'Succ':>4} | {'CF':>3} | {'Col':>3} | {'Fail':>4} || "
        f"{'Mean_med[IQR] mm':>18} | {'Max_med[IQR] mm':>17} | {'Min_med[IQR] mm':>17}"
    )
    print(header)
    print("-" * len(header))

    for p in sorted(path_stats.keys()):
        ps = path_stats[p]
        N = ps["episodes"]
        succ = ps["succ_total"]
        cf   = ps["succ_cf"]
        col  = ps["succ_collided"]
        fail = ps["fail_total"]

        mean_list = np.asarray(ps["mean_list"])
        max_list  = np.asarray(ps["max_list"])
        min_list  = np.asarray(ps["min_list"])

        # convert to mm
        mean_mm = 1e3 * mean_list
        max_mm  = 1e3 * max_list
        min_mm  = 1e3 * min_list

        if len(mean_mm) > 0:
            mean_med = float(np.median(mean_mm))
            mean_iqr = iqr(mean_mm)
        else:
            mean_med = mean_iqr = 0.0

        if len(max_mm) > 0:
            max_med = float(np.median(max_mm))
            max_iqr = iqr(max_mm)
        else:
            max_med = max_iqr = 0.0

        if len(min_mm) > 0:
            min_med = float(np.median(min_mm))
            min_iqr = iqr(min_mm)
        else:
            min_med = min_iqr = 0.0
        # write data to csv
        with open("path_stats.csv", "a") as f:
            f.write(f"{p},{N},{succ},{cf},{col},{fail},{mean_med},{mean_iqr},{max_med},{max_iqr},{min_med},{min_iqr}\n")
        print(f"{p:>4} | {N:>3} | {succ:>4} | {cf:>3} | {col:>3} | {fail:>4} || "
              f"{mean_med:6.2f}[{mean_iqr:4.2f}] | {max_med:6.2f}[{max_iqr:4.2f}] | {min_med:6.2f}[{min_iqr:4.2f}]")


def plot_scene(trajs, preop_start, env_id=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # overall counters
    pure_success = 0
    collided_success = 0
    traj_counter = 0

    # per-path accumulators
    # path_stats[p] = {
    #   "episodes": int, "succ_total": int, "succ_cf": int, "succ_collided": int, "fail_total": int,
    #   "mean_list": [], "min_list": [], "max_list": []
    # }
    path_stats = {}
    collision_steps = {"one": 0, "two": 0, "more": 0}
    print(f"[INFO] Loaded {len(trajs)} trajectories")
    for traj in trajs:
        traj_counter += 1
        if env_id is not None and traj["env_id"] != env_id:
            continue

        p = int(traj["path_idx"])
        ps = path_stats.setdefault(p, {
            "episodes": 0, "succ_total": 0, "succ_cf": 0, "succ_collided": 0, "fail_total": 0,
            "mean_list": [], "min_list": [], "max_list": []
        })
        ps["episodes"] += 1

        had_collision = bool(traj.get("collision", False))
        success = bool(traj.get("success", False))
        if success and not had_collision:
            pure_success += 1
            ps["succ_total"] += 1
            ps["succ_cf"] += 1
        elif success and had_collision:
            collided_success += 1
            ps["succ_total"] += 1
            ps["succ_collided"] += 1
        else:
            ps["fail_total"] += 1

        start, tumor, curve, collisions, ep_stats = extract_curve(traj)
        if start is None or curve is None or ep_stats is None:
            continue

        # accumulate per-episode OD-aware deviation stats for this path
        ps["mean_list"].append(ep_stats["mean"])
        ps["min_list"].append(ep_stats["min"])
        ps["max_list"].append(ep_stats["max"])

        # --- plotting ---
        if had_collision:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="gray",
                    label=f"Env {traj['env_id']} Path {traj['path_idx']} (collision)")
            if collisions is not None and len(collisions) > 0:
                ax.scatter(collisions[:, 0], collisions[:, 1], collisions[:, 2],
                           c="red", marker="x", s=50, label="collision points")
        else:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                    label=f"Env {traj['env_id']} Path {traj['path_idx']}")
            ax.scatter(start[0], start[1], start[2], c="g", marker="o", s=40)
            #ax.scatter(curve[-1, 0], curve[-1, 2], curve[-1, 2], c="r", marker="^", s=50)  # keep your end marker

        # # pre-op straight line for context
        # ax.plot([start[0], tumor[0]], [start[1], tumor[1]], [start[2], tumor[2]],
        #         linestyle="--", color="orange", alpha=0.7)
        # ax.scatter(tumor[0], tumor[1], tumor[2], c="black", marker="x", s=60)

        # per-trajectory collision step summary (optional console)
        coll_steps = int(traj.get("collision_steps", 0))
        print(f"Env {traj['env_id']} Path {traj['path_idx']} | Collision steps: {coll_steps} / {len(traj['steps'])}")

        if success and coll_steps == 1:
            collision_steps['one'] += 1
        elif success and coll_steps == 2:
            collision_steps['two'] += 1
        elif success and coll_steps > 2:
            collision_steps['more'] += 1

    total = traj_counter
    print("\n[SUMMARY]")
    print(f"Total trajectories: {total}")
    print(f"  Collision-free successes: {pure_success}")
    #print(f"  Successes with ≥1 collision: {collided_success}")
    failure = total - pure_success
    print(f"  Failures(Direct Collision/Timeout/Overshoot): {failure}")
    success_rate = (pure_success / (total)) * 100 if total > 0 else 0.0
    print(f"Success rate (pure collision-free): {success_rate:.2f}%")
    print(f"Collision steps distribution (only for collided episodes): {collision_steps}")

    # per-path deviation summary (OD-aware; mm)
    summarize_paths(path_stats)

    ax.set_title(f"Tumor Trajectories (Collision-free SR: {success_rate:.2f}%)")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_pickle", required=True, help="Trajectories pickle (from play script)")
    ap.add_argument("--env_id", type=int, default=None, help="Filter by env_id (optional)")
    ap.add_argument("--preop_pickle", default=None, help="original dataset pickle to overlay straight line (optional)")
    args = ap.parse_args()

    trajs = load_trajectories(args.traj_pickle)
    preop = load_preop(args.preop_pickle) if args.preop_pickle else {}
    preop_start = None
    if args.env_id is not None and args.env_id in preop:
        preop_start = preop[args.env_id]["start_positions"]  # not strictly needed for this summary

    if args.env_id is not None:
        trajs = [t for t in trajs if t["env_id"] == args.env_id]
    if len(trajs) == 0:
        print("[WARN] No trajectories to plot")
        return

    plot_scene(trajs, preop_start=preop_start, env_id=args.env_id)


if __name__ == "__main__":
    main()
