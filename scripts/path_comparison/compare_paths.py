import numpy as np
import torch
from pxr import Usd, UsdGeom
from isaaclab.utils.io import load_pickle, dump_pickle


tumor_positions = []
tumor_quaternions = []
tumor_centroids = []
scored_paths = []
top_paths = []
# Read Tumor Dataset Pickle and get entry points and start poses
data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_2205.pkl")    
for i in range(len(data)):
    print(f"[INFO] Loading data for env {i}")
    try:
        env_data = data[i]
        for key in  ["tumor_position", "tumor_quat", "tumor_centroid", "entry_points", "top_entry_points", "start_pose"]:
            assert key in env_data, f"[ERROR] Missing key '{key}' in entry {i}"
        tumor_positions.append(env_data["tumor_position"])
        tumor_quaternions.append(env_data["tumor_quat"])
        tumor_centroids.append(env_data["tumor_centroid"])
        scored_paths.append(env_data["entry_points"])
        top_paths.append(env_data["top_entry_points"])
    except KeyError as e:
        print(f"[ERROR] ENV {i}: Missing key {e}")
        continue
    except AssertionError as e:
        print(f"[ERROR] ENV {i}: {e}")
        continue
    except Exception as e:
        print(f"[ERROR] ENV {i}: Failed to load data: {e}")
        continue

print("Top Paths Pre: ", top_paths[0], len(top_paths[0]))

top_paths_post = []
data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/tumor_dataset_post_new_0606.pkl")    
for i in range(len(data)):
    print(f"[INFO] Loading data for env {i}")
    try:
        env_data = data[i]
        for key in ["tumor_position", "tumor_quat", "tumor_centroid","top_entry_points"]:
            assert key in env_data, f"[ERROR] Missing key '{key}' in entry {i}"
        top_paths_post.append(env_data["top_entry_points"])
    except KeyError as e:
        print(f"[ERROR] ENV {i}: Missing key {e}")
        continue
    except AssertionError as e:
        print(f"[ERROR] ENV {i}: {e}")
        continue
    except Exception as e:
        print(f"[ERROR] ENV {i}: Failed to load data: {e}")
        continue

print("Top Paths Post: ", len(top_paths_post[0]))
print(top_paths_post)
print("\n🔍 Comparing pre vs post-deformation scores...\n")

num_envs = len(top_paths)
for env_id in range(num_envs):
    print(f"[ENV {env_id}]")
    pre_list = top_paths[env_id]
    post_list = top_paths_post[env_id]

    assert len(pre_list) == len(post_list), f"[ERROR] Env {env_id}: Length mismatch in top paths."

    for idx, (pre, post) in enumerate(zip(pre_list, post_list)):
        ep_pre = np.array(pre["entry_point"])
        ep_post = np.array(post["entry_point"])

        # ✅ Assert entry points are identical (within tolerance)
        assert np.allclose(ep_pre, ep_post, atol=1e-6), f"[ERROR] Env {env_id} Path {idx}: Entry points do not match"

        score_pre = pre["score"]
        print(f"  Path {idx}: Pre-entry point = {ep_pre}, Score = {score_pre}")
        score_post = post["score_post"]
        print(f"  Path {idx}: Post-entry point = {ep_post}, Score = {score_post}")

        if score_post is None:
            print(f"[WARN] Env {env_id} Path {idx}: No post-deformation score found")
            continue

        delta = score_post - score_pre
        print(f"  Path {idx}: Pre = {score_pre:3}, Post = {score_post:3}, Δ = {delta:+}")
    print()

