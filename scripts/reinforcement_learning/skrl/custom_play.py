# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework used for training the skrl agent.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm used for training the skrl agent.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_episodes", type=int, default=100, help="Stop after this many full episodes (success or truncated).")
parser.add_argument("--save", action="store_true", default=False, help="Save the trajectories to a pickle file at the end of play.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- rest ----------------
import os
import time
import pickle
import atexit
import signal
import sys
from datetime import datetime

import gymnasium as gym
import torch
import numpy as np
import skrl
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner  # pragma: no cover

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

SAVE = args_cli.save  # whether to save the pickle of trajectories
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install supported version using 'pip install skrl>={SKRL_VERSION}'")
    raise SystemExit(1)

algorithm = args_cli.algorithm.lower()

def to_cpu_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def main():
    # ---- load cfgs ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")
    resume_path = os.path.abspath(args_cli.checkpoint)

    # ---- env ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.getcwd(), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # ---- agent / runner ----
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # trajectory accumulators
    buffers = {}
    completed = []
    step_counter = 0
    episode_counter = 0

    def save_trajectories():
        for eid, rec in list(buffers.items()):
            if len(rec.get("steps", [])) > 0:
                completed.append({
                    "env_id": eid,
                    "path_idx": rec.get("path_idx", -1),
                    "steps": rec["steps"],
                    "success": False,
                    "collision": bool(rec.get("had_collision", False)),
                    "collision_steps": int(rec.get("collision_steps", 0)),
                })
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = "/home/sanjay/thesis_replications/forked/IsaacLab/logs/skrl/Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0/2025-08-08_17-13-56_ppo_torch/metrics"
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            out_path = os.path.join(out_path, f"agent_paths_{ts}.pkl")
        print(f"[INFO] Saving {len(completed)} trajectories to: {out_path}")
        try:
            if SAVE:
                with open(out_path, "wb") as f:
                    pickle.dump(completed, f)
            else:
                print("[WARN] Skipping pickle save; set SAVE=True to enable.")
        except Exception as e:
            print(f"[WARN] Failed to write pickle: {e}")
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass

    atexit.register(save_trajectories)

    def handle_signal(signum, frame):
        print(f"[INFO] Caught signal {signum}, saving trajectories...")
        save_trajectories()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # kill / window close

    # ---- per-step append helper ----
    def append_step(info):
        nonlocal buffers, completed, step_counter, episode_counter

        if not isinstance(info, dict) or "env_ids" not in info or "active_path_index" not in info:
            return

        env_ids = info["env_ids"]
        path_idx = info["active_path_index"]
        success = info.get("success", torch.zeros_like(env_ids, dtype=torch.bool))
        truncated = info.get("truncated", torch.zeros_like(env_ids, dtype=torch.bool))
        reward_collision = info.get("reward_collision", torch.zeros_like(env_ids, dtype=torch.float32))
        direct_collision = info.get("direct_collision", torch.zeros_like(env_ids, dtype=torch.float32))
        time_out = info.get("time_out", torch.zeros_like(env_ids, dtype=torch.float32))
        overshoot_positive = info.get("overshoot_positive", torch.zeros_like(env_ids, dtype=torch.float32))
        overshoot_negative = info.get("overshoot_negative", torch.zeros_like(env_ids, dtype=torch.float32))

        env_ids_np = to_cpu_np(env_ids).astype(np.int64)
        path_idx_np = to_cpu_np(path_idx).astype(np.int64)
        succ_np = to_cpu_np(success).astype(bool)
        trunc_np = to_cpu_np(truncated).astype(bool)
        coll_np = to_cpu_np(reward_collision).astype(float)
        B = env_ids_np.shape[0]
        for i in range(B):
            eid = int(env_ids_np[i])
            pid = int(path_idx_np[i])

            if eid not in buffers:
                buffers[eid] = {"path_idx": pid, "steps": [], "had_collision": False, "collision_steps": 0}

            snap = {"t": step_counter}
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        snap[k] = to_cpu_np(v)
                    elif v.size(0) == B:
                        snap[k] = to_cpu_np(v[i])
                    else:
                        snap[k] = to_cpu_np(v)
                else:
                    snap[k] = v
            buffers[eid]["steps"].append(snap)

            if coll_np[i] <= -49.5:
                buffers[eid]["had_collision"] = True
                buffers[eid]["collision_steps"] += 1

            if succ_np[i] or trunc_np[i]:
                episode_counter += 1

                # --- classify episode ---
                if bool(succ_np[i]) and not buffers[eid]["had_collision"] and not bool(direct_collision[i]):
                    status = "success"
                elif bool(direct_collision[i]):
                    status = "failure"
                else:
                    status = "truncated"

                completed.append({
                    "env_id": eid,
                    "path_idx": buffers[eid]["path_idx"],
                    "steps": buffers[eid]["steps"],
                    "success": (status == "success"),
                    "collision": bool(buffers[eid]["had_collision"]),
                    "collision_steps": int(buffers[eid]["collision_steps"]),
                    "time_out": bool(time_out[i]),
                    "overshoot_positive": bool(overshoot_positive[i]),
                    "overshoot_negative": bool(overshoot_negative[i]),
                    "direct_collision": bool(direct_collision[i]),
                    "status": status,   # NEW
                })

                print(f"[INFO] Episode {episode_counter} finished | Status={status} | "
                    f"Collision={bool(buffers[eid]['had_collision'])}, "
                    f"Collision steps={buffers[eid]['collision_steps']}")

                buffers[eid] = {"path_idx": pid, "steps": [], "had_collision": False, "collision_steps": 0}


    # ---- rollout loop ----
    obs, _ = env.reset()
    try:
        while simulation_app.is_running():
            if episode_counter >= args_cli.max_episodes:
                print(f"[INFO] Reached {episode_counter} episodes. Stopping.")
                break

            t0 = time.time()
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, info = env.step(actions)

            append_step(info)
            step_counter += 1

            if args_cli.video and step_counter >= args_cli.video_length:
                break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        num_success = sum(1 for t in completed if t["status"] == "success")
        num_failure = sum(1 for t in completed if t["status"] == "failure")
        num_trunc   = sum(1 for t in completed if t["status"] == "truncated")

        print(f"[INFO] Total episodes: {episode_counter}")
        print(f"[INFO] Success: {num_success}")
        print(f"[INFO] Failure (direct collision): {num_failure}")
        print(f"[INFO] Truncated (timeout/overshoot/other): {num_trunc}")

        # Optional: print details for debugging
        for t in completed:
            if t["status"] == "truncated":
                if t["time_out"]:
                    print(f"[INFO] Episode {t['env_id']} truncated due to timeout.")
                if t["overshoot_positive"]:
                    print(f"[INFO] Episode {t['env_id']} overshot positively.")
                if t["overshoot_negative"]:
                    print(f"[INFO] Episode {t['env_id']} overshot negatively.")


if __name__ == "__main__":
    main()
