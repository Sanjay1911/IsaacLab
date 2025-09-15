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
SAVE = False  # whether to save the pickle of trajectories
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

    def save_trajectories():
        # finalize partial episodes
        for eid, rec in list(buffers.items()):
            if len(rec.get("steps", [])) > 0:
                completed.append({
                    "env_id": eid,
                    "path_idx": rec.get("path_idx", -1),
                    "steps": rec["steps"],
                    "success": False,  # partial; not terminal
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
        nonlocal buffers, completed, step_counter

        if not isinstance(info, dict) or "env_ids" not in info or "active_path_index" not in info:
            return

        env_ids = info["env_ids"]             
        path_idx = info["active_path_index"]  
        success = info.get("success", torch.zeros_like(env_ids, dtype=torch.bool))
        truncated = info.get("truncated", torch.zeros_like(env_ids, dtype=torch.bool))
        reward_collision = info.get("reward_collision", torch.zeros_like(env_ids, dtype=torch.float32))

        env_ids_np = to_cpu_np(env_ids).astype(np.int64)
        path_idx_np = to_cpu_np(path_idx).astype(np.int64)
        succ_np = to_cpu_np(success).astype(bool)
        trunc_np = to_cpu_np(truncated).astype(bool)
        coll_np = to_cpu_np(reward_collision).astype(float)

        B = env_ids_np.shape[0]
        for i in range(B):
            eid = int(env_ids_np[i])
            pid = int(path_idx_np[i])

            # If this env switches to a new path while an episode is open, close the old one first
            if (eid not in buffers) or (buffers[eid]["path_idx"] != pid and len(buffers[eid]["steps"]) > 0):
                prev = buffers.get(eid)
                if prev and len(prev["steps"]) > 0:
                    completed.append({
                        "env_id": eid,
                        "path_idx": prev["path_idx"],
                        "steps": prev["steps"],
                        "success": False,  # mid-switch, treat as non-terminal
                        "collision": bool(prev.get("had_collision", False)),
                        "collision_steps": int(prev.get("collision_steps", 0)),
                    })
                # start a new buffer for this (env, path)
                buffers[eid] = {"path_idx": pid, "steps": [], "had_collision": False, "collision_steps": 0}

            # Ensure buffer exists and has the tracking fields
            if eid not in buffers:
                buffers[eid] = {"path_idx": pid, "steps": [], "had_collision": False, "collision_steps": 0}

            # Build per-step snapshot for this env row
            snap = {"t": step_counter}
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        snap[k] = to_cpu_np(v)
                    elif v.size(0) == B:
                        snap[k] = to_cpu_np(v[i])
                    else:
                        snap[k] = to_cpu_np(v)  # unusual shape; store as-is
                else:
                    snap[k] = v
            buffers[eid]["steps"].append(snap)

            # --- collision accumulation across the episode ---
            # Treat any step with reward_collision == -50 as a direct hit
            if coll_np[i] <= -49.5:  # tolerance for fp
                buffers[eid]["had_collision"] = True
                buffers[eid]["collision_steps"] += 1

            print(f" Env {eid} Path {pid} Step {step_counter} | reward_collision={coll_np[i]:.1f}")

            # --- on terminal (success or truncated), finalize this episode ---
            if succ_np[i] or trunc_np[i]:
                completed.append({
                    "env_id": eid,
                    "path_idx": buffers[eid]["path_idx"],
                    "steps": buffers[eid]["steps"],
                    "success": bool(succ_np[i]),
                    "collision": bool(buffers[eid]["had_collision"]),
                    "collision_steps": int(buffers[eid]["collision_steps"]),
                })
                # clear for next episode on this env (path may or may not change)
                buffers[eid] = {"path_idx": pid, "steps": [], "had_collision": False, "collision_steps": 0}


    # ---- rollout loop ----
    obs, _ = env.reset()
    try:
        while simulation_app.is_running():
            t0 = time.time()
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, info = env.step(actions)

            # log
            append_step(info)
            step_counter += 1

            #print(f"INFO KEYS: {info.keys()}")
            
            #print(f" Success: {info.get('success')} | Truncated: {info.get('truncated')}")
            if args_cli.video and step_counter >= args_cli.video_length:
                break

            # real-time pacing
            if args_cli.real_time:
                sleep_time = dt - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        pass  # atexit

if __name__ == "__main__":
    main()
