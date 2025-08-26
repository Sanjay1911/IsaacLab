
# NVIDIA IsaacLab Training Guide - Master Thesis Sanjayyadav Tamilselvam

This readme contains setup instructions, file structures, and training commands for running reinforcement learning experiments using **NVIDIA IsaacLab**.

---

## 📂 Project Setup

1. Navigate to the base project directory:

   ```bash
   cd /home/sanjay_isaac
   ```

2. This folder contains three sub-folders:

   * **env\_isaaclab1** → Python virtual environment
   * **forked/IsaacLab** → Main folder containing IsaacLab
   * **Others**

3. Open the main folder in **VS Code**:

   ```bash
   code /home/sanjay_isaac/forked/IsaacLab
   ```

   The terminal will automatically activate the Python virtual environment.

4. In VS Code, you will see associated project folders in the sidebar.

---

## 📁 Important Folders for Training

* **logs/** - Stores the logs for each training session. Folder where the logs are saved can be defined in below file under **experiment-->directory**
    ```bash
      /home/czlocal/sanjay_isaac/forked/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/biopsy_skrl/agents/skrl_dict_multi_discrete_ppo_cfg.yaml
    ```

* **source/isaaclab\_tasks/isaaclab\_tasks/direct/biopsy\_skrl/** - Contains the main scripts for training the biopsy task.

* **scripts/reinforcement\_learning/skrl/** - Includes the reinforcement learning library scripts for training and playing.

---

## 🔀 Git Branch

Ensure you are on the correct branch:

```bash
git checkout path-planner
git pull
```

---

## ▶️ Training Commands

IsaacLab has a detailed [documentation for training](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/index.html#) RL tasks.

Use the following commands to start training:

1. **Train with visual rendering (faster, no video logs):**

   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
   --task Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0 \
   --seed 42 \
   --num_envs 16
   ```

2. **Train with video recording:**

   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
   --task Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0 \
   --seed 42 \
   --num_envs 32 \
   --video \
   --video_length 750 \
   --video_interval 1500
   ```

* **Task:** Task name is registered with gym and can be found in the below file. 
    ```bash 
    /home/czlocal/sanjay_isaac/forked/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/biopsy_skrl/__init__.py
    ``` 
* **Command (1):** Trains with visual rendering only (faster).
* **Command (2):** Trains and stores videos every **1500 steps** with a video length of **750 steps** (configurable).

---

## 📂 Main Folder Structure

```bash
    ├── agents
    │   ├── __init__.py
    │   └── skrl_dict_multi_discrete_ppo_cfg.yaml
    ├── biopsy_env_cfg.py
    ├── biopsy_env.py
    ├── biopsy_skrl.py
    ├── __init__.py
```

* **agents/** → Contains the main YAML configs for training (usually unchanged).
* **biopsy\_env.py** → Core file containing the reinforcement learning pipeline and helper functions.
* **biopsy\_env\_cfg.py** → Contains the observation and action space configurations.

---

## How it works? (not for faint hearted people)

### BiopsyDirectEnv – Execution Flow (ASCII Diagram)

Below is a compact, **markdown + ASCII** flow to show how actions flow through the environment → motion → sensing → observations → rewards → termination.

```
┌────────────────────────────────────────────────────────────────────┐
│                 1) ENV INIT (__init__, _setup_scene)               │
│  - Load assets: ground, light, Vessel.usd, tumor.obj, needle       │
│  - Load pickles: tumor dataset & brain shift data                  │
│  - Build start poses & path priors; set sensors (raycasters)       │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│     2) STEP: _pre_physics_step(actions from policy/agent)          │
│  - Decode action → [insertion_depth, roll/twist] (space-aware)     │               │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│             3) APPLY ACTION: _apply_action                         │
│  - Constant-curvature update: generate_needle_step_ludwig()        │
│  - Compose new SE(3) pose; write pose/vel to sim                   │
│  - (Optional) draw markers, paths for visual debug                 │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│           4) SENSING / GEOMETRY                                    │
│  - Raycasts sample Vessel surface near needle tip                  │
│  - boundary_check_vessel(): cylinder filter + sector danger bins   │
│  - calc_normalized_progress(): path progress, deviation, distances │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                5) OBS: _get_observations                           │
│  - Build dict:                                                     │
│    • current_action (twist)                                        │
│    • normalized_depth_t, normalized_depth_t_ndt                    │
│    • deviation_t, deviation_t_ndt                                  │
│    • signed_delta_x, signed_delta_z (needle-frame offsets)         │
│    • heading_x, heading_y, heading_z                               │
│    • danger_bins (16 sectors)                                      │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                6) REWARD: _get_rewards / compute_reward            │
│  - + progress toward tumor (depth delta)                           │
│  - + low deviation (path alignment)                                │
│  - + tumor reached (threshold)                                     │
│  - ± collision risk via danger_bins (hard/soft penalties/bonus)    │
│  - action smoothness via previous twist bin                        │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                7) DONE: _get_dones                                 │
│  - Success: tip-to-tumor ≤ threshold                               │
│  - Truncate: timeout or overshoot outside segment range            │
└───────────────┬────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                8) RESET: _reset_idx                                │
│  - Sample new start pose index; reset buffers & bins               │
│  - Reposition needle; (optionally) re-randomize tumor pose         │
└────────────────────────────────────────────────────────────────────┘
```

## Quick Where-to-Look Map 

* **Nothing moves / NaNs in pose** → Check `_apply_action` and `generate_needle_step_ludwig()` (action scaling, kappa, insertion depth).
* **Obs all zeros / missing keys** → `_get_observations`, `calc_normalized_progress()` (path\_direction, closest\_point).
* **Rewards explode or flat** → `compute_reward()` (deviation units mm vs m; danger\_bins shape; previous\_twist\_action).
* **No collision feedback** → `boundary_check_vessel()` and `visualize_vessel_danger()` (ray hits valid mask, cylinder filter radii/height).
* **Assets not found** → `MinimalSceneCfg` file paths (`Vessels.usd`, `tumor.obj`) and pickle paths in `__init__`.

## Key I/O at a Glance 

* **`_pre_physics_step(actions)`**
  **Args:** `actions (B, ·)` from policy (Box/Discrete/MultiDiscrete).
  **Returns:** None; sets `self.actions → [insertion_depth, twist_rad]`.

* **`_apply_action()`**
  **Args:** Uses `self.actions`, current root pose.
  **Returns:** None; writes next pose/vel to sim; updates debug visuals.

* **`calc_normalized_progress()`**
  **Args:** Uses start pose, tumor centroid, current tip pose.
  **Returns:** `normalized_progress, deviation, perpendicular_vec, closest_point, path_length, d_tip_tumor, d_start_tumor, projected_dist`.

* **`boundary_check_vessel()`**
  **Args:** Ray hits from `raycast_vessel`, current pose.
  **Returns:** `(B,16)` danger bins (0–1), higher means riskier sector.

* **`_get_observations()`**
  **Args:** Internal cached geometry + danger bins.
  **Returns:** Dict for policy: action, progress/deviation (t & t-Δt), signed offsets, heading, danger bins.

* **`compute_reward(progress_t, progress_t_dt, deviation_t, action, d_tip_tumor, danger_bins)`**
  **Args:** Current/prev progress, deviation, twist, distances, danger.
  **Returns:** Scalar reward per env with weighted components & collision penalties.

* **`_get_dones()`**
  **Args:** Internals + thresholds.
  **Returns:** `(success_mask, truncated_mask)`.

--- 

## ⚠️ Critical File Paths

Some scripts may break if the following files are missing or paths are incorrect:

1. **Pre-op Path Pickle File**

   ```python
   self.tumor_pickle = load_pickle(
       "/home/czlocal/sanjay_isaac/forked/IsaacLab/custom/path_comparison/pickle_finale/rl_dataset_100envs.pkl"
   )
   ```

2. **Brain Shift Pickle File**

   ```python
   shift_data = load_pickle(
       "/home/czlocal/sanjay_isaac/forked/IsaacLab/custom/path_comparison/path_comparison/precomputed_brain_deformations100envs.pkl"
   )
   ```

3. **Vessel and Tumor Asset Configs** (in Class **MinimalSceneCfg**)

   ```python
   vessel = AssetBaseCfg(
       prim_path="{ENV_REGEX_NS}/Vessel",
       spawn=sim_utils.UsdFileCfg(
           usd_path="/home/czlocal/sanjay_isaac/forked/Vessels.usd"
       ),
       init_state=AssetBaseCfg.InitialStateCfg(
           pos=(0.01174, -0.0055, 0.195),
           rot=(0.70710, 0.70710, 0.0, 0.0),
       ),
   )

   tumor = AssetBaseCfg(
       prim_path="{ENV_REGEX_NS}/Tumor",
       spawn=sim_utils.MeshFileCfg(
           file_path="/home/czlocal/sanjay_isaac/forked/tumor.obj"
       ),
       init_state=AssetBaseCfg.InitialStateCfg(
           pos=(0.0, 0.0, 0.20),
           rot=(0.70710, 0.70710, 0.0, 0.0),
       ),
   )
   ```

---

## If you've any further questions about training / logging, please write to me at [Sanjay](sanjayyadav.tamil@gmail.com) or contact [Ludwig Haide](ludwig.haide@zeiss.com)