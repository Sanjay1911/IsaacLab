# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Future imports for compatibility
from __future__ import annotations

# Standard libraries
import os, sys
import random
import traceback
from datetime import datetime

# Numerical & scientific computing
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import torch
torch.set_printoptions(profile="full")
import torch.nn.functional as F

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

# Gym and RL-related modules
import gymnasium as gym

# USD and Omniverse / IsaacSim core libraries
from pxr import UsdGeom, Gf
import omni.log

# Stage and prim management
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.cloner import GridCloner

# Transform utilities
from isaacsim.core.utils.torch.transformations import (
    tf_combine, tf_inverse, tf_vector
)

# IsaacLab core components
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg

# RL environment and scene configuration
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

# Assets and actuators
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject

# Sensors and raycasters
from isaaclab.sensors.ray_caster import (
    RayCasterCameraCfg, RayCasterCfg, patterns
)

# Spawning and asset management
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg

# IsaacLab utilities
from isaaclab.utils import configclass, convert_dict_to_backend
from isaaclab.utils.io import dump_pickle, load_pickle

# Math utilities
from isaaclab.utils.math import matrix_from_quat, skew_symmetric_matrix, quat_from_matrix, sample_uniform

# Visualization and markers
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Open3D for point cloud / mesh processing
import open3d as o3d
import matplotlib.pyplot as plt

@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    skull = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Skull",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/skull.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    vessel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Vessel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sanjay/thesis_replications/forked/Vessels.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    tumor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tumor",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/tumor.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    needle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        spawn=sim_utils.CylinderCfg(
            radius=0.002,
            height=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    raycast_camera_vessel = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0) ,convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.01,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_camera_tumor = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(-0.0012, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.01,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_vessel = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        attach_yaw_only=False,
        max_distance=0.01,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-180, 180], horizontal_fov_range=[-180, 180], horizontal_res=1.0
        )
    )

    raycast_tumor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        attach_yaw_only=False,
        max_distance=0.01,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-60, 60], horizontal_fov_range=[-20, 20], horizontal_res=1.0
        )
    )


@configclass
class BiopsyDirectEnvCfg(DirectRLEnvCfg):
    #env
    episode_length_s = 4.1666  # 250 timesteps
    decimation = 50
    action_space = 3
    observation_space = 23
    state_space = 0

    #simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    #scene
    scene: MinimalSceneCfg = MinimalSceneCfg(num_envs=1, env_spacing=0.5, replicate_physics=False)

    action_scale = 0.001
    dof_velocity_scale = 0.1

    # reward scales
    w_progress = 5.0
    w_deviation = 3.0
    w_collision = 1.5
    w_inside_tumor = 10.0
    w_action = 1.0  


class BiopsyDirectEnv(DirectRLEnv):
    """Direct RL environment for the biopsy task."""
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg : BiopsyDirectEnvCfg

    def __init__(self, cfg: BiopsyDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)
        
        if not self.cfg.viewer.headless:
            import omni.log
            omni.log.warn("Running in headless mode. No rendering will be performed.")
            from isaacsim.util.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.cloner = GridCloner(spacing=self.cfg.scene.env_spacing)
        self.offsets, _ = self.cloner.get_clone_transforms(self.num_envs)
        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/needle")
        import omni.log
        omni.log.info(f"Prim: {prim}")
        omni.log.info(f"Is valid: {prim.IsValid()}")
        #self.draw = _debug_draw.acquire_debug_draw_interface()
        try:
            self.U1 = torch.tensor(0.01, device=self.device, dtype=torch.float32)  # mm/s
            self.U2 = torch.tensor(0.1, device=self.device, dtype=torch.float32)  # mm/s
            self.REB = torch.tensor(0.001, device=self.device, dtype=torch.float32)  # mm
            self.PHI_Deg = torch.tensor(30, device=self.device, dtype=torch.float32)  # degrees
            self.PHI_Rad = torch.deg2rad(self.PHI_Deg)  # radians
            self.CURV = torch.tensor(0.2, device=self.device, dtype=torch.float32)  # mm
            self.dt_steer = torch.tensor(0.5, device=self.device, dtype=torch.float32)  # seconds
            self.INSERTION_DEPTH = torch.tensor(0.001, device=self.device, dtype=torch.float32)  # mm
            self.TUMOR_REACH_THRESHOLD = torch.tensor(0.0075, device=self.device, dtype=torch.float32)  # mm
            self.DIST_THRESHOLD = torch.tensor(0.005, device=self.device, dtype=torch.float32)  # mm
            self.K_DEV = torch.tensor(10.0, device=self.device, dtype=torch.float32)  # Deviation scaling factor
        except Exception as e:
            print("Error initializing constants:", e)

        self.insertion_lookup_table = {0: self.INSERTION_DEPTH, 1: self.INSERTION_DEPTH + 0.001, 2: self.INSERTION_DEPTH + 0.002}
        # read pickled data
        omni.log.info("Loading tumor data...")
        self.tumor_positions = []
        self.tumor_quaternions = []
        self.tumor_centroids = []
        self.scored_paths = []
        self.tumor_top_entry_points = []
        self.start_pose = []
        self.start_positions = []
        self.start_quaternions = []
        self.tumor_pickle = load_pickle("custom/path_comparison/pickle_finale/rl_dataset_10envs.pkl")  #/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_2205_cleaned.pkl
        for i in range(min(self.num_envs, len(self.tumor_pickle))):
            omni.log.info(f"Loading tumor data for env: {i}")
            try:
                env_data = self.tumor_pickle[i]
                for key in ["tumor_position", "tumor_quat", "tumor_centroid", "scored_paths", "top_entry_points", "start_pose"]:
                    assert key in env_data, f"[ERROR] Missing key '{key}' in entry {i}"
                self.tumor_positions.append(env_data["tumor_position"])
                self.tumor_quaternions.append(env_data["tumor_quat"])
                self.tumor_centroids.append(env_data["tumor_centroid"])
                self.scored_paths.append(env_data["scored_paths"])
                self.tumor_top_entry_points.append(env_data["top_entry_points"])
                self.start_pose.append(env_data["start_pose"])
                self.start_positions.append([pose["position"] for pose in env_data["start_pose"]])
                self.start_quaternions.append([pose["quaternion"] for pose in env_data["start_pose"]])
            except KeyError as e:
                omni.log.warn(f"KeyError: {e} for env {i}. Tumor data may be incomplete.")
            except AssertionError as e:
                omni.log.warn(f"AssertionError: {e} for env {i}. Tumor data may be incomplete.")
            except Exception as e:
                omni.log.warn(f"Exception: {e} for env {i}. Tumor data may be incomplete.")


        omni.log.info(f"Start positions count: {len(self.start_positions)}")
        omni.log.info("Tumor data for envs loaded successfully.")
        start_positions_np = np.array(self.start_positions)         # [N, 10, 3]
        tumor_centroids_np = np.array(self.tumor_centroids)  # [N, 3]
        offsets_np = np.array(self.offsets)                         # [N, 3]
        offset_to_skull = np.array([0.0, 0.049, 0.0])  # offset by 4.95 cm in y-axis
        offsets_expanded = offsets_np[:, None, :]                   # [N, 1, 3]
        start_positions_offset = start_positions_np + offsets_expanded + offset_to_skull  # [N, 10, 3]
        tumor_centroids_offset = tumor_centroids_np + offsets_np  # [N, 3]
        self.start_positions_tensor = torch.tensor(start_positions_offset, device=self.device, dtype=torch.float32)
        self.tumor_centroids_tensor = torch.tensor(tumor_centroids_offset, device=self.device, dtype=torch.float32)
        # Convert start poses to lookup tensors
        self.start_poses = torch.zeros((self.num_envs, len(self.start_positions[0]), 7), dtype=torch.float32, device=self.device)  # (envs, poses, 7)
        for i in range(self.num_envs):
            for j in range(len(self.start_positions[0])):
                self.start_poses[i, j, :3] = torch.tensor(self.start_positions[i][j], device='cuda:0', dtype=torch.float32)
                self.start_poses[i, j, 3:] = torch.tensor(self.start_quaternions[i][j], device='cuda:0', dtype=torch.float32)
        # for i in range(self.num_envs):
        #     print(f"Env {i}: unique poses = {len(set(tuple(p.tolist()) for p in self.start_poses[i]))}")
        # for i in range(1):  # check env 0
        #     for j in range(10):  # assuming 10 poses
        #         print(f"start_poses[{i}, {j}] = {self.start_poses[i, j]}")

        # add an offset to the start positions (tensor) so that its close to the skull
        
        if not self.cfg.viewer.headless:
            for env_id in range(self.num_envs):
                self.draw_points(self.start_positions_tensor[env_id, 0], color=(1.0, 0.0, 0.0, 1.0), size=5.0)

        assert len(self.tumor_positions) == self.num_envs, f"Number of tumor positions {len(self.tumor_positions)} does not match number of envs {self.num_envs}"
        self.shuffled_tumor_centroids = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.tooltip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tooltip_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.tumor = self.scene["tumor"]

        # active path index
        self.active_path_index = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)  # [B, 10]

        # Set needle to start position for each environment from pickle data
        # Assume: root_state shape is (num_envs, 13) → [x, y, z, qw, qx, qy, qz, ...]
        root_state = self.scene["needle"].data.root_state_w.clone()  # Clone once, outside loop
        for env_id in range(self.num_envs):
            try:
                start_pose = self.start_positions_tensor[env_id][0]  # Tensor [3]
                start_quat = torch.tensor(self.start_quaternions[env_id][0], device=self.device, dtype=torch.float32)  # Tensor [4]
                root_state[env_id, :3] = start_pose
                root_state[env_id, 3:7] = start_quat
                # print(f"Env {env_id} → Pose: {start_pose.cpu().numpy()}, Quat: {start_quat.cpu().numpy()}")
            except Exception as e:
                print(f"[ERROR] Failed to set needle for env {env_id}: {e}")

        self.scene["needle"].write_root_pose_to_sim(root_state[:, :7])
        self.scene["needle"].reset()
        print("Write successful for all environments.")

        # Sensors
        self.raycast_cam_tumor = self.scene["raycast_camera_tumor"]
        self.raycast_cam_vessel = self.scene["raycast_camera_vessel"]
        self.raycast_vessel = self.scene["raycast_vessel"]
        self.raycast_tumor = self.scene["raycast_tumor"]
        self.raycast_cam_vessel_max_distance = 0.001
        if not self.cfg.viewer.headless and self.cfg.viewer.save:
            import omni.replicator.core as rep
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"/home/sanjay/thesis_replications/forked/IsaacLab/custom/raycaster_output/{datetime_str}/{str(self.raycast_cam_vessel_max_distance)}"
            self.rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

        # Brain Shift 
        self.brain_shift_data = []
        shift_data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/path_comparison/pickle_finale/precomputed_brain_deformations_10envs.pkl")
        print(f"[INFO] Loaded brain shift data for {len(shift_data)} envs")
        for env_id in range(self.scene.num_envs):
            try:
                # omni.log.info(f"[INFO] Extracting top-1 shift steps for env {env_id}")
                env = shift_data[env_id]
                # omni.log.info("Shape of env:", len(env), "Number of entries:", len(env[0]))
                top_entry = env[0]  # top-ranked entry out of 10
                # omni.log.info("Shape of top entry:", len(top_entry))
                # omni.log.info("Top entry:", len(top_entry[0]))  # deformation steps
                self.brain_shift_data.append(top_entry[0])  # append the top entry's deformation steps
            except Exception as e:
                print(f"[ERROR] Failed to extract for env {env_id}: {e}")
        if not len(self.brain_shift_data):
            raise ValueError("Not enough shift steps in brain_shift_data.")
        
        try:
            print(f"[INFO] Brain shift data for envs: {len(self.brain_shift_data)}")
            self.get_vessel_points()
        except Exception as e:
            omni.log.error(f"Error getting vessel points: {e}")
            traceback.print_exc()

        self.stage = stage_utils.get_current_stage()
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.set_tumor_positions()

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        self.camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera"))

        # Observation space terms:
        self.pcd = o3d.geometry.PointCloud()  # Placeholder for point cloud data
        self.prev_normalized_progress = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_deviation = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_tooltip_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Rewards
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        self.metrics_log = {
            "distance": [[] for _ in range(self.num_envs)],
            "normalized_progress": [[] for _ in range(self.num_envs)],
            "deviation": [[] for _ in range(self.num_envs)],
            "projected_dist": [[] for _ in range(self.num_envs)],
            "path_length": [[] for _ in range(self.num_envs)],
        }


    def _setup_scene(self):
        """
        Setup the scene for the environment. But since InteractiveSceneCfg is used, the scene is already setup in the
        constructor of the parent class.

        To add new assets to the scene, use the `scene` attribute of the environment.
        refer https://forums.developer.nvidia.com/t/importing-scene-created-in-isaac-sim-to-isaac-lab/315590
        """
        self._needle = self.scene["needle"]
        self._tumor = self.scene["tumor"]

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process agent actions:
        - actions[:, :3]: delta position (x, y, z)
        - orientation remains fixed to preop base orientation
        """
        # print(f"[DEBUG-STEP] Pre-physics steps called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        self.actions = actions.clone() 
        # print(f"Actions received: {type(self.single_action_space)}")
        if isinstance(self.single_action_space, gym.spaces.Box):
            low = torch.tensor(self.single_action_space.low, device=self.device)
            high = torch.tensor(self.single_action_space.high, device=self.device)
            print(f"Picked actions: {self.actions}")
            # Scale action values to the range of the action space from [-1, 1] to [low, high] using the formula:
            # scaled_action = ((x-a)/(b-a)) * (d-c) + c where x belongs to [a, b] and scaled_action belongs to [c, d]
            # Here, a = -1, b = 1, c = low, d = high
            a, b = -1, 1  # Action space range
            # This scales the actions to the range [low, high]
            self.actions = (((self.actions - a) / (b - a)) * (high - low)) + low  # Normalize actions to [-1, 1]
            self.new_actions = 0.5 * (self.actions + 1.0) * (high - low) + low
            # self.actions = torch.clamp(self.actions, low, high)
            print(f"Scaled actions: {self.actions}, {self.new_actions}")
        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            print(f"Picked actions (Discrete): {self.actions}")
            # For discrete actions, we assume the action is an index into a lookup table
            insertion_depths = torch.full((self.num_envs,), self.INSERTION_DEPTH.item(), device=self.device, dtype=torch.float32)  # Default depth
            twist_angles_deg = self.actions.float() * 22.5  # 0.0 degrees for no twist
            twist_angles_rad = torch.deg2rad(twist_angles_deg)
            print(f"Insertion depths: {insertion_depths}")
            print(f"Twist angles (deg): {twist_angles_rad}")
            self.actions = torch.stack([insertion_depths.view(-1), twist_angles_rad.view(-1)], dim=1)

            print(f"Updated actions: {self.actions}")
        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            # print(f"Picked actions (MultiDiscrete): {self.actions}")
            insertion_bins = self.actions[:, 0]
            twist_bins = self.actions[:, 1]

            insertion_depths = torch.tensor(
                [self.insertion_lookup_table[i.item()] for i in insertion_bins],
                device=self.device,
                dtype=torch.float32
            )
            #insertion_depths = torch.full((self.num_envs,), self.INSERTION_DEPTH.item(), device=self.device, dtype=torch.float32)  # Default depth
            twist_angles_deg = twist_bins.float() * 22.5  # 0.0 degrees for no twist
            twist_angles_rad = torch.deg2rad(twist_angles_deg)
            # print(f"Insertion depths: {insertion_depths}")
            # print(f"Twist angles (deg): {twist_angles_deg}")
            # print(f"Twist angles (rad): {twist_angles_rad}")
            self.actions = torch.stack([insertion_depths, twist_angles_rad], dim=1)
            # print(f"Updated actions: {self.actions}")

    def _apply_action(self):
        # print(f"[DEBUG-STEP] Apply action called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        root_state = self._needle.data.root_state_w.clone()  
        new_root_state = root_state.clone()
        pos = root_state[:, :3]              
        quat = root_state[:, 3:7]           
        rot = matrix_from_quat(quat)         
        current_pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1) 
        current_pose[:, :3, :3] = rot
        current_pose[:, :3, 3] = pos    
        start_pose = self.get_start_pose_active(num_envs=self.num_envs)
        prior_paths = self.get_prior_paths(start_pose, self.tumor_centroids_tensor)
        insertion_depths = self.actions[:, 0]  
        twist_angles = self.actions[:, 1]     
        next_poses = []
        for i in range(self.num_envs):
            # direction vector (start → tumor)
            preop_direction = self.tumor_centroids_tensor[i] - start_pose[i]
            preop_direction = preop_direction / torch.norm(preop_direction)

            # Compute next pose
            next_pose = self.generate_needle_step_with_rebound(
                current_pose=current_pose[i],
                insertion_depth=insertion_depths[i],
                twist_angle_rad=twist_angles[i],
                prior_path=prior_paths[i]
            )
            next_poses.append(next_pose)

        next_poses = torch.stack(next_poses)

        new_pos = next_poses[:, :3, 3]
        new_rot = next_poses[:, :3, :3]
        new_quat = quat_from_matrix(new_rot)
        new_quat = torch.stack([new_quat[:, 3], new_quat[:, 0], new_quat[:, 1], new_quat[:, 2]], dim=-1)

        new_root_state[:, :3] = new_pos
        new_root_state[:, 3:7] = new_quat
        self._needle.write_root_pose_to_sim(new_root_state[:, :7])
        self._needle.write_root_velocity_to_sim(torch.zeros_like(new_root_state[:, 7:]))
        self._needle.reset()
        # print(f"[DEBUG-STEP] Action applied at time step: {self.common_step_counter}")
        if not self.cfg.viewer.headless:
            for i in range(self.num_envs):
                start = self.start_positions_tensor[i, self.active_path_index[i]]
                end = self.tumor_centroids_tensor[i]
                tip = new_pos[i]
                self.draw_lines(start, end, color="yellow")
                self.draw_points(tip, color=(0.2, 0.8, 0.2, 1.0), size=4.0)
            for i in range(self.num_envs):
                prior_positions = prior_paths[i][:, :3, 3]
                self.draw_points(prior_positions, color=(1.0, 0.0, 0.0, 1.0), size=2.0)


    def _compute_intermediate_values(self, env_ids):
        """
        Compute intermediate values for the environment. This includes computing the action to be applied to the robot
        and the observations to be returned to the agent.
        """
        #self.potentials[env_ids], self.prev_potentials[env_ids] = self.compute_intermediate_values(self.tooltip_pos, self.shuffled_tumor_centroids[env_ids], self.prev_potentials[env_ids])
        pass
    
    def reset_start_positions(self, env_ids):
        """
        Reset the start positions of the needle for the given environment IDs.
        """
        root_state = self._needle.data.root_state_w.clone()

        for env_id in env_ids:
            idx = self.active_path_index[env_id].item()
            pos = self.start_positions_tensor[env_id][idx]              # [3]
            quat = torch.tensor(self.start_quaternions[env_id][idx],    # [4]
                                device=self.device, dtype=torch.float32)

            root_state[env_id, :3] = pos
            root_state[env_id, 3:7] = quat
            self.tooltip_pos[env_id] = pos
            self.tooltip_rot[env_id] = quat

            print(f"[RESET] Env {env_id} start → Pose: {pos.cpu().numpy()}, Quat: {quat.cpu().numpy()}")

        self._needle.write_root_pose_to_sim(root_state[:, :7])
        self._needle.reset()

    def _reset_idx(self, env_ids):
        """
        Reset the environment index.
        a) Reset the tooltip position and rotation based on new start pose index from the pickle data.
        b) Reset the sensor and other buffers.
        c) TODO: Reset also brain shift data if needed.
        d) Reset the tumor poses using sample_uniform to add a small random offset.
        """
        # print(f"[DEBUG-STEP] Reset called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        super()._reset_idx(env_ids)
        # Recompute any intermediate buffers (like tooltip pos, etc.)
        self.active_path_index[env_ids] = torch.randint(
            high=10,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.int32
        )
        self.reset_start_positions(env_ids)

        # Inside reset() or just before _get_observations()
        insertion_depths = torch.full((self.num_envs,), self.INSERTION_DEPTH.item(), device=self.device)
        twist_angles_rad = torch.zeros((self.num_envs,), device=self.device)  # or any dummy twist
        self.actions = torch.stack([insertion_depths, twist_angles_rad], dim=1)  # Shape: [B, 2]
        # Reset tumor poses using sample_uniform
        tumor_world_poses = self.tumor.get_world_poses(env_ids)
        position, quaternion = tumor_world_poses
        omni.log.info(f"Initial tumor world poses: {tumor_world_poses}")
        try:
            omni.log.info(f"Resetting tumor poses for env_ids: {env_ids}, Tumor world poses: {tumor_world_poses[0]}")
            sampled_offset = sample_uniform(lower=-0.001, upper=0.001, size=(len(env_ids), 3), device=self.device)
            updated_position = position + sampled_offset
            # stack updated poses with quaternions from tumor_world_poses
            self.tumor.set_world_poses(positions=updated_position, orientations=quaternion, indices=env_ids)
            # print(f"[INFO] Tumor poses reset success for env_ids: {env_ids}")
        except Exception as e:
            omni.log.error(f"Error sampling tumor poses due to: {e}")
        omni.log.info(f"Calling _reset_idx for env_ids: {env_ids}")
        self._compute_intermediate_values(env_ids)

    def _get_dones(self):
        """
        Get the done flags for the environment. This includes:
        - Strong Collision with the vessels (TODO)
        - Time out if the episode length exceeds the maximum
        - Tooltip reaches the tumor
        """
        # print(f"[DEBUG-STEP] Dones called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        _, _, perpendicular_vector, path_length, d_ttip_tumor, d_start_tumor, projected_dist = self.calc_normalized_progress()
        tumor_reached = (d_ttip_tumor <= self.TUMOR_REACH_THRESHOLD).squeeze(-1)  # shape: (B,)
        # print(f"[DEBUG] Tumor reached: {tumor_reached}")
        time_out = (self.episode_length_buf >= self.max_episode_length - 1)  # already shape: (B,)
        crossed_path_length = (projected_dist > path_length) 
        overshoot = (d_ttip_tumor > d_start_tumor).squeeze(-1)  # shape: (B,)
        # print(f"[DEBUG] reached ? : {tumor_reached}, time out ? : {time_out}, crossed path length ? : {crossed_path_length}, overshoot ? : {overshoot}")
        truncated_condition = time_out | (crossed_path_length & overshoot)
        #for i in range(self.num_envs):
            #print("Saving plot")
            #self.save_episode_plot(env_id=i, step_id=self.common_step_counter)
        # if tumor_reached.any() or truncated_condition.any():
        #     for i in range(self.num_envs):
        #         #if tumor_reached[i].item() or truncated_condition[i].item():
        #             #self.save_episode_plot(env_id=i, step_id=self.common_step_counter)
        #         if tumor_reached[i].item():
        #             print(f"[INFO] Env {i} reached the tumor.")
        #         if truncated_condition[i].item():
        #             print(f"[INFO] Env {i} is truncated due to time out or overshoot.")
        return tumor_reached, truncated_condition

    def _get_observations(self):
        """
        Get the observations for the environment. This includes:
        - Tooltip Pose
        - Normalized Depth to Tumor at time-step t and t-dt
        - Direction to Tumor Centroid
        - Downsampled PCD from RayCast Sensor
        - Previous Action
        """
        # print(f"[DEBUG-STEP] Observations called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        normalized_progress, deviation, perpendicular_vector, path_length, d_ttip_tumor, d_start_tumor, projected_dist = self.calc_normalized_progress()
        normalized_progress_t_ndt = self.prev_normalized_progress.clone()
        prev_deviation_t_ndt = self.prev_deviation.clone()

        self.prev_normalized_progress = normalized_progress.detach()
        self.prev_deviation = deviation.detach()

        start_pose = self.get_start_pose_active(num_envs=self.num_envs)       
        prior_paths = self.get_prior_paths(start_pose, self.tumor_centroids_tensor)
        tip_positions = self.tool_tip_pos                                      
        target_idx, target_pose, next_pos = self.find_closest_path_index(tip_positions, prior_paths)
        path_vec = F.normalize(next_pos - target_pose[:, :3, 3], dim=-1)      
        u_y, u_z = self.compute_basis(path_vec) 
        signed_delta_y = torch.sum(perpendicular_vector * u_y, dim=-1, keepdim=True)  # scalar
        signed_delta_z = torch.sum(perpendicular_vector * u_z, dim=-1, keepdim=True)  # scalar
        omni.log.info(f"Signed Delta Y: {signed_delta_y}, Signed Delta Z: {signed_delta_z}")
        current_action = self.actions.clone()
        ttip_pos_t_ndt = self.prev_tooltip_pos.clone()             
        self.prev_tooltip_pos = self.tool_tip_pos.detach()          
        delta = self.tool_tip_pos - ttip_pos_t_ndt                   
        norms = torch.norm(delta, dim=-1, keepdim=True) + 1e-8       
        heading = delta / norms                                    
        heading = torch.where(norms > 1e-6, heading, torch.zeros_like(heading))
        heading_t = torch.sum(heading * path_vec, dim=-1, keepdim=True)  # scalar
        omni.log.info(f"Tooltip position at t: {self.tool_tip_pos}")
        omni.log.info(f"Tooltip position at t-dt: {ttip_pos_t_ndt}")
        omni.log.info(f"Heading at t: {heading_t}")
        heading_y = torch.sum(heading * u_y, dim=-1, keepdim=True)
        heading_z = torch.sum(heading * u_z, dim=-1, keepdim=True)
        omni.log.info(f"Heading Y: {heading_y}, Heading Z: {heading_z}")
        # for i in range(self.num_envs):
        #     self.metrics_log["normalized_progress"][i].append(normalized_progress[i].item())
        #     self.metrics_log["deviation"][i].append(deviation[i].item())
        #     self.metrics_log["distance"][i].append(d_ttip_tumor[i].item())
        #     self.metrics_log["projected_dist"][i].append(projected_dist[i].item())
        #     self.metrics_log["path_length"][i].append(path_length[i].item())

        # --- Tumor geometry ---
        # to_tumor_centroid = self.shuffled_tumor_centroids - self.tool_tip_pos ----> This can be used in reward calculation
        #obs = {"current_action": current_action, "normalized_depth_t": normalized_progress, "normalized_depth_t_ndt": normalized_progress_t_ndt, "deviation_t": deviation, "deviation_t_ndt": prev_deviation_t_ndt, "signed_delta_y": signed_delta_y, "signed_delta_z": signed_delta_z, "heading_y": heading_y, "heading_z": heading_z, "heading_t": heading_t}
        obs = {
            "current_action": self.actions[:, 1].unsqueeze(-1),  # Extract actual twist angle in radians
            "normalized_depth_t": normalized_progress.unsqueeze(-1),       # from [B] → [B, 1]
            "normalized_depth_t_ndt": normalized_progress_t_ndt.unsqueeze(-1),
            "deviation_t": deviation.unsqueeze(-1),
            "deviation_t_ndt": prev_deviation_t_ndt.unsqueeze(-1),
            "signed_delta_y": signed_delta_y,  # already [B, 1]
            "signed_delta_z": signed_delta_z,
            "heading_y": heading_y,
            "heading_z": heading_z,
            "heading_t": heading_t,
        }

        for k, v in obs.items():
            print(f"Observation {k}: {v}, type: {type(v)}")
            print(f"{k}: {v.shape}")

        return {"policy": obs}

    def _get_rewards(self):  # TODO: to get calculated Rewards
        """
        Get the rewards for the environment. This includes:
        a) RayCaster Camera reward based on distance to image plane and distance to camera

        """
        # print(f"[DEBUG-STEP] Rewards called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        obs = self._get_observations()["policy"]
        progress_t = obs["normalized_depth_t"]             
        progress_t_dt = obs["normalized_depth_t_ndt"]      
        deviation_t = obs["deviation_t"]                   
        deviation_t_dt = obs["deviation_t_ndt"]            
        action = obs["current_action"]                     
        total_reward = self.compute_reward(progress_t, progress_t_dt, deviation_t, deviation_t_dt, action)
        return total_reward

    def _get_states(self):  # TODO: States for Asymmetric RL
        pass

    #@torch.jit.script
    def compute_reward(self, progress_t, progress_t_dt, deviation_t, deviation_t_dt, action):  # TODO: actual Rewards
        """
        Reward structure:
        + reward for being closer to tumor
        - penalty for being close to vessels
        - penalty for high real-time collision score
        + bonus for being inside tumor
        """
        reward_progress = progress_t - progress_t_dt
        reward_deviation = torch.exp(-self.K_DEV * deviation_t ** 2)
        reward_reached_tumor = (progress_t < self.TUMOR_REACH_THRESHOLD)
        reward_action = torch.norm(action, dim=-1)  # L2 norm of the action vector
        reward = (
            (self.cfg.w_progress * reward_progress)
            + (self.cfg.w_deviation * reward_deviation)
            + (self.cfg.w_inside_tumor * reward_reached_tumor.float())
            - (self.cfg.w_action * reward_action)
        )
        reward = torch.clip(reward, min=-100.0, max=100.0)
        return reward  # ensure shape [B]

    # ## --------------------------------------- ## #
    # ## Additional Utility Functions for Visualization and Debugging ## #
    # ## --------------------------------------- ## #

    def get_start_pose_active(self, num_envs):
        """
        Returns the start pose for the active path index for each environment.
        """
        return torch.stack([
            self.start_positions_tensor[i, self.active_path_index[i]]
            for i in range(num_envs)
        ], dim=0)

    def get_prior_paths(self, start_pose, tumor_centroids):
        return [
            self.discretize_preop_path(start_pose[i], tumor_centroids[i])
            for i in range(self.num_envs)
        ]
        
    def compute_basis(self, path_vecs):  # [B, 3]
        ref = torch.tensor([0.0, 0.0, 1.0], device=path_vecs.device).expand_as(path_vecs)
        alt_ref = torch.tensor([0.0, 1.0, 0.0], device=path_vecs.device).expand_as(path_vecs)

        # Check where path_vec is too close to [0, 0, 1]
        is_collinear = torch.allclose(path_vecs, ref, atol=1e-2)
        ref[is_collinear] = alt_ref[is_collinear]

        u_y = torch.cross(path_vecs, ref, dim=-1)
        u_y = F.normalize(u_y, dim=-1)
        u_z = torch.cross(path_vecs, u_y, dim=-1)
        u_z = F.normalize(u_z, dim=-1)
        return u_y, u_z

    def find_closest_path_index(self, tip_positions, prior_paths):
        closest_indices = []
        target_poses = []
        next_positions = []

        for i, path in enumerate(prior_paths):
            path_positions = path[:, :3, 3]
            dists = torch.norm(path_positions - tip_positions[i], dim=1)
            idx = torch.argmin(dists)
            target_pose = path[idx]

            if idx < path.shape[0] - 1:
                next_pos = path[idx + 1][:3, 3]
            else:
                next_pos = path[idx - 1][:3, 3]

            closest_indices.append(idx)
            target_poses.append(target_pose)
            next_positions.append(next_pos)

        return (
            torch.tensor(closest_indices, device=self.device),
            torch.stack(target_poses),
            torch.stack(next_positions)
        )

    def save_episode_plot(self, env_id: int, step_id: int):
        log = self.metrics_log
        time_steps = range(len(log["distance"][env_id]))
        plt.figure(figsize=(12, 8))

        # Subplots
        # plt.subplot(2, 2, 1)
        # plt.plot(time_steps, log["distance"][env_id], label="Distance to Tumor")
        # plt.title("Distance to Tumor")
        # plt.xlabel("Timestep")
        # plt.ylabel("Distance (m)")
        # plt.grid(True)

        plt.subplot(2, 2, 1)
        plt.plot(time_steps, log["normalized_progress"][env_id], label="Normalized Progress", color='green')
        plt.title("Normalized Progress")
        plt.xlabel("Timestep")
        plt.ylim(0, 1.2)
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(time_steps, log["deviation"][env_id], label="Deviation", color='red')
        plt.title("Deviation from Path")
        plt.xlabel("Timestep")
        plt.grid(True)

        # plt.subplot(2, 2, 4)
        # plt.plot(time_steps, log["projected_dist"][env_id], label="Projected", color='orange')
        # plt.plot(time_steps, log["path_length"][env_id], label="Path Length", linestyle='--', color='gray')
        # plt.title("Progress vs Total Path")
        # plt.xlabel("Timestep")
        # plt.legend()
        # plt.grid(True)

        os.makedirs("/home/sanjay/thesis_replications/forked/IsaacLab/custom/plots", exist_ok=True)
        fname = f"/home/sanjay/thesis_replications/forked/IsaacLab/custom/plots/episode_env{env_id}_step{step_id}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def calc_normalized_progress(self):
        # Needle Kinematics
        root_state = self._needle.data.root_state_w.clone() 
        self.tool_tip_pos = root_state[:, :3]  
        self.tool_tip_rot = root_state[:, 3:7] 
        start_pose = torch.stack([
            self.start_positions_tensor[i, self.active_path_index[i]]
            for i in range(self.num_envs)
        ])
        # Tumor Position and Orientation  
        tumor_root_poses = self.tumor.get_world_poses(self.env_ids)  
        tumor_pos, tumor_quat = tumor_root_poses
        # Calculate normalized progress and deviation
        d_ttip_tumor = torch.norm(self.tool_tip_pos - tumor_pos, dim=-1)  # distance from tooltip to tumor
        d_startpos_tumor = torch.norm(start_pose - tumor_pos, dim=-1)  # distance from start position to tumor
        tooltip_vec = self.tool_tip_pos - start_pose  
        path_vector = tumor_pos - start_pose 
        path_direction = F.normalize(path_vector, dim=-1)  # normalized direction vector from start to tumor
        projected_dist = torch.sum(tooltip_vec * path_direction, dim=-1) 
        path_length = torch.norm(path_vector, dim=-1)
        normalized_progress = projected_dist / path_length
        normalized_progress = torch.clamp(normalized_progress, 0.0, 1.0)
        perpendicular_vec = tooltip_vec - (projected_dist.unsqueeze(-1) * path_direction)  # [B, 3]
        deviation = torch.norm(perpendicular_vec, dim=-1)
        omni.log.info(f"Deviation:{deviation}")
        omni.log.info(f"Normalized progress: {normalized_progress}, Deviation: {deviation}")
        omni.log.info(f"Tooltip position: {self.tool_tip_pos}, Tumor position: {tumor_pos},")
        omni.log.info(f"Shape of tumor positions: {tumor_pos.shape}, Tumor quaternion: {tumor_quat.shape}, tooltip: {self.tool_tip_pos.shape}")
        omni.log.info(f"Distance to tumor from tooltip: {d_ttip_tumor}, Distance from start position to tumor: {d_startpos_tumor}")
        return normalized_progress, deviation, perpendicular_vec, path_length, d_ttip_tumor, d_startpos_tumor, projected_dist

    def discretize_preop_path(self, start_pose, tumor_centroid, num_points=42):
        """
        Generate a discretized path from the start pose to the tumor centroid.
        """
        start_pos = start_pose.to(dtype=torch.float32, device=self.device)
        end_pos = tumor_centroid.to(dtype=torch.float32, device=self.device)
        path = linspace(start_pos, end_pos, num_points)  # torch.linspace only accepts 1D tensors hence refer to https://github.com/pytorch/pytorch/issues/61292, TorchScript does not support self in scripted functions — it expects a pure function, not a method of a class.
        path_poses = torch.eye(4, device=self.device).repeat(num_points, 1, 1)
        path_poses[:, :3, 3] = path
        return path_poses
    
    def twist_to_matrix(self, v, w):
        mat = torch.zeros((4, 4), device=self.device, dtype=torch.float32)
        mat[:3, :3] = skew_symmetric_matrix(w)
        mat[:3, 3] = v
        return mat

    def generate_needle_step_with_rebound(
        self,
        current_pose: torch.Tensor,            # SE(3), shape (4, 4)
        insertion_depth: float,                # mm
        twist_angle_rad: float,                # radians
        prior_path: torch.Tensor,              # shape (N, 4, 4)
        threshold_deg: float = 5.0            # deviation threshold in degrees
    ) -> torch.Tensor:
        """
        Apply one SE(3) twist-based step with optional rebound correction toward a soft path prior.

        Returns:
            torch.Tensor: Updated SE(3) pose (4x4)
        """
        # Twist kinematics
        phi = torch.deg2rad(torch.tensor(self.PHI_Deg, device=self.device, dtype=torch.float32))
        u2 = twist_angle_rad  # / self.dt_steer  # [rad/sec]

        # Twist vectors in local frame
        v_local = torch.tensor([0.0,
                                insertion_depth * torch.sin(phi),
                                -insertion_depth * torch.cos(phi)], device=self.device)

        w_local = torch.tensor([insertion_depth / self.CURV,
                                0.0,
                                u2], device=self.device)

        # Rotate to global frame
        R = current_pose[:3, :3]
        v = R @ v_local
        w = R @ w_local

        # Build se(3) matrix
        xi_hat = self.twist_to_matrix(v, w)


        # Find nearest point on prior path
        tip_pos = current_pose[:3, 3]
        prior_positions = prior_path[:, :3, 3]
        dists = torch.norm(prior_positions - tip_pos, dim=1)
        target_idx = torch.argmin(dists)
        target_pose = prior_path[target_idx]

        # Compute deviation angle
        current_dir = current_pose[:3, 2]
        to_target = target_pose[:3, 3] - tip_pos
        to_target = to_target / torch.norm(to_target)
        angle_diff = torch.acos(torch.clamp(torch.dot(current_dir, to_target), -1.0, 1.0))

        threshold_rad = torch.deg2rad(torch.tensor(threshold_deg, device=self.device))

        if angle_diff > threshold_rad:
            # Rebound correction
            t_mod = self.REB / insertion_depth
            g_partial = current_pose @ torch.linalg.matrix_exp(xi_hat * (1.0 - t_mod))

            # Small forward translation along z-axis
            z_axis = g_partial[:3, 2]
            trans = torch.eye(4, device=self.device)
            trans[:3, 3] = z_axis * self.REB

            # Rotation to realign
            cos_theta = torch.clamp(torch.dot(z_axis, to_target), -1.0, 1.0)
            theta = torch.acos(cos_theta)
            rotation_axis = torch.cross(z_axis, to_target)
            if torch.norm(rotation_axis) < 1e-6:
                rot_z = torch.eye(4, device=self.device)
            else:
                rotation_axis = rotation_axis / torch.norm(rotation_axis)
                skew = skew_symmetric_matrix(rotation_axis)
                rot_z = torch.eye(4, device=self.device)
                rot_z[:3, :3] = (
                    torch.eye(3, device=self.device) +
                    torch.sin(theta) * skew +
                    (1 - torch.cos(theta)) * (skew @ skew)
                )

            rot_z = torch.eye(4, device=self.device)
            rot_z[:3, :3] = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta),  torch.cos(theta), 0],
                [0,                0,                 1]
            ], device=self.device)

            next_pose = g_partial @ trans @ rot_z
        else:
            # Normal forward twist motion
            next_pose = current_pose @ torch.linalg.matrix_exp(xi_hat)
            # Check if the step is backwards
            step_vec = next_pose[:3, 3] - current_pose[:3, 3]
            forward_dir = current_pose[:3, 2]
            # If step is backwards (negative dot product), zero it
            if torch.dot(step_vec, forward_dir) < 0:
                next_pose = current_pose.clone()

        return next_pose


    def set_tumor_positions(self):
        for env_id in self.env_ids:
            env_id = int(env_id)
            offset = torch.tensor(self.offsets[env_id], dtype=torch.float32, device=self.device)
            tumor_data = self.tumor_pickle[env_id]
            tumor_pos = torch.tensor(tumor_data["tumor_position"], dtype=torch.float32, device=self.device)
            tumor_quat = torch.tensor(tumor_data["tumor_quat"], dtype=torch.float32, device=self.device)
            tumor_centroid = torch.tensor(tumor_data["tumor_centroid"], dtype=torch.float32, device=self.device)
            self.shuffled_tumor_centroids[env_id] = tumor_centroid  #+ offset
            self.tumor.set_local_poses(tumor_pos.unsqueeze(0), tumor_quat.unsqueeze(0), [env_id])
            # print(f"Set tumor position for env {env_id}: {tumor_pos}")

    def save_point_cloud_ply(self, filename, points: np.ndarray):
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def draw_cylinder(self, center, axis, radius, height, num_points=100):
        """
        Draw a cylinder centered at `center`, aligned along `axis` with specified radius and height.
        The cylinder is visualized using its top and bottom circular cross-sections.
        """
        # Normalize the axis (ensure it's a unit vector)
        axis = axis / np.linalg.norm(axis)

        # Create points on the top and bottom circular cross-sections of the cylinder
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_points = radius * np.array([np.cos(theta), np.sin(theta)]).T
        
        # Extend circle_points to 3D by adding a column of zeros for the z-coordinate
        circle_points_3d = np.hstack([circle_points, np.zeros((num_points, 1))])

        # Compute the top and bottom circle points (offset along the axis)
        top_circle = circle_points_3d + center + height / 2 * axis  # Top circle offset along axis
        bottom_circle = circle_points_3d + center - height / 2 * axis  # Bottom circle offset along axis

        # Draw the points on the top and bottom circles
        self.draw_points(top_circle, color=(0.0, 1.0, 0.0, 1.0), size=4.0)  # Green color for the top circle
        self.draw_points(bottom_circle, color=(1.0, 0.0, 0.0, 1.0), size=4.0)  # Red color for the bottom circle

        # Draw lines connecting corresponding points from the top and bottom circles
        for top, bottom in zip(top_circle, bottom_circle):
            self.draw_lines(top, bottom, color="yellow")  # Yellow lines for the sides of the cylinder

        # # Optionally, connect the points on the top and bottom circles to visualize the circumference
        # for i in range(num_points):
        #     self.draw_lines(top_circle[i], top_circle[(i + 1) % num_points], color="yellow")
        #     self.draw_lines(bottom_circle[i], bottom_circle[(i + 1) % num_points], color="yellow")

    def filter_hits_in_cylinder(self, hits_np, center, axis, radius=0.01, height=0.05):
        """
        Filters 3D points inside a cylinder centered at `center`, aligned along `axis`.
        """
        vecs = hits_np - center  # vectors from center to hit points
        proj_lengths = np.dot(vecs, axis)  # projection on axis (height direction)
        radial_vecs = vecs - np.outer(proj_lengths, axis)
        radial_dists = np.linalg.norm(radial_vecs, axis=1)
        # print(f"Radial distances: {radial_dists, }, Projection lengths: {proj_lengths}")
        # Condition: within radius and within height range
        mask = (proj_lengths >= -height / 2) & (proj_lengths <= height / 2) & (radial_dists <= radius)
        return hits_np[mask]
    
    def boundary_check_tumor(self):
        hits = self.raycast_tumor.data.ray_hits_w
        for env_id in range(self.num_envs):
            hits_env = hits[env_id]  # shape (R, 3)
            valid_mask = torch.isfinite(hits_env).all(dim=-1)  # shape (R,)
            valid_hits = hits_env[valid_mask]  # shape (V, 3)
            tool_tip = torch.tensor([[0.0, 0.0, 0.0]])  # Fill with dummy
            needle_center = tool_tip[env_id].cpu().numpy()
            # Assuming the tool's orientation provides the correct insertion axis
            insertion_axis = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Fill with dummy
            insertion_axis_np = insertion_axis[env_id].cpu().numpy()
            axis = insertion_axis_np[:3]  # (x, y, z)
            axis = axis / np.linalg.norm(axis)
            valid_hits_np = valid_hits.cpu().numpy()
            filtered_hits = self.filter_hits_in_cylinder(
                hits_np=valid_hits_np,
                center=needle_center,
                axis=axis,
                radius=0.05,
                height=0.25
            )

            print(f"[env {env_id}] Hits inside cylinder Tumor: {filtered_hits.shape[0]}/{valid_hits_np.shape[0]}")
            #self.draw_points(filtered_hits, color=(1.0, 0.0, 1.0, 1.0), size=4.0) 
            if valid_hits_np.shape[0] != 0 and filtered_hits.shape[0] > 0:
                print("Valid hits shape:", valid_hits_np.shape, filtered_hits.shape)
                return filtered_hits  # Return filtered hits for further processing or visualization
            
    def boundary_check_vessel(self):
        # Get all ray hits (num_envs, num_rays, 3)
        hits = self.raycast_vessel.data.ray_hits_w  
        valid_mask = torch.isfinite(hits).all(dim=-1)  
        
        # Get all needle positions and orientations
        positions = self._needle.data.root_link_pos_w  
        quats = self._needle.data.root_link_quat_w     

        # Convert quaternions to rotation matrices → z-axes (insertion directions)
        quats_np = quats.cpu().numpy()  
        Rs = R.from_quat(quats_np).as_matrix()  
        z_axes = Rs[:, :, 2]  
        sparse_points_all = []

        for env_id in range(self.num_envs):
            valid_hits_env = hits[env_id][valid_mask[env_id]]  
            if valid_hits_env.numel() == 0:
                omni.log.warn(f"[env {env_id}] No valid hits for vessel raycast.")
                sparse_points_all.append(None)
                continue

            center = positions[env_id].cpu().numpy()
            axis = z_axes[env_id]
            axis = axis / np.linalg.norm(axis)
            hits_np = valid_hits_env.cpu().numpy()

            # Filter hits inside the cylinder
            filtered_hits = self.filter_hits_in_cylinder(
                hits_np=hits_np,
                center=center,
                axis=axis,
                radius=0.05,
                height=0.25,
            )

            if not self.cfg.viewer.headless and filtered_hits is not None:
                self.draw_points(filtered_hits, color=(0.0, 1.0, 1.0, 1.0), size=4.0)

            if filtered_hits.shape[0] == 0:
                omni.log.warn(f"[env {env_id}] No valid hits inside cylinder.")
                sparse_points_all.append(None)
                continue

            # Prepare point cloud
            self.pcd.points = o3d.utility.Vector3dVector(filtered_hits)

            if len(self.pcd.points) < 64:
                points_np = np.asarray(self.pcd.points)
                num_missing = 64 - len(points_np)
                idxs = np.random.choice(len(points_np), num_missing, replace=True)
                noise = np.random.normal(loc=0.0, scale=1e-4, size=(num_missing, 3))
                padded = np.concatenate([points_np, points_np[idxs] + noise], axis=0)
                sparse = torch.tensor(padded, dtype=torch.float32, device=self.device)
            else:
                sparse_pcd = self.pcd.farthest_point_down_sample(64)
                sparse = torch.tensor(np.asarray(sparse_pcd.points), dtype=torch.float32, device=self.device)

            sparse_points_all.append(sparse)

        return sparse_points_all 

    def distance_to_vessel(self):
        """
        Compute the mean raycast distance from the tooltip to the vessel for each environment.
        Returns:
            torch.Tensor: shape (num_envs, 1), distance in meters.
        """
        distances = self.raycast_cam_vessel.data.output.get("distance_to_camera", None)

        if distances is None or distances.shape[0] == 0:
            omni.log.warn("[distance_to_vessel] Raycast distances not yet populated.")
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        max_dist = self.raycast_cam_vessel_max_distance  # meters
        B, H, W, _ = distances.shape
        mean_dists = torch.zeros((B, 1), dtype=torch.float32, device=self.device)

        for env_id in range(B):
            dists = distances[env_id, :, :, 0]  # (H, W)
            valid = (~torch.isinf(dists)) & (dists <= max_dist)
            num_valid = valid.sum().item()

            if num_valid > 0:
                if self.cfg.viewer.save:
                    # Extract camera data
                    camera_index = 0
                    # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                    single_cam_data = convert_dict_to_backend(
                        {k: v[camera_index] for k, v in self.raycast_cam_vessel.data.output.items()}, backend="numpy"
                    )
                    # Extract the other information
                    single_cam_info = self.raycast_cam_vessel.data.info[camera_index]

                    # Pack data back into replicator format to save them using its writer
                    rep_output = {"annotators": {}}
                    for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                        if info is not None:
                            rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                        else:
                            rep_output["annotators"][key] = {"render_product": {"data": data}}
                    # Save images
                    rep_output["trigger_outputs"] = {"on_time": self.raycast_cam_vessel.frame[camera_index]}
                    self.rep_writer.write(rep_output)
                    
                print(f"[env {env_id}] Valid rays: {num_valid}")
                raw = dists[valid]
                print(f"[env {env_id}] Raw distances: {raw.max()}, {raw.min()}, {raw.mean()}")
                mean = dists[valid].mean()
                mean_dists[env_id, 0] = mean
                omni.log.info(f"[env {env_id}] Vessel distance: mean={mean.item():.6f}, valid rays={num_valid}")
            else:
                omni.log.warn(f"[env {env_id}] No valid vessel rays within {max_dist * 1000:.1f} mm")
                mean_dists[env_id, 0] = 100.0

        return mean_dists  # shape: (B, 1)

    def distance_to_tumor(self):
        """
        Compute the mean raycast distance from the tooltip to the tumor for each environment.
        Returns:
            torch.Tensor: shape (num_envs, 1), distance in meters.
        """
        distances = self.raycast_cam_tumor.data.output.get("distance_to_camera", None)

        if distances is None or distances.shape[0] == 0:
            omni.log.warn("[distance_to_tumor] Raycast distances not yet populated.")
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        max_dist = 0.0005  # meters
        B, H, W, _ = distances.shape
        mean_dists = torch.zeros((B, 1), dtype=torch.float32, device=self.device)

        for env_id in range(B):
            dists = distances[env_id, :, :, 0]  # (H, W)
            valid = (~torch.isinf(dists)) & (dists <= max_dist)
            num_valid = valid.sum().item()

            if num_valid > 0:
                raw = dists[valid]
                mean = dists[valid].mean()
                mean_dists[env_id, 0] = mean
                omni.log.info(f"Raw: {raw}, Mean: {mean}, Valid rays: {num_valid}")
                # omni.log.info(f"[env {env_id}] Tumor distance: raw: {raw.item()}, mean={mean.item():.5f}, valid rays={num_valid}")
            else:
                omni.log.warn(f"[env {env_id}] No valid tumor rays within {max_dist * 1000:.1f} mm")
                mean_dists[env_id, 0] = 100.0

        return mean_dists  # shape: (B, 1)

    def brain_shift(self, points_attr, original_np, env_id):
        new_np = self.brain_shift_data[env_id][0]
        print(f"[INFO] [Env {env_id}] Original points shape: {original_np.shape}, New points shape: {new_np.shape}")
        assert new_np.shape == original_np.shape, f"Shape mismatch at env {env_id} ({new_np.shape} vs {original_np.shape})"
        # usd_pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in new_np]
        # points_attr.Set(usd_pts)
        # prim_path = points_attr.GetPrim().GetPath().pathString
        # print(f"[INFO] [Env {env_id}] Updated mesh points at {prim_path}")
        # #self.raycast_vessel.update_dynamic_mesh_for_env(prim_path, new_np, env_id=0)
        # stage_utils.update_stage()
        # displacement = np.linalg.norm(new_np - original_np, axis=1)
        # changed_mask = displacement > 1e-5
        # changed_percent = 100.0 * np.sum(changed_mask) / displacement.shape[0]
        # print(f"[INFO] [Env {env_id}] Vertices changed: {changed_percent:.2f}%")
        # print(f"[INFO] [Env {env_id}] Mean displacement: {np.mean(displacement):.6f}")
        # print(f"[INFO] [Env {env_id}] Max displacement:  {np.max(displacement):.6f}")

    def get_vessel_points(self):
        try:
            
            # for prim in stage.Traverse():
            #     if prim.IsA(UsdGeom.Mesh):
            #         print(f"[DEBUG] Found UsdGeom.Mesh at {prim.GetPath()}")
            #         print(f"[DEBUG] Found prim: {prim.GetPath()} of type {prim.GetTypeName()}")
            self.stage = stage_utils.get_current_stage()
            for i in range(self.num_envs):
                prim_path = f"/World/envs/env_{i}/Vessel/Vessels/Vessels"
                raw_prim = self.stage.GetPrimAtPath(prim_path)
                prim = UsdGeom.Mesh(raw_prim)
                points_attr = prim.GetPointsAttr()
                if not points_attr.IsDefined():
                    print(f"[ERROR] points attribute not defined at {prim_path}")
                    continue

                original_np = np.asarray(points_attr.Get())
                if original_np.size == 0:
                    print(f"[ERROR] No points found in mesh at {prim_path}")
                    continue
                elif original_np.size:
                    # print(f"[DEBUG] Original points shape: {original_np.shape} at {prim_path}")
                    self.brain_shift(points_attr, original_np, i)

        except Exception as e:
            print(f"[ERROR] Failed to access mesh points: {e}")
        
    def get_real_time_collision_scores(self, entry_point, tumor_center, vessel_points, num_samples=1000):
        vessels_kd = cKDTree(vessel_points)

        # Sample points along straight line path
        line = np.linspace(0, 1, num_samples).reshape(-1, 1) * (tumor_center - entry_point) + entry_point

        # Compute distances to nearest vessel point
        dists, _ = vessels_kd.query(line)
        collisions = dists < self.DIST_THRESHOLD

        collision_score = np.sum(collisions)
        collision_ratio = collision_score / num_samples

        return collision_score, collision_ratio
    
    def make_transform(self, position, quaternion):
        """Convert pos + quat to a 4x4 transform matrix."""
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        if isinstance(quaternion, torch.Tensor):
            quaternion = quaternion.cpu().numpy()

        rot = R.from_quat(quaternion[[1, 2, 3, 0]])  # Convert (w,x,y,z) → (x,y,z,w)
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = torch.tensor(rot.as_matrix())
        T[:3, 3] = torch.tensor(position)
        return T

    def extract_pose_from_transform(self,T):
        """Convert 4x4 matrix to (position, quaternion)"""
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3].numpy())
        quat = torch.tensor(rot.as_quat())  # (x, y, z, w)
        quat = quat[[3, 0, 1, 2]]  # → (w, x, y, z)
        return pos, quat
    
    def draw_points(self, points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):    
        #self.draw.clear_points()
        # Convert to numpy if torch
        if isinstance(points_np, torch.Tensor):
            points_np = points_np.detach().cpu().numpy()
        # Handle empty input
        if points_np is None or points_np.size == 0:
            return
        # Handle different shapes
        if points_np.ndim == 3 and points_np.shape[1:] == (4, 4):
            # Batch of SE(3) poses → extract positions from transformation matrices
            points_np = points_np[:, :3, 3]
        elif points_np.ndim == 2 and points_np.shape[1] == 4:
            # Possibly incorrectly passed SE(3) matrix as flattened rows
            points_np = points_np[:, :3]
        elif points_np.ndim == 2 and points_np.shape[1] == 3:
            # Already list of 3D points
            pass
        elif points_np.ndim == 1 and points_np.shape[0] == 4:
            # Single 4D vector → assume it's a homogeneous point, strip last
            points_np = points_np[:3].reshape(1, 3)
        elif points_np.ndim == 1 and points_np.shape[0] == 3:
            # Single 3D point
            points_np = points_np.reshape(1, 3)
        else:
            raise ValueError(f"Unsupported shape for draw_points: {points_np.shape}")

        # Final drawing
        point_list = [tuple(p) for p in points_np]
        colors = [color] * len(point_list)
        sizes = [size] * len(point_list)

        self.draw.draw_points(point_list, colors, sizes)


    def draw_lines(self, start, end, color):
        if isinstance(start, torch.Tensor):
            start_pose = start.cpu().numpy()
        else:
            start_pose = np.asarray(start)

        if isinstance(end, torch.Tensor):
            end_pose = end.cpu().numpy()
        else:
            end_pose = np.asarray(end)

        # Ensure shape is (B, 3)
        if start_pose.ndim == 1:
            start_pose = start_pose[None, :]
        if end_pose.ndim == 1:
            end_pose = end_pose[None, :]

        b = start_pose.shape[0]

        color_green = (0.0, 1.0, 0.0, 1.0)  # green line
        color_yellow = (1.0, 1.0, 0.0, 1.0)  # yellow line
        color_red = (1.0, 0.0, 0.0, 1.0)  # red line
        if color == "yellow":
            colors = [color_yellow for _ in range(b)]
        elif color == "red":
            colors = [color_red for _ in range(b)]
        else:
            colors = [color_green for _ in range(b)]
        sizes = [1.0 for _ in range(b)]
        # print("Drawing line from", start_pose, "to", end_pose)
        self.draw.draw_lines(
            start_pose.tolist(),
            end_pose.tolist(),
            colors,
            sizes
        )

    def draw_entry_points(self):
        for env_id, points in enumerate(self.tumor_entry_points):
            omni.log.info(f"Drawing entry points for env: {env_id}")
            try:
                offset = self.offsets[env_id]
                offset_points = points + offset
                for point in offset_points:
                    self.draw_points([point], color=(0.2, 0.8, 0.2, 1.0), size=4.0)
            except Exception as e:
                omni.log.error(f"Exception: {e} for env {env_id}. Tumor data may be incomplete.")

    def draw_tumor_centroids(self):
        try:
            for env_id, points in enumerate(self.tumor_centroids):
                offset = self.offsets[env_id]  # shape (3,)
                offset_points = points + offset  # (N, 3) + (3,) → (N, 3)
                self.draw_points([offset_points], color=(0.0, 0.0, 1.0, 1.0), size=4.0)        
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw tumor centroids: {e}")

    def draw_top_points(self):   
        try:
            for env_id, top_points in enumerate(self.tumor_top_entry_points):
                for i in range(len(top_points)):
                    offset = self.offsets[env_id]  # shape (3,)
                    offset_points = top_points[i]["entry_point"] + offset  # (N, 3) + (3,) → (N, 3)
                    self.draw_points([offset_points], color=(0.0, 1.0, 0.0, 1.0), size=4.0)
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw top entry points because: {e}")
        
    def draw_path(self):
        try:
            for env_id, top_points in enumerate(self.tumor_top_entry_points):
                tumor_center = self.tumor_centroids[env_id]
                offset = self.offsets[env_id]  # shape (3,)

                for i in range(len(top_points)):
                    entry_point = top_points[i]["entry_point"]
                    p1 = entry_point + offset
                    p2 = tumor_center + offset

                    self.draw_lines(p1, p2, color="green")
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw lines: {e}")