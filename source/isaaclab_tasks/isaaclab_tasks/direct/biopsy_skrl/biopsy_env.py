# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Sequence
import torch
torch.set_printoptions(profile="full")
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from pxr import UsdGeom, UsdPhysics, Gf
import open3d as o3d
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import math
import random
import omni.log
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.cloner import GridCloner
import isaacsim.core.utils.stage as stage_utils
# try:
#     from isaacsim.util.debug_draw import _debug_draw
# except ImportError:
#     from omni.isaac.debug_draw import _debug_draw
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, normalize, quat_mul, quat_mul,quat_inv,quat_apply,axis_angle_from_quat,quat_from_angle_axis

from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns, RayCaster
from isaaclab_assets import UR5_CFG
from isaaclab.utils.io import dump_pickle, load_pickle
from isaaclab.utils.warp import convert_to_warp_mesh, multi_raycast_mesh
import warp as wp
#from . import biopsy_preop
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

    # vessel_native = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/OVessel",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/czlocal/sanjay_isaac/forked/Vessels.usd"
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    # )

    tumor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tumor",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/tumor.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    raycast_camera_vessel = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0) ,convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.001,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_camera_tumor = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.001,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_vessel = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        attach_yaw_only=True,
        max_distance=0.01,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-180, 180], horizontal_fov_range=[-180, 180], horizontal_res=1.0
        )
    )

    raycast_tumor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        attach_yaw_only=True,
        max_distance=0.01,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-60, 60], horizontal_fov_range=[-20, 20], horizontal_res=1.0
        )
    )


@configclass
class BiopsyDirectEnvCfg(DirectRLEnvCfg):
    #env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
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
    w_tumor_dist = 5.0
    w_vessel_penalty = 3.0
    w_collision_score = 1.5
    bonus_inside_tumor = 10.0
    pass


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
    #preop : biopsy_preop.BiopsyPreop

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

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.cloner = GridCloner(spacing=self.cfg.scene.env_spacing)
        self.offsets, _ = self.cloner.get_clone_transforms(self.num_envs)
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        omni.log.info(f"Robot DOF limits: {self.robot_dof_lower_limits}, {self.robot_dof_upper_limits}")
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("holder_needle_slider")[0]] = 0.1
        #self.preop = biopsy_preop.BiopsyPreop()
        #self.preop.print_test()
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/needle_tool/holder_link")
        omni.log.info(f"Prim: {prim}")
        omni.log.info(f"Is valid: {prim.IsValid()}")

        holder_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/needle_tool/holder_link")),
            self.device,
        )
        tooltip_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/needle_tool/tooltip")),
            self.device,
        )
        needle_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/needle_tool/needle_link")),
            self.device,
        )
        omni.log.info(f"POSEs: {holder_pose, tooltip_pose, needle_pose}")
        self.tooltip_index = self._robot.find_bodies("tooltip")[0][0]
        self.holder_index = self._robot.find_bodies("holder_link")[0][0]
        self.needle_index = self._robot.find_bodies("needle_link")[0][0]
        omni.log.info(f"Link Indices: {self.tooltip_index, self.holder_index, self.needle_index}")
        self.active_path_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        omni.log.info(f"Robot Tool Tip Position before extension: {self._robot.data.body_pos_w[:, self.tooltip_index]}")
        # Extend the needle for checking if raycast camera hits the tumor
        joint_pos, joint_vel = self._robot.data.default_joint_pos.clone(), self._robot.data.default_joint_vel.clone()
        joint_pos[:, 0] = -0.17
        joint_vel[:] = 0.0
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.set_joint_position_target(joint_pos)
        self._robot.write_data_to_sim()
        print(f"Robot Tool Tip Position: {self._robot.data.body_pos_w[:, self.tooltip_index]}")
        #self.draw = _debug_draw.acquire_debug_draw_interface()

        self.DIST_THRESHOLD = 0.005
        # read pickled data
        omni.log.info("Loading tumor data...")
        self.tumor_positions = []
        self.tumor_quaternions = []
        self.tumor_centroids = []
        self.scored_paths = []
        self.tumor_top_entry_points = []
        self.start_positions = []
        self.start_quaternions = []
        self.tumor_pickle = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/pickle_finale/rl_dataset_10envs.pkl")  #/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_2205_cleaned.pkl
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
        #self.draw_entry_points()
        #self.draw_path()
        self.draw_attempt = 0
        assert len(self.tumor_positions) == self.num_envs, f"Number of tumor positions {len(self.tumor_positions)} does not match number of envs {self.num_envs}"
        self.shuffled_tumor_centroids = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.tooltip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tooltip_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.tumor = self.scene["tumor"]

        # Sensors
        self.raycast_cam_tumor = self.scene["raycast_camera_tumor"]
        self.raycast_cam_vessel = self.scene["raycast_camera_vessel"]
        self.raycast_vessel = self.scene["raycast_vessel"]
        self.raycast_tumor = self.scene["raycast_tumor"]

        # Brain Shift 
        self.brain_shift_data = []
        shift_data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/custom/path_comparison/path_comparison/pickle_finale/precomputed_brain_deformations10_10envs.pkl")
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

        self.stage = stage_utils.get_current_stage()
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.set_tumor_positions()
        #self.get_vessel_points()
        # Observation space terms:
        self.robot_root_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.robot_root_vel = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)  # (vx, vy, vz, wx, wy, wz)
        self.robot_tooltip_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.collision_scores = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)  # Placeholder for collision scores
        self.entry_pose_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)  # Index of the entry pose in the start poses
        self.success_flag = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)  # Success flag for each environment
        self.entry_confidence = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)  # Confidence for the entry pose
        self.pcd = o3d.geometry.PointCloud()  # Placeholder for point cloud data
        self.retracting = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.trial_counts = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

        self.current_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.orientation_offset = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_envs, dtype=torch.float32, device=self.device)

        self.pose_applied = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.trial_phase = ["preop"] * self.num_envs
        self.trial_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        self.debug_mode = True
        self.debug_force_fail = False
        self.debug_force_success = True


    def _setup_scene(self):
        """
        Setup the scene for the environment. But since InteractiveSceneCfg is used, the scene is already setup in the
        constructor of the parent class.

        To add new assets to the scene, use the `scene` attribute of the environment.
        refer https://forums.developer.nvidia.com/t/importing-scene-created-in-isaac-sim-to-isaac-lab/315590
        """
        self._robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self._robot

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process agent actions:
        - actions[:, :3]: delta position (x, y, z)
        - orientation remains fixed to preop base orientation
        """
        self.actions = actions.clone()  # Store the actions for later use
        # actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # max_translation = self.cfg.action_scale
        # delta_pos = torch.clamp(actions[:, :3], -max_translation, max_translation)

        # self.current_pos = delta_pos  # (B, 3)

    def _apply_action(self):
        if isinstance(self.single_action_space, Box):
            self.current_pos = self.cfg.action_scale * self.actions[:, :3]
        elif isinstance(self.single_action_space, Discrete):
            print("Trying to apply actions but currenty nothing is defined")
            pass
        elif isinstance(self.single_action_space, Dict):
            pass

    def apply_action(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        pose_ids = self.active_path_idx[env_ids]  # [B]

        # Get environment offsets
        offsets = self.scene.env_origins[env_ids]  # [B, 3]

        # Gather base poses
        base_pos = torch.stack([
            self.start_positions[env_id][pose_ids[env_id]].to(self.device)
            for env_id in env_ids
        ])  # [B, 3]
        base_quat = torch.stack([
            self.start_quaternions[env_id][pose_ids[env_id]].to(self.device)
            for env_id in env_ids
        ])  # [B, 4]

        # Apply delta from pre-physics
        delta_pos = self.current_pos[env_ids]  # [B, 3]
        tooltip_pos = base_pos + delta_pos + offsets  # final position [B, 3]
        tooltip_quat = base_quat  # [B, 4]

        # Debugging: Log the tooltip position and rotation if in debug mode
        if self.debug_mode:
            print(f"DEBUG: Applying action for {self.num_envs} environments.")
            print(f"DEBUG: Tooltip position: {tooltip_pos}")
            print(f"DEBUG: Tooltip quaternion: {tooltip_quat}")

        # Compute holder poses
        holder_quats, holder_positions = self.tooltip_to_holder(tooltip_quat, tooltip_pos)  # [B, 4], [B, 3]

        # Apply only for those envs that haven't already had pose applied
        mask = ~self.pose_applied[env_ids]  # [B]
        selected_envs = env_ids[mask]

        if selected_envs.numel() > 0:
            root_state = torch.zeros((selected_envs.shape[0], 13), device=self.device)
            root_state[:, :3] = holder_positions[mask]
            root_state[:, 3:7] = holder_quats[mask]
            root_state[:, 7:] = 0.0

            # Debugging: Log applied pose if debug mode is enabled
            if self.debug_mode:
                print(f"DEBUG: Root state before sending to simulation: {root_state}")

            self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids=selected_envs)
            self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=selected_envs)
            self.pose_applied[selected_envs] = True

        # === Insertion logic ===
        slider_idx = self._robot.find_joints("holder_needle_slider")[0]
        needle_step = 0.001

        retracting = self.retracting[env_ids]
        self.robot_dof_targets[env_ids[~retracting], slider_idx] -= needle_step
        self.robot_dof_targets[env_ids[retracting], slider_idx] += 2 * needle_step

        # Debugging: Log the robot's joint target positions if in debug mode
        if self.debug_mode:
            print(f"DEBUG: Robot joint targets: {self.robot_dof_targets}")

        # === Phase switch ===
        depths = self.robot_dof_targets[env_ids, slider_idx]
        for i, env_id in enumerate(env_ids):
            if depths[i] <= -0.09:
                if self.trial_phase[env_id] == "preop":
                    # Debugging: Log trial phase transition
                    if self.debug_mode:
                        print(f"DEBUG: Transitioning trial phase for env {env_id} to RL.")
                    self.trial_phase[env_id] = "rl"
                    self.pose_applied[env_id] = False
                    self.robot_dof_targets[env_id, slider_idx] = 0.0
                    self.retracting[env_id] = True
                else:
                    self.trial_done[env_id] = True
                    # Debugging: Log trial completion for env
                    if self.debug_mode:
                        print(f"DEBUG: Trial done for env {env_id}.")
            if self.retracting[env_id] and depths[i] >= 0.0:
                self.retracting[env_id] = False

        # Handle debug failure condition
        if self.debug_force_fail:
            if self.debug_mode:
                print("DEBUG: Forcing failure in the current insertion attempt.")
            # Force a failure behavior, e.g., reset or stop further actions.
            self.trial_done[env_ids] = True

        # Handle debug success condition
        if self.debug_force_success:
            if self.debug_mode:
                print("DEBUG: Forcing success in the current insertion attempt.")
            self.trial_done[env_ids] = True

        self._robot.set_joint_position_target(self.robot_dof_targets)



    def _compute_intermediate_values(self, env_ids):
        """
        Compute intermediate values for the environment. This includes computing the action to be applied to the robot
        and the observations to be returned to the agent.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        
        self.tool_tip_pos = self._robot.data.body_pos_w[env_ids, self.tooltip_index]
        self.tool_tip_quat = self._robot.data.body_quat_w[env_ids, self.tooltip_index]

    
    def tooltip_to_holder(self, start_quat, start_pos):
        is_batched = len(start_quat.shape) == 2

        if not is_batched:
            start_quat = start_quat.unsqueeze(0)
            start_pos = start_pos.unsqueeze(0)

        start_quat = start_quat.to(dtype=torch.float32)
        start_pos = start_pos.to(dtype=torch.float32)

        batch_size = start_quat.shape[0]

        tooltip_to_holder_pos = torch.tensor(
            [0.02464, -0.00005, -0.0265], dtype=torch.float32, device=self.device
        ).expand(batch_size, -1)

        tooltip_to_holder_quat = torch.tensor(
            R.from_euler("xyz", [0, 0, 1.5707]).as_quat(canonical=False),
            dtype=torch.float32, device=self.device
        ).expand(batch_size, -1)

        holder_to_tooltip_quat, holder_to_tooltip_pos = tf_inverse(
            tooltip_to_holder_quat, tooltip_to_holder_pos
        )

        holder_quat, holder_pos = tf_combine(
            start_quat, start_pos, holder_to_tooltip_quat, holder_to_tooltip_pos
        )

        if not is_batched:
            return holder_quat.squeeze(0), holder_pos.squeeze(0)
        return holder_quat, holder_pos

    def reset_idx(self, env_ids):
        """
        Reset the environment and set fixed tumor positions.
        This method ensures that the tumor position remains fixed and is set only once per environment.
        """
        super()._reset_idx(env_ids)
        
        root_state = self._robot.data.default_root_state.clone()
        if not hasattr(self, "tumor_position_set"):
            self.tumor_position_set = True
            omni.log.info("Initializing tumor positions...")
            for env_id in env_ids:
                tumor_data = self.tumor_pickle[env_id]
                tumor_pos = torch.tensor(tumor_data["tumor_position"], dtype=torch.float32, device=self.device)
                tumor_quat = torch.tensor(tumor_data["tumor_quat"], dtype=torch.float32, device=self.device)

                self.tumor.set_local_poses(tumor_pos.unsqueeze(0), tumor_quat.unsqueeze(0), [env_id])
                print(f"Fixed tumor position for env {env_id}: {tumor_pos}")
        else:
            print(f"Tumor position for environments {env_ids} remains fixed.")

        for env_id in env_ids:
            tumor_data = self.tumor_pickle[env_id]
            path_idx = random.randint(0, len(tumor_data["start_pose"]) - 1)
            start_pose = tumor_data["start_pose"][path_idx]
            start_pos = start_pose["position"].to(self.device)
            start_quat = start_pose["quaternion"].to(self.device)
            ttip_quat, ttip_pos = self.tooltip_to_holder(start_quat, start_pos)
            root_state[env_id, :3] = ttip_pos
            root_state[env_id, 3:7] = ttip_quat
            root_state[env_id, 7:] = 0.0
            self.active_path_idx[env_id] = path_idx
            # Set flags and reset the environment status
            self.trial_phase[env_id] = "preop"  # Start with preop phase
            self.trial_done[env_id] = False  # Not done yet
            self.pose_applied[env_id] = False  # Ensure pose isn't applied initially
            self.retracting[env_id] = False  # Ensure retracting flag is reset
            self.trial_counts[env_id] = 0  # Reset trial count
        self._robot.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
        self._compute_intermediate_values(env_ids)
        omni.log.info(f"Reset complete for envs: {env_ids}")


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
            print(f"Set tumor position for env {env_id}: {tumor_pos}")

    def _reset_idx(self, env_ids):
        """
            A) Randomly select a tumor centroid from the list of centroids
            B) Randomly select a start pose from the list of start poses based on the selected centroid
            C) Set tooltip position and rotation to the start pose
            D) Set the robot joint positions to the start pose
            E) Set the robot joint velocities to zero
            F) Save the active path index per environment
        """
        super()._reset_idx(env_ids)
        omni.log.info(f"Env ID: {env_ids}, {type(env_ids)}")
        root_state = self._robot.data.default_root_state.clone()
        for env_id in env_ids:
            print("Resetting environments with IDs:", env_id)
            env_id = int(env_id)
            offset = torch.tensor(self.offsets[env_id], dtype=torch.float32, device=self.device)
            self.trial_phase[env_id] = "preop"
            self.trial_done[env_id] = False
            self.pose_applied[env_id] = False
            tumor_data = self.tumor_pickle[env_id]
            path_idx = random.randint(0, len(tumor_data["start_pose"]) - 1)
            start_pose = tumor_data["start_pose"][path_idx]
            start_pos = start_pose["position"].to(self.device) + offset
            start_quat = start_pose["quaternion"].to(self.device)
            ttip_quat, ttip_pos = self.tooltip_to_holder(start_quat, start_pos)
            root_state[env_id, :3] = ttip_pos
            root_state[env_id, 3:7] = ttip_quat
            root_state[env_id, 7:] = 0.0 
            self.active_path_idx[env_id] = path_idx
            print(f"[env {env_id}] Using path index {path_idx} for tumor at {tumor_data['tumor_position']}")
        self._robot.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
        
        # Recompute any intermediate buffers (like tooltip pos, etc.)
        self._compute_intermediate_values(env_ids)

    def _get_dones(self):
        dummy_dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dummy_dones, time_out

    def _get_observations(self):
        """
        Get the observations for the environment. This includes:
        a) Tooltip Pose
        b) Joint Velocities
        c) Depth to Tumor
        d) Direction to Tumor Centroid
        e) Downsampled PCD from RayCast Sensor
        f) Preop Pose (start pose)
        g) Try history and results
        h) One-hot encoding of active path index
        i) Estimated confidence (depth score, vessel hits, past success)
        """
        # --- Core kinematics ---
        self.tool_tip_pos = self._robot.data.body_pos_w[:, self.tooltip_index]  # [B, 3]
        self.tooltip_rot = self._robot.data.body_quat_w[:, self.tooltip_index]  # [B, 4]

        # --- Tumor geometry ---
        # to_tumor_centroid = self.shuffled_tumor_centroids - self.tool_tip_pos ----> This can be used in reward calculation
        depth_to_tumor = self.distance_to_tumor()  # [B]
        if depth_to_tumor.ndim == 1:
            depth_to_tumor = depth_to_tumor.unsqueeze(-1) # Ensure it's [B, 1] for consistency
        tip_to_vessel = self.distance_to_vessel()  # [B]
        print(f"Distance to tumor: {depth_to_tumor}, Distance to vessel: {tip_to_vessel}")

        pcd_vessels = self.boundary_check_vessel()
        omni.log.info(f"PCD Vessel Shape: {pcd_vessels.shape if pcd_vessels is not None else 'None'}")
        omni.log.info(f"PCD Vessel: {pcd_vessels.ndim}")
        if pcd_vessels is None:
            pcd_vessels = torch.zeros((self.num_envs, 64, 3), device=self.device)
        elif pcd_vessels.ndim == 2:
            # duplicate same PCD for all envs
            pcd_vessels = pcd_vessels.unsqueeze(0).repeat(self.num_envs, 1, 1)
        elif pcd_vessels.shape[0] != self.num_envs:
            raise ValueError(f"Expected pcd_vessels to have {self.num_envs} samples, got {pcd_vessels.shape}")


        # --- Confidence score (simple heuristic) ---
        vessel_penalty = (tip_to_vessel <= 0.005).float()  # e.g., if close to vessel
        confidence = torch.where(depth_to_tumor > 0, 1.0 / (depth_to_tumor + 1e-5), torch.tensor(0.0, device=self.device))  # [B]
        confidence = confidence * (1.0 - vessel_penalty) 
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = torch.clamp(confidence, min=1e-3, max=10.0)  
        print(f"Confidence scores: {confidence}")

        # Preop pose (start pose for current path) ---
        # preop_pos = torch.stack([
        #     torch.tensor(self.start_positions[env_id][self.active_path_idx[env_id]], device=self.device)
        #     for env_id in range(self.num_envs)
        # ])  # [B, 3]
        # preop_quat = torch.stack([
        #     torch.tensor(self.start_quaternions[env_id][self.active_path_idx[env_id]], device=self.device)
        #     for env_id in range(self.num_envs)
        # ])  # [B, 4]

        # # --- Try history (binary success/failure for 3 attempts) ---
        # try_history = torch.zeros((self.num_envs, 3), device=self.device)  # [B, 3]
        
        # --- One-hot encoding for active path index ---
        #path_one_hot = torch.nn.functional.one_hot(self.active_path_idx, num_classes=3).float()  # [B, 3]

        obs = {"tooltip_position": self.tool_tip_pos, "tooltip_quaternion": self.tool_tip_quat, "raycaster": pcd_vessels, "depth_tumor": depth_to_tumor}  # "trial": try_history, 
        for k, v in obs.items():
            print(f"{k}: {v.shape}")

        return {"policy": obs}


    def _get_rewards(self):  # TODO: to get calculated Rewards
        """
        Get the rewards for the environment. This includes:
        a) RayCaster Camera reward based on distance to image plane and distance to camera

        """
        total_reward = self.__compute_reward()
        return total_reward

    def _get_states(self):  # TODO: States
        pass

    def _get_actions(self):  # TODO: Actions
        pass

    def __compute_reward(self):  # TODO: actual Rewards
        """
        Reward structure:
        + reward for being closer to tumor
        - penalty for being close to vessels
        - penalty for high real-time collision score
        + bonus for being inside tumor
        """

        # === 1. Compute raycast-based distances ===
        tip_to_tumor = self.distance_to_tumor()      # shape: (N,)
        tip_to_vessel = self.distance_to_vessel()    # shape: (N,)

        # === 2. Real-time collision score (e.g. from your scoring function or heuristics) ===
        # Dummy example: replace with your actual score function call
        #current_collision_score = self.get_real_time_collision_scores()  # shape: (N,)

        # === 3. Check if tip is inside tumor (binary mask for bonus) ===
        #inside_tumor = self.check_tip_inside_tumor()  # shape: (N,), bool tensor

        # === 4. Reward weights ===
        w_tumor_dist = 5.0
        w_vessel_penalty = 3.0
        w_collision_score = 1.5
        bonus_inside_tumor = 10.0

        # === 5. Compute total reward ===
        reward = (
            -w_tumor_dist * tip_to_tumor
            -w_vessel_penalty * tip_to_vessel
            #-w_collision_score * current_collision_score
            #+ bonus_inside_tumor * inside_tumor.float()
        )

        # === 6. Optional: clip or normalize if needed ===
        reward = torch.clip(reward, min=-100.0, max=100.0)

        return reward

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

        # Condition: within radius and within height range
        mask = (proj_lengths >= -height / 2) & (proj_lengths <= height / 2) & (radial_dists <= radius)
        return hits_np[mask]
    
    def boundary_check_tumor(self):
        hits = self.raycast_tumor.data.ray_hits_w
        for env_id in range(self.num_envs):
            hits_env = hits[env_id]  # shape (R, 3)
            valid_mask = torch.isfinite(hits_env).all(dim=-1)  # shape (R,)
            valid_hits = hits_env[valid_mask]  # shape (V, 3)
            tool_tip = self._robot.data.body_pos_w[:, self.tooltip_index]
            needle_center = tool_tip[env_id].cpu().numpy()
            # Assuming the tool's orientation provides the correct insertion axis
            insertion_axis = self._robot.data.body_quat_w[:, self.tooltip_index]  # This should be the rotation (orientation) of the tool
            insertion_axis_np = insertion_axis[env_id].cpu().numpy()
            axis = insertion_axis_np[1:]  # (x, y, z)
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
                
            if valid_hits_np.shape[0] != 0 and filtered_hits.shape[0] > 0:
                print("Valid hits shape:", valid_hits_np.shape, filtered_hits.shape)
                return filtered_hits  # Return filtered hits for further processing or visualization
            
                #self.draw_points(filtered_hits, color=(1.0, 0.0, 1.0, 1.0), size=4.0) 
                # Only draw cylinder for tumor hits (not for vessel hits)
                #self.draw_cylinder(center=needle_center, axis=insertion_axis, radius=0.05, height=0.25)

    def boundary_check_vessel(self):
        hits = self.raycast_vessel.data.ray_hits_w
        for env_id in range(self.num_envs):
            hits_env = hits[env_id]  # shape (R, 3)
            valid_mask = torch.isfinite(hits_env).all(dim=-1)  # shape (R,)
            valid_hits = hits_env[valid_mask]  # shape (V, 3)
            tool_tip = self._robot.data.body_pos_w[:, self.tooltip_index]
            needle_center = tool_tip[env_id].cpu().numpy()
            # Assuming the tool's orientation provides the correct insertion axis
            insertion_axis = self._robot.data.body_quat_w[:, self.tooltip_index]  # This should be the rotation (orientation) of the tool
            insertion_axis_np = insertion_axis[env_id].cpu().numpy()
            omni.log.info(f"Type: {type(insertion_axis_np)}")
            omni.log.info(f"Insertion Axis: {insertion_axis_np}")
            axis = insertion_axis_np[1:]  # (x, y, z)
            axis = axis / np.linalg.norm(axis)
            omni.log.info(f"Axis:{type(axis)}")
            valid_hits_np = valid_hits.cpu().numpy()
            filtered_hits = self.filter_hits_in_cylinder(
                hits_np=valid_hits_np,
                center=needle_center,
                axis=axis,
                radius=0.05,
                height=0.25
            )

            print(f"[env {env_id}] Hits inside cylinder Vessel: {filtered_hits.shape[0]}/{valid_hits_np.shape[0]}")
            self.pcd.points = o3d.utility.Vector3dVector(filtered_hits)
            sparse_points = self.pcd.farthest_point_down_sample(64)
            omni.log.info(f"type: {type(sparse_points)}")
            sparse_points = torch.tensor(np.asarray(sparse_points.points), dtype=torch.float32, device=self.device)
            # sparse_points_np = np.asarray(downsampled_pcd.points)
            # print(f"[env {env_id}] Sparse points shape: {sparse_points.shape}")
            if valid_hits_np.shape[0] != 0:
                print("Valid hits shape:", valid_hits_np.shape, filtered_hits.shape)
                return sparse_points
                #self.draw_points(filtered_hits, color=(0.0, 1.0, 1.0, 1.0), size=4.0) 
                # Only draw cylinder for vessel hits (if needed)
                #self.draw_cylinder(center=needle_center, axis=axis, radius=0.05, height=0.25)

    def distance_to_vessel(self):
        distances = self.raycast_cam_vessel.data.output["distance_to_camera"]
        if distances is None or distances.shape[0] == 0:
            print("[WARN] Raycast distances not yet populated.")
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        num_envs = distances.shape[0]
        H, W = distances.shape[1:3]
        total_rays = H * W

        max_dist = 0.1
        mean_dists = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        for env_id in range(num_envs):
            dists = distances[env_id, :, :, 0]
            valid = (~torch.isinf(dists)) & (dists <= max_dist)
            num_valid = valid.sum().item()
            if num_valid > 0:
                mean = dists[valid].mean()
                mean_dists[env_id] = mean
                print(f"[env {env_id}] Mean distance from tip to vessels (valid hits): {mean.item():.6f}")
            else:
                mean_dists[env_id] = 0.0
                print(f"[env {env_id}] No valid hits within {max_dist*1000:.1f} mm")
            print(f"[env {env_id}] Valid rays hitting the Vessels: {num_valid}/{total_rays}")
        return mean_dists

    def distance_to_tumor(self, env_ids=None):
        distances = self.raycast_cam_tumor.data.output["distance_to_camera"]
        if distances is None or distances.shape[0] == 0:
            print("[WARN] Raycast distances not yet populated.")
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        num_envs = distances.shape[0]
        H, W = distances.shape[1:3]
        total_rays = H * W

        max_dist = 0.1  
        mean_dists = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)

        for env_id in range(num_envs):
            dists = distances[env_id, :, :, 0]  # shape (H, W)
            valid = (~torch.isinf(dists)) & (dists <= max_dist)
            num_valid = valid.sum().item()
            if num_valid > 0:
                mean = dists[valid].mean()
                mean_dists[env_id] = mean
                print(f"[env {env_id}] Mean distance from tip to Tumor(valid hits): {mean.item():.6f}")
            else:
                mean_dists[env_id] = 0.0  # or float("inf") if you want to mark as invalid
                print(f"[env {env_id}] No valid hits within {max_dist*1000:.1f} mm")
            print(f"[env {env_id}] Valid rays hitting the Tumor: {num_valid}/{total_rays}")

        return mean_dists

    def brain_shift(self, points_attr, original_np, env_id):
        new_np = self.brain_shift_data[env_id][0]
        print(f"[INFO] [Env {env_id}] Original points shape: {original_np.shape}, New points shape: {new_np.shape}")
        assert new_np.shape == original_np.shape, f"Shape mismatch at env {env_id} ({new_np.shape} vs {original_np.shape})"
        usd_pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in new_np]
        points_attr.Set(usd_pts)
        prim_path = points_attr.GetPrim().GetPath().pathString
        print(f"[INFO] [Env {env_id}] Updated mesh points at {prim_path}")
        #self.raycast_vessel.update_dynamic_mesh_for_env(prim_path, new_np, env_id=0)
        stage_utils.update_stage()
        displacement = np.linalg.norm(new_np - original_np, axis=1)
        changed_mask = displacement > 1e-5
        changed_percent = 100.0 * np.sum(changed_mask) / displacement.shape[0]
        print(f"[INFO] [Env {env_id}] Vertices changed: {changed_percent:.2f}%")
        print(f"[INFO] [Env {env_id}] Mean displacement: {np.mean(displacement):.6f}")
        print(f"[INFO] [Env {env_id}] Max displacement:  {np.max(displacement):.6f}")

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
                    print(f"[DEBUG] Original points shape: {original_np.shape} at {prim_path}")
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