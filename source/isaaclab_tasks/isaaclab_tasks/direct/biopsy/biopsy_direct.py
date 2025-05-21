# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import math
import random
import omni.log
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.cloner import GridCloner
try:
    from isaacsim.util.debug_draw import _debug_draw
except ImportError:
    from omni.isaac.debug_draw import _debug_draw
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab_assets import UR5_CFG
from isaaclab.utils.io import dump_pickle, load_pickle
from . import biopsy_preop
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
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/vessels.obj"
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

    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    raycast_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor", "{ENV_REGEX_NS}/Vessel"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0) ,convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )


@configclass
class BiopsyDirectEnvCfg(DirectRLEnvCfg):
    #env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
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
    scene: MinimalSceneCfg = MinimalSceneCfg(num_envs=48, env_spacing=0.5, replicate_physics=True)

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0
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
    preop : biopsy_preop.BiopsyPreop

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

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("holder_needle_slider")[0]] = 0.1
        
        self.preop = biopsy_preop.BiopsyPreop()
        self.preop.print_test()
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/needle_tool/holder_link")
        print("Prim: ", prim)
        print("Is valid:", prim.IsValid())

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

        self.cloner = GridCloner(spacing=self.cfg.scene.env_spacing)
        self.draw = _debug_draw.acquire_debug_draw_interface()
        #read pickled data
        print("Loading tumor data...")
        self.tumor_positions = []
        self.tumor_quaternions = []
        self.tumor_centroids = []
        self.tumor_entry_points = []
        self.tumor_top_entry_points = []
        self.start_positions = []
        self.start_quaternions = []
        self.tumor_pickle = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_cleaned.pkl")
        for i in range(min(self.num_envs, len(self.tumor_pickle))):
            omni.log.info(f"Loading tumor data for env: {i}")
            try:
                env_data = self.tumor_pickle[i]
                for key in ["tumor_position", "tumor_quat", "tumor_centroid", "entry_points", "top_entry_points", "start_pose"]:
                    assert key in env_data, f"[ERROR] Missing key '{key}' in entry {i}"
                self.tumor_positions.append(env_data["tumor_position"])
                self.tumor_quaternions.append(env_data["tumor_quat"])
                self.tumor_centroids.append(env_data["tumor_centroid"])
                self.tumor_entry_points.append(env_data["entry_points"])
                self.tumor_top_entry_points.append(env_data["top_entry_points"])
                self.start_positions.append(env_data["start_pose"]["position"])
                self.start_quaternions.append(env_data["start_pose"]["quaternion"])
            except KeyError as e:
                omni.log.warn(f"KeyError: {e} for env {i}. Tumor data may be incomplete.")
            except AssertionError as e:
                omni.log.warn(f"AssertionError: {e} for env {i}. Tumor data may be incomplete.")
            except Exception as e:
                omni.log.warn(f"Exception: {e} for env {i}. Tumor data may be incomplete.")
        
        omni.log.info("Tumor data for envs loaded successfully.")
        self.draw_entry_points()
        self.draw_path()
        print("Tumor positions: ", len(self.tumor_positions), self.num_envs)

    def _setup_scene(self):
        """
        Setup the scene for the environment. But since InteractiveSceneCfg is used, the scene is already setup in the
        constructor of the parent class.

        To add new assets to the scene, use the `scene` attribute of the environment.
        refer https://forums.developer.nvidia.com/t/importing-scene-created-in-isaac-sim-to-isaac-lab/315590
        """
        self._robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self._robot

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self.robot_dof_targets)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        Compute intermediate values for the environment. This includes computing the action to be applied to the robot
        and the observations to be returned to the agent.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # reset robot joint positions

        #refresh intermediate variables so that _get_observations can use them
        self._compute_intermediate_values(env_ids)

    def _get_dones(self):
        pass

    def _get_observations(self):
        """
        Get the observations for the environment. This includes:
        a) robot joint positions
        b) robot joint velocities
        c) end-effector position and orientation
        d) distance to the tumor centroid
        e) top start poses
        f) visual observations from the raycaster camera
        g) collision scores
        """
        pass

    def _get_rewards(self):
        pass

    def _get_states(self):
        pass

    def _get_actions(self):
        pass

    def __compute_reward(self):
        pass
    
    def calculate_offsets(self, num_envs: int):
        offsets, _ = self.cloner.get_clone_transforms(num_envs)
        return offsets
    
    def draw_points(self, points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):    
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
        #print("Drawing line from", start_pose, "to", end_pose)
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
                offsets = self.calculate_offsets(self.num_envs)
                offset = offsets[env_id]
                offset_points = points + offset
                for point in offset_points:
                    self.draw_points([point], color=(0.2, 0.8, 0.2, 1.0), size=4.0)
            except Exception as e:
                omni.log.error(f"Exception: {e} for env {env_id}. Tumor data may be incomplete.")

    def draw_tumor_centroids(self):
        try:
            for env_id, points in enumerate(self.tumor_centroids):
                offsets = self.calculate_offsets(self.num_envs)
                offset = offsets[env_id]  # shape (3,)
                offset_points = points + offset  # (N, 3) + (3,) → (N, 3)
                self.draw_points([offset_points], color=(0.0, 0.0, 1.0, 1.0), size=4.0)        
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw tumor centroids: {e}")

    def draw_top_points(self):   
        try:
            for env_id, top_points in enumerate(self.tumor_top_entry_points):
                for i in range(len(top_points)):
                    offsets = self.calculate_offsets(self.num_envs)
                    offset = offsets[env_id]  # shape (3,)
                    offset_points = top_points[i]["entry_point"] + offset  # (N, 3) + (3,) → (N, 3)
                    self.draw_points([offset_points], color=(0.0, 1.0, 0.0, 1.0), size=4.0)
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw top entry points because: {e}")
        
    def draw_path(self):
        try:
            for env_id, top_points in enumerate(self.tumor_top_entry_points):
                tumor_center = self.tumor_centroids[env_id]
                offsets = self.calculate_offsets(self.num_envs)
                offset = offsets[env_id]  # shape (3,)

                for i in range(len(top_points)):
                    entry_point = top_points[i]["entry_point"]
                    p1 = entry_point + offset
                    p2 = tumor_center + offset

                    self.draw_lines(p1, p2, color="green")
        except Exception as e:
            omni.log.error(f"[ERROR] Failed to draw lines: {e}")