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
import time
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
import omni.kit.app
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
from isaaclab.utils.math import matrix_from_quat, skew_symmetric_matrix, quat_from_matrix, quat_inv, euler_xyz_from_quat, axis_angle_from_quat

# Visualization and markers
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Open3D for point cloud / mesh processing
import open3d as o3d
import matplotlib.pyplot as plt

SAVE_PATH = "/home/czlocal/sanjay_isaac/forked/IsaacLab/custom/path_comparison/plots/"
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

    # skull = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Skull",
    #     spawn=sim_utils.MeshFileCfg(
    #         file_path="/home/czlocal/sanjay_isaac/curobo/src/curobo/content/assets/scene/skull.obj"
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    # )

    vessel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Vessel",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/czlocal/sanjay_isaac/forked/Vessels.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    tumor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tumor",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/czlocal/sanjay_isaac/curobo_thesis_fork/src/curobo/content/assets/scene/tumor.obj"
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
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel", "{ENV_REGEX_NS}/Tumor"],
        update_period=1/60,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0) ,convention="world"),
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

    # raycast_camera_tumor = RayCasterCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/needle",
    #     mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
    #     update_period=0.1,
    #     offset=RayCasterCameraCfg.OffsetCfg(pos=(-0.0012, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0), convention="world"),
    #     data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
    #     debug_vis=False,
    #     max_distance=0.01,
    #     pattern_cfg=patterns.PinholeCameraPatternCfg(
    #         focal_length=24.0,
    #         horizontal_aperture=20.955,
    #         height=420,
    #         width=640,
    #     ),
    # )

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
    episode_length_s = 2.0833  # 250 timesteps
    decimation = 1
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
    scene: MinimalSceneCfg = MinimalSceneCfg(num_envs=1, env_spacing=0.2, replicate_physics=False)

    action_scale = 0.001
    dof_velocity_scale = 0.1

    # reward scales
    w_progress = 0.25
    w_deviation = 5.0
    w_collision = 1.0
    w_inside_tumor = 20.0
    w_action = 0.5  


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
            try:
                import isaacsim.examples.ui.extension as custom_ui
                self.ui_instance = custom_ui.EXTENSION_INSTANCE
                print("Custom UI extension imported successfully (post)")
            except Exception as e:
                print(f"Error importing omni.kit.app: {e}")

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
            self.REB = torch.tensor(0.001, device=self.device, dtype=torch.float32)  # mm
            self.PHI_Deg = torch.tensor(30, device=self.device, dtype=torch.float32)  # degrees
            self.CURV = torch.tensor(0.05, device=self.device, dtype=torch.float32)  # mm
            self.dt_steer = torch.tensor(0.5, device=self.device, dtype=torch.float32)  # seconds
            self.INSERTION_DEPTH = torch.tensor(0.001, device=self.device, dtype=torch.float32)  # mm
            self.TUMOR_REACH_THRESHOLD = torch.tensor(0.0075, device=self.device, dtype=torch.float32)  # mm
            self.DIST_THRESHOLD = torch.tensor(0.005, device=self.device, dtype=torch.float32)  # mm
            self.K_DEV = torch.tensor(10000.0, device=self.device, dtype=torch.float32)  # Deviation scaling factor
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
        self.tumor_pickle = load_pickle("/home/czlocal/sanjay_isaac/forked/IsaacLab/custom/path_comparison/pickle_finale/rl_dataset_100envs.pkl")  #/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_2205_cleaned.pkl
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
        
        # self.quaternions has shape [N, 10, 4]. I want to define each quaternion to be oriented such that 
        # negative y points in direction of tumor_centroids_tensor - start_positions_tensor which has shape [N, 10, 3]:
        self.start_quaternions_tensor = torch.zeros((self.num_envs, len(self.start_positions[0]), 4), dtype=torch.float32, device=self.device)  # (envs, poses, 4)
        for i in range(self.num_envs):
            for j in range(len(self.start_positions[0])):
                # Get the start position and tumor centroid for this environment and pose
                start_pos = self.start_positions_tensor[i, j]
                tumor_centroid = self.tumor_centroids_tensor[i]

                # Compute the direction vector from start position to tumor centroid
                direction_vector = tumor_centroid - start_pos
                direction_vector = direction_vector / torch.norm(direction_vector)
                # Create a rotation that aligns the negative y-axis with the direction vector
                negative_y = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=torch.float32)
                up_ref = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
                # if direction_vector ~ up_ref, use +x as fallback
                near = (torch.abs(torch.dot(direction_vector, up_ref)) > 0.999).all()
                ref = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32) if near else up_ref
                r = torch.cross(ref, direction_vector)
                r = r / torch.norm(r)
                u = torch.cross(direction_vector, r)
                x_col = r
                y_col = -direction_vector
                z_col = u
                R = torch.stack([x_col, y_col, z_col], dim=1)  # Create rotation matrix
                quat = quat_from_matrix(R)  # Convert rotation matrix to quaternion
                
                #dot_product = torch.dot(negative_y, direction_vector)
                #axis = torch.cross(negative_y, direction_vector)
                #axis = axis / torch.norm(axis)
                #angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
                #half_angle = angle / 2.0
                #w = torch.cos(half_angle)
                #xyz = axis * torch.sin(half_angle)
                self.start_quaternions_tensor[i, j] = quat
        self.start_quaternions = self.start_quaternions_tensor.cpu().numpy().tolist()  # Convert to list for compatibility with other code
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
        shift_data = load_pickle("/home/czlocal/sanjay_isaac/forked/IsaacLab/custom/path_comparison/path_comparison/precomputed_brain_deformations100envs.pkl")
        # print(f"[INFO] Loaded brain shift data for {len(shift_data)} envs")
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
        
        # try:   # TODO Uncomment if needed
        #     print(f"[INFO] Brain shift data for envs: {len(self.brain_shift_data)}")
        #     self.get_vessel_points()
        # except Exception as e:
        #     omni.log.error(f"Error getting vessel points: {e}")
        #     traceback.print_exc()

        self.stage = stage_utils.get_current_stage()
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.set_tumor_positions()

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        self.needle_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/World/envs/env_0/needle/geometry/mesh"))
        self.raycast_cam_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/World/envs/env_0/raycast_camera_vessel/geometry/mesh")
        )
        self.raycast_vessel_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/World/envs/env_0/raycast_vessel/geometry/mesh")
        )

        # Observation space terms:
        self.pcd = o3d.geometry.PointCloud()  # Placeholder for point cloud data
        self.prev_normalized_progress = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_deviation = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_tooltip_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.danger_bins = torch.zeros((self.num_envs, 16), dtype=torch.float32, device=self.device)  # Danger bins for the raycaster

        # Rewards
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.d_ttip_tumor = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)  # Distance from tooltip to tumor
        self.metrics_log = {
            "distance": [[] for _ in range(self.num_envs)],
            "normalized_progress": [[] for _ in range(self.num_envs)],
            "deviation": [[] for _ in range(self.num_envs)],
            "projected_dist": [[] for _ in range(self.num_envs)],
            "path_length": [[] for _ in range(self.num_envs)],
        }

        # UI
        self.success_counter = 0


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
        #print("New action update received at", self.common_step_counter) 
        if isinstance(self.single_action_space, gym.spaces.Box):
            low = torch.tensor(self.single_action_space.low, device=self.device)
            high = torch.tensor(self.single_action_space.high, device=self.device)
            # print(f"Picked actions: {self.actions}")
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
            # print(f"Picked actions (Discrete): {self.actions}")
            # For discrete actions, we assume the action is an index into a lookup table
            insertion_depths = torch.full((self.num_envs,), self.INSERTION_DEPTH.item(), device=self.device, dtype=torch.float32)  # Default depth
            twist_angles_deg = self.actions.float() * 22.5  # 0.0 degrees for no twist
            twist_angles_rad = torch.deg2rad(twist_angles_deg)
            # print(f"Insertion depths: {insertion_depths}")
            # print(f"Twist angles (deg): {twist_angles_rad}")
            self.actions = torch.stack([insertion_depths.view(-1), twist_angles_rad.view(-1)], dim=1)

            # print(f"Updated actions: {self.actions}")
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
            print(f"Twist angles (deg): {twist_angles_deg}")
            print(f"Twist angles (rad): {twist_angles_rad}")
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
        prior_paths = self.get_prior_paths(start_pose, self.tumor_centroids_tensor) # TODO: Possible Bug source since this calculates path in global pose from the pickle - No because i already added offsets
        insertion_depths = self.actions[:, 0]  
        twist_angles = self.actions[:, 1]     
        next_poses = []
        for i in range(self.num_envs):
            # direction vector (start → tumor)
            preop_direction = self.tumor_centroids_tensor[i] - start_pose[i]
            preop_direction = preop_direction / torch.norm(preop_direction)

            # next_pose = self.generate_needle_step_with_rebound(
            #     current_pose=current_pose[i],
            #     insertion_depth=insertion_depths[i],
            #     twist_angle_rad=twist_angles[i],
            #     prior_path=prior_path,
            # )
            # Compute next pose
            # print("Calling for next pose", self.common_step_counter)
            # print("PREOP DIRECTION---------", preop_direction)
            # print("EULER ANGLE ----------", euler_xyz_from_quat(quat))
            next_pose = self.generate_needle_step_ludwig(
                current_pose=current_pose[i],
                insertion_depth=insertion_depths[i],
                preop_path=preop_direction,
                roll_delta_rad=twist_angles[i],
            )
            next_poses.append(next_pose)

        next_poses = torch.stack(next_poses)

        new_pos = next_poses[:, :3, 3]
        new_rot = next_poses[:, :3, :3]
        new_quat = quat_from_matrix(new_rot)
        #new_quat = torch.stack([new_quat[:, 3], new_quat[:, 0], new_quat[:, 1], new_quat[:, 2]], dim=-1)

        new_root_state[:, :3] = new_pos
        new_root_state[:, 3:7] = new_quat
        self._needle.write_root_pose_to_sim(new_root_state[:, :7])
        self._needle.write_root_velocity_to_sim(torch.zeros_like(new_root_state[:, 7:]))
        
        #self._needle.reset()
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
                #self.draw_points(prior_positions, color=(1.0, 0.0, 0.0, 1.0), size=2.0)


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

            # print(f"[RESET] Env {env_id} start → Pose: {pos.cpu().numpy()}, Quat: {quat.cpu().numpy()}")

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
        insertion_depths = torch.full((self.num_envs,), self.INSERTION_DEPTH.item(), device=self.device)
        twist_angles_rad = torch.zeros((self.num_envs,), device=self.device)  # or any dummy twist
        self.actions = torch.stack([insertion_depths, twist_angles_rad], dim=1)  # Shape: [B, 2]
        self.previous_twist_action[env_ids] = 0.0 
        self.danger_bins = torch.zeros((self.num_envs, 16), dtype=torch.float32, device=self.device) # reset bins to zero after reset as there are no update calls and hence no values
        # # Reset tumor poses using sample_uniform TODO: Uncomment if needed
        # tumor_world_poses = self.tumor.get_world_poses(env_ids)
        # position, quaternion = tumor_world_poses
        # omni.log.info(f"Initial tumor world poses: {tumor_world_poses}")
        # try:
        #     omni.log.info(f"Resetting tumor poses for env_ids: {env_ids}, Tumor world poses: {tumor_world_poses[0]}")
        #     sampled_offset = sample_uniform(lower=-0.001, upper=0.001, size=(len(env_ids), 3), device=self.device)
        #     updated_position = position + sampled_offset
        #     # stack updated poses with quaternions from tumor_world_poses
        #     self.tumor.set_world_poses(positions=updated_position, orientations=quaternion, indices=env_ids)
        #     # print(f"[INFO] Tumor poses reset success for env_ids: {env_ids}")
        # except Exception as e:
        #     omni.log.error(f"Error sampling tumor poses due to: {e}")
        # omni.log.info(f"Calling _reset_idx for env_ids: {env_ids}")
        self._compute_intermediate_values(env_ids)

    def _get_dones(self):
        """
        Success  = Tooltip is close to the tumor (≤ TUMOR_REACH_THRESHOLD in meters)
                AND aligned with the end of the path (s ≈ 1 ± tolerance).
        Truncate = Timeout OR overshoot beyond valid path range.
        """
        (normalized_progress,
        deviation,
        perpendicular_vector,
        closest_point,
        path_length,
        d_ttip_tumor,
        d_start_tumor,
        projected_dist) = self.calc_normalized_progress()
        eps = 1e-8
        s_unclamped = projected_dist / (path_length + eps)
        dist_thresh = self.TUMOR_REACH_THRESHOLD      
        s_tol = getattr(self, "S_PROGRESS_TOLERANCE", 1e-3)  
        self.d_ttip_tumor = d_ttip_tumor.clone()
        success = (
            (d_ttip_tumor <= dist_thresh) # &
            #(s_unclamped >= 1.0 - s_tol) &
            #(s_unclamped <= 1.0 + s_tol)
        )
        time_out = (self.episode_length_buf >= self.max_episode_length - 1)
        overshoot_positive = s_unclamped > (1.0 + s_tol)
        overshoot_negative = s_unclamped < (0.0 - s_tol)
        truncated = (~success) & (time_out | overshoot_positive | overshoot_negative)
        self.extras.update({
            "success": success,
            "truncated": truncated
        })
        return success, truncated



    def _get_observations(self):
        """
        Get the observations for the environment. This includes:
        - Tooltip Pose
        - Normalized Depth to Tumor at time-step t and t-dt
        - Direction to Tumor Centroid (via straight-line path basis)
        - Signed lateral offsets (Y/Z) to the path
        - Heading components (t, y, z)
        - Previous Action (twist angle)
        """
        # Compute straight-line progress and geometry (analytic)
        (normalized_progress,
        deviation,
        perpendicular_vector,
        closest_point,
        path_length,
        d_ttip_tumor,
        d_start_tumor,
        projected_dist) = self.calc_normalized_progress()

        self.d_ttip_tumor = d_ttip_tumor.clone()

        normalized_progress_t_ndt = self.prev_normalized_progress.clone()
        prev_deviation_t_ndt = self.prev_deviation.clone()
        self.prev_normalized_progress = normalized_progress.detach()
        self.prev_deviation = deviation.detach()

        path_vec = self.path_direction    

        position = self.tool_tip_pos
        quaternion = self.tool_tip_rot
        current_pose_matrix = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)  # [B, 4, 4]
        current_pose_matrix[:, :3, 3] = position
        current_pose_matrix[:, :3, :3] = matrix_from_quat(quaternion)
        
        closest_point_local = self.get_vector_in_needle_frame(closest_point, current_pose_matrix)
        signed_delta_x = closest_point_local[:, 0]
        signed_delta_z = closest_point_local[:, 2]   

        omni.log.info(f"Signed Delta X: {signed_delta_x}, Signed Delta Z: {signed_delta_z}")

        # Take path vec, translate it to needle_position, then express in needle frame:
        translated_heading_vec = path_vec + self.tool_tip_pos
        heading_vec_needle_frame = self.get_vector_in_needle_frame(translated_heading_vec, current_pose_matrix)  # [B, 3]
        heading_vec_needle_frame = heading_vec_needle_frame / torch.norm(heading_vec_needle_frame, dim=-1, keepdim=True)  # Normalize to unit vector
        heading_x = heading_vec_needle_frame[:, 0]
        heading_y = heading_vec_needle_frame[:, 1]
        heading_z = heading_vec_needle_frame[:, 2]
        danger_bins = self.boundary_check_vessel()
        self.extras.update({
            "env_ids": self.env_ids.clone().detach(),                              # [B]
            "active_path_index": self.active_path_index.clone().detach(),          # [B]
            "normalized_progress_t": normalized_progress.clone().detach(),         # [B]
            "normalized_progress_t_ndt": normalized_progress_t_ndt.clone().detach(),  # [B]
            "deviation_t": deviation.clone().detach(),                             # [B]
            "deviation_t_ndt": prev_deviation_t_ndt.clone().detach(),              # [B]
            "signed_delta_x": signed_delta_x.clone().detach(),                     # [B, 1]
            "signed_delta_z": signed_delta_z.clone().detach(),                     # [B, 1]
            "heading_x": heading_x.clone().detach(),                               # [B, 1]
            "heading_y": heading_y.clone().detach(),                               # [B, 1]
            "heading_z": heading_z.clone().detach(),                               # [B, 1]
            "d_ttip_tumor": d_ttip_tumor.clone().detach(),                         # [B]
            "tooltip_pos": self.tool_tip_pos.clone().detach(),                     # [B, 3]
            "tooltip_rot": self.tool_tip_rot.clone().detach(),                     # [B, 4]
        })

        # --- Assemble observations ---
        obs = {
            "current_action": self.actions[:, 1].unsqueeze(-1),        # [B,1]
            "normalized_depth_t": normalized_progress.unsqueeze(-1),   # [B,1]
            "normalized_depth_t_ndt": normalized_progress_t_ndt.unsqueeze(-1),
            "deviation_t": deviation.unsqueeze(-1),                    # [B,1]
            "deviation_t_ndt": prev_deviation_t_ndt.unsqueeze(-1),     # [B,1]
            "signed_delta_x": signed_delta_x,                          # [B,1]
            "signed_delta_z": signed_delta_z,                          # [B,1]
            "heading_x": heading_x,                                    # [B,1]
            "heading_y": heading_y,                                    # [B,1]
            "heading_z": heading_z,                                    # [B,1]
            "danger_bins": self.danger_bins,                        # [B, 16]
        }
        # print(f"[DEBUG-STEP] Observations at step {self.common_step_counter}: {obs}")
        self.boundary_check_vessel()
        self.distance_to_vessel()

        if self.num_envs == 1:
            self.update_obs_ui(obs["heading_x"], obs["heading_y"], obs["heading_z"], obs["signed_delta_x"], obs["signed_delta_z"], obs["deviation_t"], self.d_ttip_tumor)#, self.danger_bins)
        self.prev_obs_for_reward = {k: v.clone().detach() for k, v in obs.items()}
        # print(f"Observation at {self.common_step_counter} are {self.prev_obs_for_reward}")
        return {"policy": obs}

    def _get_rewards(self):  # TODO: to get calculated Rewards
        """
        Get the rewards for the environment. This includes:
        a) RayCaster Camera reward based on distance to image plane and distance to camera

        """
        # print(f"[DEBUG-STEP] Rewards called at time step: {self.common_step_counter}, {self._sim_step_counter}")
        obs = self.prev_obs_for_reward
        progress_t = obs["normalized_depth_t"]             
        progress_t_dt = obs["normalized_depth_t_ndt"]      
        deviation_t = obs["deviation_t"]                          
        action = obs["current_action"]      
        d_ttip_tumor = self.d_ttip_tumor.clone()  # Distance from tooltip to tumor     
        danger_bins = obs["danger_bins"]         
        total_reward = self.compute_reward(progress_t, progress_t_dt, deviation_t, action, d_ttip_tumor, danger_bins)
        if self.order_checker:
            self.order_checker+=1
            print(f"[ORDER DEBUG]Order at Rewards step: {self.order_checker}")
        return total_reward

    def _get_states(self):  # TODO: States for Asymmetric RL
        pass

    #@torch.jit.script
    def compute_reward(self, progress_t, progress_t_dt, deviation_t, action, d_ttip_tumor, danger_bins):
        """
        Reward structure:
        + reward for being closer to tumor
        + reward for lower deviation from preop path
        + reward for progressing towards the tumor
        - penalty for high real-time collision score
        """
        progress = (progress_t - progress_t_dt).squeeze(-1)
        reward_progress = torch.clamp(progress / (self.INSERTION_DEPTH + 1e-9), -1.0, 1.0)
        reward_deviation = torch.exp(-self.K_DEV * deviation_t.squeeze(-1) ** 2)
        reward_reached_tumor = (d_ttip_tumor <= self.TUMOR_REACH_THRESHOLD).float()
        a = action if action.dim() == 1 else action.squeeze(-1)
        d_mm = deviation_t.squeeze(-1) * 1000.0  
        reward_deviation_new = torch.exp(-0.2 * d_mm)    
        reward_deviation_new = torch.where(d_mm > 20.0, -5.0, reward_deviation_new) 
        #print(f"Reward deviation old {reward_deviation} vs Reward deviation new {reward_deviation_new}")
        a_deg = torch.rad2deg(a) % 360.0                   
        offset = 11.25
        a_bin = torch.div((a_deg - offset + 360.0) % 360.0, 22.5, rounding_mode="floor").long()


        if danger_bins.dim() == 1:
            danger_bins = danger_bins.unsqueeze(0)

        danger_a = torch.gather(danger_bins, 1, a_bin.unsqueeze(-1)).squeeze(-1)

        reward_collision = torch.zeros_like(danger_a)
        reward_collision = torch.where(danger_a == 1.0, -50.0, reward_collision)
        reward_collision = torch.where(danger_a == 0.5, -25.0, reward_collision)
        reward_collision = torch.where(danger_a == 0.0, +0.0, reward_collision)

        if self.previous_twist_action is not None:
            reward_action = torch.abs(a_bin - self.previous_twist_action.long()) / 16.0
        else:
            reward_action = torch.abs(a_bin) / 16.0

        reward = (
            (self.cfg.w_progress * reward_progress)
            + (self.cfg.w_deviation * reward_deviation_new)
            + (self.cfg.w_inside_tumor * reward_reached_tumor)
            + (self.cfg.w_collision * reward_collision)
        )
        reward = torch.clip(reward, min=-100.0, max=100.0)

        # -------------------
        # Debug logs
        # -------------------
        print("\n=== REWARD DEBUG ===")
        print(f"Original action (rad): {a}")
        print(f"Action in degrees    : {a_deg.detach().cpu().numpy()}")
        print(f"Action bin index     : {a_bin.detach().cpu().numpy()}")
        print(f"Danger bins          : {danger_bins.detach().cpu().numpy()}")
        print(f"Danger value chosen  : {danger_a.detach().cpu().numpy()}")
        print(f"Reward progress      : {reward_progress.detach().cpu().numpy()}")
        print(f"Reward deviation     : {reward_deviation_new.detach().cpu().numpy()}")
        print(f"Reward reached tumor : {reward_reached_tumor.detach().cpu().numpy()}")
        print(f"Reward collision     : {reward_collision.detach().cpu().numpy()}")
        print(f"Final reward         : {reward.detach().cpu().numpy()}")
        print("====================\n")

        if reward_collision.item() == -50.0:
            time.sleep(1)
        # -------------------
        # Update buffers
        # -------------------
        self.extras.update({
            "reward_progress": self.cfg.w_progress * reward_progress,
            "reward_deviation": self.cfg.w_deviation * reward_deviation_new,
            "reward_reached_tumor": self.cfg.w_inside_tumor * reward_reached_tumor,
            "reward_collision": self.cfg.w_collision * reward_collision,
        })

        self.previous_twist_action = a_bin.clone().detach()
        return reward

    # ## --------------------------------------- ## #
    # ## Additional Utility Functions for Visualization and Debugging ## #
    # ## --------------------------------------- ## #

    def update_reset_ui(self, success_counter):
        if self.ui_instance is not None:
            self.ui_instance._models["success"].set_value(success_counter)
            

    def update_obs_ui(
        self,
        heading_x,        # Heading (T)
        heading_y,        # Heading Y plane
        heading_z,        # Heading Z plane
        signed_delta_x,   # Signed ΔX
        signed_delta_z,   # Signed ΔZ
        deviation_t,      # Lateral deviation
        d_ttip_tumor,     # Distance tip→tumor
        danger_bins       # Danger bins
    ):
        #print(f"[DEBUG]: {heading_t}, {heading_y}, {heading_z}, {signed_delta_y}, {signed_delta_z}, {deviation_t}, {d_ttip_tumor}")
        # helper: tensor/ndarray/python -> clean float 
        def _f(x):
            try:
                return float(getattr(x, "squeeze", lambda: x)())
            except Exception:
                return float(x)
        self.ui_instance._models["step"].set_value(self.common_step_counter)
        # value per plot (in the same order you created them)
        values = [
            ("_plot_data",   "timeseries_plot",   "timeseries_plot_val",   _f(heading_x)),
            ("_plot_data_1", "timeseries_plot_1", "timeseries_plot_val_1", _f(heading_y)),
            ("_plot_data_2", "timeseries_plot_2", "timeseries_plot_val_2", _f(heading_z)),
            ("_plot_data_3", "timeseries_plot_3", "timeseries_plot_val_3", _f(signed_delta_x)),
            ("_plot_data_4", "timeseries_plot_4", "timeseries_plot_val_4", _f(signed_delta_z)),
            ("_plot_data_5", "timeseries_plot_5", "timeseries_plot_val_5", _f(deviation_t)),
            ("_plot_data_6", "timeseries_plot_6", "timeseries_plot_val_6", _f(d_ttip_tumor)),
        ]

        # push each value and refresh its plot
        for buf_attr, plot_key, val_key, v in values:
            buf = getattr(self.ui_instance, buf_attr, None)
            if buf is None:
                buf = [0.0] * 360
                setattr(self.ui_instance, buf_attr, buf)
            buf.append(v)
            if len(buf) > 360:
                buf.pop(0)

            plot = self.ui_instance._models.get(plot_key)
            val_model = self.ui_instance._models.get(val_key)
            if plot is not None:
                plot.set_data(*buf)          
            if val_model is not None:
                val_model.set_value(v)

    def visualize_vessel_danger(self, env_id: int,
                                filtered_hits: torch.Tensor,
                                current_pose: torch.Tensor,
                                n_sectors: int = 16,
                                radius_m: float = 0.002,     # scan radius in meters
                                threshold_cm: float = 0.05,
                                plot: bool = True): # threshold in cm (e.g. 0.2 cm = 2 mm)
        """
        Visualize rays around needle tip, colored by vessel proximity.

        Args:
            env_id: environment index
            filtered_hits: (N,3) vessel points in world frame
            current_pose: (4,4) pose of needle in world frame
            n_sectors: number of angular sectors
            radius_m: maximum OCT/cylinder radius in meters
            threshold_cm: distance threshold for "dangerous" in centimeters
        """

        if filtered_hits is None or filtered_hits.numel() == 0:
            print(f"[env {env_id}] NO FILTERED HITS")
            return torch.zeros(n_sectors, dtype=torch.float32, device=self.device)

        tip = current_pose[:3, 3].cpu().numpy()   
        R = current_pose[:3, :3].cpu().numpy()    

        hits_local = self.get_vector_in_needle_frame(
            filtered_hits.unsqueeze(0),          
            current_pose.unsqueeze(0)             
        ).squeeze(0).cpu().numpy()                

        x, y = hits_local[:, 0], hits_local[:, 1]
        r = np.sqrt(x**2 + y**2)                  
        ang = np.arctan2(y, x)
        ang2 = (ang + 2*np.pi) % (2*np.pi)
        offset = np.deg2rad(11.25)  # ≈ 0.19635 rad
        ang_shifted = (ang2 - offset + 2*np.pi) % (2*np.pi)
        bins = np.floor(ang_shifted * n_sectors / (2*np.pi)).astype(int)


        r_min = np.full(n_sectors, np.inf)
        #print(f"Original R_MIN: {r_min}")
        closest_idx = np.full(n_sectors, -1)
        for i, sec_id in enumerate(bins):
            if r[i] < r_min[sec_id]:
                r_min[sec_id] = r[i]
                closest_idx[sec_id] = i
        
        needle_radius = 0.002 / 2     # 1.0 mm
        safety_margin = 0.0005        # 0.5 mm
        threshold_m = needle_radius + safety_margin

        danger = np.zeros(n_sectors)
        for b in range(n_sectors):
            if np.isfinite(r_min[b]):
                if r_min[b] <= threshold_m:
                    danger[b] = 1.0   # definite collision (inside needle body + safety margin)
                elif r_min[b] <= radius_m:
                    # soft penalty: closer → higher danger
                    danger[b] = max(0.0, 1.0 - (r_min[b] - needle_radius) / (radius_m - needle_radius))
                else:
                    danger[b] = 0.0
            #print(f"[env {env_id}] sector {b}: r_min={r_min[b]:.3f}, danger={danger[b]:.3f}")
        hits_world = (hits_local @ R.T) + tip
        for b in range(n_sectors):
            idx = closest_idx[b]
            if idx >= 0:
                end_world = hits_world[idx]
                if danger[b] == 1.0:
                    col = "red"
                elif danger[b] > 0.5:
                    col = "yellow"
                else:
                    col = "green"
                self.draw_lines(tip, end_world, color=col)

            # --- Matplotlib polar plot ---
        if plot is True:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

            # Plot all hits
            ax.scatter(ang2, r, s=10, c="blue", alpha=0.5, label="Vessel hits")

            # Plot closest hit per sector, colored by danger
            for b in range(n_sectors):
                idx = closest_idx[b]
                if idx >= 0:
                    if danger[b] > 0.6:
                        col = "red"
                    elif danger[b] > 0.2:
                        col = "yellow"
                    else:
                        col = "green"
                    ax.scatter(ang2[idx], r[idx], s=40, c=col, edgecolors="k", zorder=3)

            ax.set_ylim([0, radius_m])
            ax.set_title(f"Env {env_id} Vessel Danger Map")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_xticks(np.linspace(0, 2*np.pi, n_sectors, endpoint=False))
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{SAVE_PATH}/{datetime_str}_env_{env_id}_vessel_danger_map.png", bbox_inches="tight")
            plt.close(fig)

        #print(f"[env {env_id}] sector danger: {danger} {danger.shape}")
        danger_tensor = torch.tensor(danger, dtype=torch.float32, device=self.device)
        return danger_tensor


    def get_vector_in_needle_frame(self, vector: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert a vector from world frame to needle frame using the current pose.
        :param vector: Tensor of shape (B, 3) representing the vector in world frame.
        :param current_pose: Tensor of shape (B, 4, 4) representing the current pose of the needle.
        :return: Tensor of shape (B, 3) representing the vector in needle frame.
        """
        # print("current pose: ", current_pose)
        # print("vector: ", vector)
        R = current_pose[:, :3, :3]
        p = current_pose[:, :3, 3]
        # Convert vector to needle frame
        vector_in_needle_frame = (R.transpose(-1, -2) @ (vector - p).unsqueeze(-1)).squeeze(-1)
        return vector_in_needle_frame
    
    def _rot_y(self, theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.eye(3, device=self.device, dtype=torch.float32)
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
        return R

    def _rot_x(self, theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta); s = torch.sin(theta)
        R = torch.eye(3, device=self.device, dtype=torch.float32)
        R[1,1], R[1,2], R[2,1], R[2,2] = c, -s, s, c
        return R
    
    def _rot_y(self, angle):
        c = torch.cos(angle); s = torch.sin(angle)
        R = torch.eye(3, device=self.device, dtype=torch.float32)
        R[0,0] =  c;  R[0,2] =  s
        R[2,0] = -s;  R[2,2] =  c
        return R

    def _rot_z(self, phi: torch.Tensor) -> torch.Tensor:
        c = torch.cos(phi); s = torch.sin(phi)
        R = torch.eye(3, device=self.device, dtype=torch.float32)
        R[0,0], R[0,1], R[1,0], R[1,1] = c, -s, s, c
        return R

    def _make_T(self, R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        T = torch.eye(4, device=self.device, dtype=torch.float32)
        T[:3, :3] = R
        T[:3,  3] = p
        return T
    
    def generate_needle_step_ludwig(self,
                                current_pose: torch.Tensor,  # (4,4)
                                insertion_depth: float,
                                preop_path: torch.Tensor,
                                roll_delta_rad: float        # φ (radians), agent action
                                ) -> torch.Tensor:
        """
        Constant-curvature step in the needle frame with chosen 
        roll angle around instrument axis
        T_next = T_current * [ Rz(φ) Rx(θ),  Rz(φ) p0 ; 0 1 ],
        with θ = κ s, p0 = [0, R(1 - cosθ), R sinθ], R = 1/κ.
        """
        # fixed step and curvature
        #
        # roll_delta_rad = 0.0
        s = torch.as_tensor(insertion_depth, device=self.device, dtype=torch.float32)
        #TODO: Check if kappa should have 1/mm or 1/m as unit, IMPORTANT
        kappa = torch.as_tensor(400, device=self.device, dtype=torch.float32) # assuming radius of 50mm --> 1/50mm - 20 if in metres
        roll_angle  = torch.as_tensor(roll_delta_rad, device=self.device, dtype=torch.float32)
        #print("________________________________",roll_angle)
        # bend angle and radius
        theta = kappa * s
        Rcurv = 1.0 / kappa

        # exact circular-arc translation in the unrolled frame
        cos_th = torch.cos(theta); 
        sin_th = torch.sin(theta)
        p0  = torch.stack([ torch.tensor(0.0, device=self.device),
                            -Rcurv * sin_th,                 
                            Rcurv * (1 - cos_th) ])
        
        #p0 = torch.zeros_like(p0, device=self.device, dtype=torch.float32)  # [3]

        # roll then bend (body update)
        Ry = self._rot_y(roll_angle) # was -roll_angle
        Rx = self._rot_x(theta) 
        # Rx = self._rot_x(theta)
        R_inc = Ry @ Rx
        p_inc = Ry @ p0

        T_inc = self._make_T(R_inc, p_inc)
        # Set T_inc to identity rotation and translation in positive z:
        #T_inc[:3, :3] = torch.eye(3, device=self.device, dtype=torch.float32)
        #path_length = torch.norm(preop_path)
        #T_inc[:3, 3] = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        return current_pose @ T_inc

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
        """
            normalized_progress: [B] in [0,1]
            deviation:           [B] shortest distance to the segment
            perpendicular_vec:   [B,3] vector from closest point on segment to tooltip
            path_length:         [B]
            d_ttip_tumor:        [B] |tooltip - tumor|
            d_startpos_tumor:    [B] |start - tumor|
            projected_dist:      [B] signed projection length along the (unclamped) line
            self.path_direction:           [B,3] (unit)
            self.closest_point_on_path:    [B,3]
        """
        # Needle state
        root_state = self._needle.data.root_state_w.clone()
        self.tool_tip_pos = root_state[:, :3]      # [B,3]
        self.tool_tip_rot = root_state[:, 3:7]     # [B,4]

        # Start pose and tumor pose
        start_pose = self.get_start_pose_active(num_envs=self.num_envs)       # [B,3]
        tumor_pos = self.tumor_centroids_tensor    # [B,3], [B,4]

        # Segment geometry
        path_vector = tumor_pos - start_pose                                   # [B,3]
        path_length = torch.norm(path_vector, dim=-1, keepdim=True) + 1e-8     # [B,1]
        path_direction = path_vector / path_length                             # [B,3]

        tooltip_vec = self.tool_tip_pos - start_pose                           # [B,3]
        t = torch.sum(tooltip_vec * path_direction, dim=-1, keepdim=True)      # [B,1] 

        # Clamp to the *segment* [0, |path|]
        min_val = torch.zeros_like(t)  
        t_clamped = torch.clamp(t, min=min_val, max=path_length)

        # Closest point and perpendicular info
        closest_point = start_pose + t_clamped * path_direction                # [B,3]
        perpendicular_vec = self.tool_tip_pos - closest_point                  # [B,3]
        deviation = torch.norm(perpendicular_vec, dim=-1)                      # [B]

        # Progress in [0,1]
        normalized_progress = torch.clamp((t / path_length).squeeze(-1), 0.0, 1.0)  # [B]

        # Distances for logging/metrics
        d_ttip_tumor = torch.norm(self.tool_tip_pos - tumor_pos, dim=-1)       # [B]
        d_startpos_tumor = torch.norm(start_pose - tumor_pos, dim=-1)          # [B]

        # Bookkeeping for downstream use
        self.path_direction = path_direction                                   # [B,3]
        self.closest_point_on_path = closest_point                             # [B,3]

        # Logs
        omni.log.info(f"Deviation: {deviation}")
        omni.log.info(f"Normalized progress: {normalized_progress}, Deviation: {deviation}")
        omni.log.info(f"Tooltip position: {self.tool_tip_pos}, Tumor position: {tumor_pos}")
        #omni.log.info(f"Shape of tumor positions: {tumor_pos.shape}, Tumor quaternion: {tumor_quat.shape}, tooltip: {self.tool_tip_pos.shape}")
        omni.log.info(f"Distance to tumor from tooltip: {d_ttip_tumor}, Distance from start position to tumor: {d_startpos_tumor}")

        # projected_dist uses the *unclamped* projection (can be <0 or >|path|), like before
        projected_dist = t.squeeze(-1)                                         # [B]

        return (normalized_progress,
                deviation,
                perpendicular_vec,
                closest_point,
                path_length.squeeze(-1),
                d_ttip_tumor,
                d_startpos_tumor,
                projected_dist)

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
    
    def generate_needle_step(
        self,
        current_pose: torch.Tensor,            # SE(3), shape (4, 4)
        insertion_depth: float,                # mm
        twist_angle_rad: float,                # radians
    ) -> torch.Tensor:  
        """
        Angle-only needle step (no rebound; no path inside dynamics).
        Implements: g_next = g * Rz(dpsi) * exp(feed_twist(s) ) in the BODY frame,
        using curvature κ = 0.02 mm^-1 (r = 50 mm).
        """

        phi = torch.deg2rad(self.PHI_Deg)  # bevel angle in radians
        s   = torch.as_tensor(insertion_depth, device=self.device, dtype=torch.float32)  # mm
        dpsi= torch.as_tensor(twist_angle_rad, device=self.device, dtype=torch.float32)  # rad

        kappa = torch.tensor(0.02, device=self.device, dtype=torch.float32)  # 1/mm for r=50 mm
        wx = s * kappa  

        v_local = torch.stack((
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
            s * torch.sin(phi),
            -s * torch.cos(phi)
        ))

        w_local = torch.stack((
            wx,
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device, dtype=torch.float32)  # spin applied separately
        ))

        xi_hat = self.twist_to_matrix(v_local, w_local)

        cz, sz = torch.cos(dpsi), torch.sin(dpsi)
        Rz = torch.eye(4, device=self.device, dtype=torch.float32)
        Rz[:3, :3] = torch.stack((
            torch.stack((cz, -sz, torch.tensor(0.0, device=self.device))),
            torch.stack((sz,  cz, torch.tensor(0.0, device=self.device))),
            torch.stack((torch.tensor(0.0, device=self.device),
                        torch.tensor(0.0, device=self.device),
                        torch.tensor(1.0, device=self.device)))
        ))

        # --- compose: spin then arc ---
        next_pose = current_pose @ Rz @ torch.linalg.matrix_exp(xi_hat)
        return next_pose

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

    def filter_hits_in_cylinder(self, hits_np, center, axis, radius=0.005, height=0.005):
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
                radius=0.005,
                height=0.005
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
        rotations = matrix_from_quat(quats) 
        current_pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1) 
        current_pose[:, :3, :3] = rotations
        current_pose[:, :3, 3] = positions

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
                radius=0.002,
                height=0.003,
            )
            if not self.cfg.viewer.headless and filtered_hits is not None:
                #self.draw_points(filtered_hits, color=(1.0, 0.0, 0.0, 1.0), size=4.0)
                try:
                    start = self.tool_tip_pos[env_id] 
                    end = filtered_hits           
                    start_batch = start.repeat(end.shape[0], 1)  
                    self.draw_lines(start_batch, end, color="red")

                except Exception as e:
                    print(f"Error drawing lines: {e}")

            #print("Calling DANGER visualization")
            danger_row = self.visualize_vessel_danger(
            env_id,
            torch.tensor(filtered_hits, device=self.device, dtype=torch.float32),
            current_pose[env_id]
            )                     # shape: (16,)
            self.danger_bins[env_id].copy_(danger_row)
            #print(f"Danger Bins: {self.danger_bins}")
        return self.danger_bins


        #     if filtered_hits.shape[0] == 0:
        #         omni.log.warn(f"[env {env_id}] No valid hits inside cylinder.")
        #         sparse_points_all.append(None)
        #         continue

        #     # Prepare point cloud
        #     self.pcd.points = o3d.utility.Vector3dVector(filtered_hits)

        #     if len(self.pcd.points) < 64:
        #         points_np = np.asarray(self.pcd.points)
        #         num_missing = 64 - len(points_np)
        #         idxs = np.random.choice(len(points_np), num_missing, replace=True)
        #         noise = np.random.normal(loc=0.0, scale=1e-4, size=(num_missing, 3))
        #         padded = np.concatenate([points_np, points_np[idxs] + noise], axis=0)
        #         sparse = torch.tensor(padded, dtype=torch.float32, device=self.device)
        #     else:
        #         sparse_pcd = self.pcd.farthest_point_down_sample(64)
        #         sparse = torch.tensor(np.asarray(sparse_pcd.points), dtype=torch.float32, device=self.device)

        #     sparse_points_all.append(sparse)

        # return sparse_points_all 

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
        if self.common_step_counter % 1000 == 0:
            self.draw.clear_points()
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
            sizes = [3.0 for _ in range(b)]
        elif color == "red":
            colors = [color_red for _ in range(b)]
            sizes = [5.0 for _ in range(b)]
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