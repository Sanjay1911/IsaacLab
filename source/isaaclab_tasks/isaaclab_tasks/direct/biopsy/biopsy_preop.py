import os, yaml
import torch
import numpy as np
from pxr import UsdGeom, UsdPhysics, Gf
import omni.log
import omni.physics.tensors.impl.api as physx
#import omni.replicator.core as rep
import trimesh
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
import open3d as o3d
import math, datetime
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
import pprint
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.cloner import GridCloner
#from isaacsim.util.debug_draw import _debug_draw
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab_assets import UR5_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.io import dump_pickle, load_pickle
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils.math import project_points, unproject_depth
from isaaclab.utils import convert_dict_to_backend
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_mul
#draw = _debug_draw.acquire_debug_draw_interface()
DIST_THRESHOLD = 0.005  
SCORE_THRESHOLD = 300
CAMERA_SAVE = False
SIM = True
MODE = "NONE"  # "RAYCAST" or None
ENV_SPACING = 0.5
all_lines = []
all_collisions = []
all_entry_points = []
all_tumor_centers = []


class BiopsyPreop:
    def __init__(self):
        print("Biopsy Preop initialized")
    
    def print_test(self):
        print("Test function called")
        
    def batch_get_start_poses(self, entries: torch.Tensor, tumors: torch.Tensor):
        """
        entries: (B, 3) batch of entry points
        tumors: (B, 3) batch of tumor centroids
        Returns:
            positions: (B, 3)
            quaternions: (B, 4)
        """
        entries = entries.double()
        tumors = tumors.double()
        
        directions = tumors - entries  # (B, 3)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        # Default "forward" z-axis
        z_axis = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64, device=entries.device).expand_as(directions)

        dot_products = (z_axis * directions).sum(dim=1)  # (B,)
        ones = torch.ones_like(dot_products)
        minus_ones = -ones

        quaternions = torch.empty((entries.shape[0], 4), dtype=torch.float64, device=entries.device)

        for i in range(entries.shape[0]):
            dot = dot_products[i]
            if torch.isclose(dot, ones[i]).item():
                quaternions[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=entries.device)
            elif torch.isclose(dot, minus_ones[i]).item():
                orth = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=entries.device)
                if torch.allclose(z_axis[i], orth):
                    orth = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=entries.device)
                axis = torch.cross(z_axis[i], orth)
                axis = axis / torch.norm(axis)
                quaternions[i] = torch.cat([torch.tensor([0.0], dtype=torch.float64, device=entries.device), axis])
            else:
                axis = torch.linalg.cross(z_axis[i], directions[i])
                axis = axis / torch.norm(axis)
                angle = torch.acos(dot)
                half_angle = angle / 2.0
                w = torch.cos(half_angle)
                xyz = axis * torch.sin(half_angle)
                quaternions[i] = torch.cat([w.unsqueeze(0), xyz])

        return entries, quaternions

    def deform_vertices_rbf(self, vertices, center, influence_radius, step, total_steps=60, max_shift=0.2):
        shift_ratio = step / total_steps
        shift_magnitude = shift_ratio * max_shift
        distances = np.linalg.norm(vertices - center, axis=1)
        control_ids = np.where(distances < influence_radius)[0]
        control_points = vertices[control_ids]
        if len(control_points) == 0:
            min_dist = np.min(np.linalg.norm(vertices - center, axis=1))
            print(f"Min distance from craniotomy center to mesh: {min_dist} but influence radius is {influence_radius}")    
            print("No control points within influence radius.")
            return vertices
        
        displacements = np.zeros_like(control_points)

        if len(control_points) > 1000:
            idx = np.random.choice(len(control_points), 1000, replace=False)
            control_points = control_points[idx]
            displacements = displacements[idx]

        for i, p in enumerate(control_points):
            dist = np.linalg.norm(p - center)
            falloff = np.exp(-dist**2 / (2 * (influence_radius / 2)**2))
            displacements[i, 2] -= shift_magnitude * falloff

        # RBF interpolation (include anchor points if needed)
        rbf = RBFInterpolator(control_points, displacements, kernel="thin_plate_spline", smoothing=1e-5)
        deformed = vertices + rbf(vertices)

        return deformed
    
    def get_entry_points(self, points_world, tumor_center, safety_offset=0.05):
        if isinstance(points_world, torch.Tensor):
            points_world = points_world.cpu().numpy()
        if isinstance(tumor_center, torch.Tensor):
            tumor_center = tumor_center.cpu().numpy()
        radius = 0.03   # 3 cm disk
        height_min = 0.05
        height_max = 0.750
        projection_dir = np.array([0, -1, 0])  

        entry_points = []
        for pt in points_world:
            vec = pt - tumor_center
            height = np.dot(vec, projection_dir)
            lateral = np.linalg.norm(vec - height * projection_dir)

            if height_min <= height <= height_max and lateral <= radius:
                # Apply safety offset in the projection direction
                pt_offset = pt + safety_offset * projection_dir
                entry_points.append(pt_offset)

        entry_points = np.array(entry_points)
        return entry_points

    def collision_scoring(self, entry_point, tumor_center, vessel_points, num_samples=1000):
        vessels_kd = cKDTree(vessel_points)

        # Sample points along straight line path
        line = np.linspace(0, 1, num_samples).reshape(-1, 1) * (tumor_center - entry_point) + entry_point

        # Compute distances to nearest vessel point
        dists, _ = vessels_kd.query(line)
        collisions = dists < DIST_THRESHOLD

        collision_score = np.sum(collisions)
        collision_ratio = collision_score / num_samples

        all_lines.append(line)
        all_collisions.append(collisions)
        all_entry_points.append(entry_point)
        all_tumor_centers.append(tumor_center)
        # print("Sample vessel point:", vessel_points[0])
        # print("Entry point:", entry_point)
        # print("Tumor center:", tumor_center)
        # print("Distance range:", np.min(dists), np.max(dists))

        # print(f"Collision Score: {collision_score}/{num_samples} ({collision_ratio:.2%})")
        return collision_score, collision_ratio

    def apply_scene_offsets(self, positions, num_envs, spacing=ENV_SPACING):
        cloner = GridCloner(spacing=spacing)
        offsets, _ = cloner.get_clone_transforms(num_envs)
        return np.array(positions) + offsets

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

    def extract_pose_from_transform(self, T):
        """Convert 4x4 matrix to (position, quaternion)"""
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3].numpy())
        quat = torch.tensor(rot.as_quat())  # (x, y, z, w)
        quat = quat[[3, 0, 1, 2]]  # → (w, x, y, z)
        return pos, quat