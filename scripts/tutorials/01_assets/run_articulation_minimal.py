# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on Preop Path Planning with simulated brainshift and Intraoperative Vision")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os, yaml
import torch
torch.set_printoptions(profile="full")
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from pxr import UsdGeom, UsdPhysics, Gf
import omni.log
import omni.physics.tensors.impl.api as physx
import omni.replicator.core as rep
import trimesh
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import rotate
import open3d as o3d
import math, datetime
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
import pprint
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.cloner import GridCloner
from isaacsim.util.debug_draw import _debug_draw
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
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils.math import project_points, unproject_depth
from isaaclab.utils import convert_dict_to_backend
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_mul
draw = _debug_draw.acquire_debug_draw_interface()
DIST_THRESHOLD = 0.005  
SCORE_THRESHOLD = 300
CAMERA_SAVE = True
SIM = True
MODE = "NONE"  # "RAYCAST" or None
ENV_SPACING = 0.5
INCLUDE_SHIFT = True  # Whether to include brain shift in the simulation
all_lines = []
all_collisions = []
all_entry_points = []
all_tumor_centers = []


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
            file_path="/home/sanjay/thesis_replications/curobo/src/curobo/content/assets/scene/tumr.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    raycast_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0.0, 0.0, 1.0), convention="world"),
        #data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        data_types=["distance_to_image_plane", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.01,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/needle_tool/tooltip",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        attach_yaw_only=True,
        max_distance=0.1,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-180, 180], horizontal_fov_range=[-180, 180], horizontal_res=1.0
        )
        # pattern_cfg=patterns.BpearlPatternCfg(
        #     horizontal_fov=360.0,
        #     horizontal_res=1.0,
        #     vertical_ray_angles=[-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, 0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],  # degrees
        # )
    )



def batch_get_start_poses(entries: torch.Tensor, tumors: torch.Tensor):
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


def batch_get_start_poses_old(entries: torch.Tensor, tumor: torch.Tensor):
    """
    entries: (B, 3) batch of entry points
    tumor: (3,) single tumor centroid
    Returns:
        positions: (B, 3)
        quaternions: (B, 4)
    """
    entries = entries.double()
    tumor = tumor.double()
    #print("entries:", entries)
    directions = tumor.unsqueeze(0) - entries  # (B, 3)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    z_axis = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64).expand_as(directions)

    dot_products = (z_axis * directions).sum(dim=1)  # (B,)
    ones = torch.ones_like(dot_products)
    minus_ones = -ones

    quaternions = torch.empty((entries.shape[0], 4), dtype=torch.float64)

    for i in range(entries.shape[0]):
        dot = dot_products[i]
        if torch.isclose(dot, ones[i]).item():
            quaternions[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        elif torch.isclose(dot, minus_ones[i]).item():
            orth = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            if torch.allclose(z_axis[i], orth):
                orth = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            axis = torch.cross(z_axis[i], orth)
            axis = axis / torch.norm(axis)
            quaternions[i] = torch.cat([
                torch.tensor([0.0], dtype=torch.float64),
                axis
            ])
        else:
            axis = torch.linalg.cross(z_axis[i], directions[i])
            axis = axis / torch.norm(axis)
            angle = torch.acos(dot)
            half_angle = angle / 2.0
            w = torch.cos(half_angle)
            xyz = axis * torch.sin(half_angle)
            quaternions[i] = torch.cat([w.unsqueeze(0), xyz])

    return entries, quaternions


def apply_scene_offsets(positions, num_envs, spacing=ENV_SPACING):
    cloner = GridCloner(spacing=spacing)
    offsets, _ = cloner.get_clone_transforms(num_envs)
    return np.array(positions) + offsets


def make_transform(position, quaternion):
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


def extract_pose_from_transform(T):
    """Convert 4x4 matrix to (position, quaternion)"""
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3].numpy())
    quat = torch.tensor(rot.as_quat())  # (x, y, z, w)
    quat = quat[[3, 0, 1, 2]]  # → (w, x, y, z)
    return pos, quat


def get_entry_points_cone(skull_points, tumor_center, cone_angle_deg=15, offset=0.07):
    """
    Select skull points that lie within a cone pointing to the tumor center.
    """
    tumor_center = torch.tensor(tumor_center, dtype=torch.float32)
    skull_points = torch.tensor(skull_points, dtype=torch.float32)

    vectors = tumor_center - skull_points
    norms = torch.norm(vectors, dim=1, keepdim=True)
    directions = vectors / norms

    reference = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)  # along -Y
    cos_angle = torch.clamp(torch.sum(directions * reference, dim=1), -1.0, 1.0)

    angle_rad = torch.acos(cos_angle)
    mask = angle_rad < math.radians(cone_angle_deg)

    filtered_points = skull_points[mask]
    adjusted_points = filtered_points + offset * reference
    return adjusted_points.cpu().numpy()


def get_entry_points(points_world, tumor_center, safety_offset=0.05):
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


def collision_scoring(entry_point, tumor_center, vessel_points, num_samples=1000):
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

    #print(f"Collision Score: {collision_score}/{num_samples} ({collision_ratio:.2%})")
    return collision_score, collision_ratio

def get_trimesh_mesh(flag="skull"):
    meshes = []
    if flag == "skull":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Skull")
    elif flag == "vessel":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Vessel")
    elif flag == "tumor":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Tumor")
    for prim in mesh_prim:
        # print("[INFO] Mesh Prim: ", prim, prim.GetTypeName())
        # print("[INFO] Mesh Prim Children: ", prim.GetAllChildren())
        mesh_prim = sim_utils.get_first_matching_child_prim(
            str(prim.GetPath()), lambda prim: prim.GetTypeName() == "Mesh"
        )

        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim at path: {prim.GetPath()}")

        try:
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            # Apply world transform (rotation + translation)
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = np.matmul(points, transform_matrix[:3, :3].T)
            points += transform_matrix[:3, 3]
            faces = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())
            counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get()) 
            if not np.all(counts == 3):
                raise ValueError("Mesh contains non-triangular faces. You need to triangulate first.")
            faces = faces.reshape((-1, 3))
            tm = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            meshes.append((tm , tm.centroid, tm.vertices, mesh_prim))
        except Exception as e:
            print("[ERROR] Failed to convert to mesh prim due to trimesh mesh :", e)
            continue
    if len(meshes) == 0:
        raise RuntimeError(f"Invalid mesh prim at path: {prim.GetPath()}")
    
    return meshes


def sample_even_fit_mesh(mesh: trimesh.Trimesh, n_spheres: int, sphere_radius: float) -> Tuple[np.array, List[float]]:
    n_pts = trimesh.sample.sample_surface_even(mesh, n_spheres)[0]
    n_radius = [sphere_radius for _ in range(len(n_pts))]
    return n_pts, n_radius


def sample_points_usd(flag="skull"):
    if flag == "skull":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Skull")
    elif flag == "vessel":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Vessel")
    elif flag == "tumor":
        mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Tumor")
    for prim in mesh_prim:
        # print("[INFO] Mesh Prim: ", prim, prim.GetTypeName())
        # print("[INFO] Mesh Prim Children: ", prim.GetAllChildren())
        mesh_prim = sim_utils.get_first_matching_child_prim(
            str(prim.GetPath()), lambda prim: prim.GetTypeName() == "Mesh"
        )

        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim at path: {prim.GetPath()}")

        try:
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            faces = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get())
            counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get()) 
            if not np.all(counts == 3):
                raise ValueError("Mesh contains non-triangular faces. You need to triangulate first.")
            faces = faces.reshape((-1, 3))
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = np.matmul(points, transform_matrix[:3, :3].T)
            points += transform_matrix[:3, 3]
            samples = min(5000, points.shape[0])
            points = points[np.random.choice(points.shape[0], samples, replace=False)]
            return points
            # indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            # wp_mesh = convert_to_warp_mesh(points, indices, device=sim.device)
            # if wp_mesh is not None: 
            #     print("[INFO] Converted to warp mesh")
        except Exception as e:
            print("[ERROR] Failed to convert to UsdGeom mesh prim due to : ", e)
            pass

def deform_vertices_rbf(vertices, center, influence_radius, step, total_steps=60, max_shift=0.2):
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
    

def draw_lines(start, end, color):
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
        #print(start_pose)
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
    draw.draw_lines(
        start_pose.tolist(),
        end_pose.tolist(),
        colors,
        sizes
    )

def draw_points(points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):
    #draw.clear_points()
    point_list = [tuple(p) for p in points_np]
    colors = [color] * len(point_list)
    sizes = [size] * len(point_list)
    draw.draw_points(point_list, colors, sizes)


def visualise_new_paths(vessel_points):
    vis_geometries = []
    
    vessel_pcd = o3d.geometry.PointCloud()
    vessel_pcd.points = o3d.utility.Vector3dVector(vessel_points)
    vessel_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis_geometries.append(vessel_pcd)

    for line, collisions in zip(all_lines, all_collisions):
        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(line)
        colors = np.array([[1, 0, 0] if c else [0, 1, 0] for c in collisions])
        path_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis_geometries.append(path_pcd)

    for entry, tumor in zip(all_entry_points, all_tumor_centers):
        for point, color in zip([entry, tumor], [[0, 0, 1], [1, 1, 0]]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            vis_geometries.append(sphere)

    # Show everything together
    o3d.visualization.draw_geometries(vis_geometries)


def save_point_cloud_ply(filename, points: np.ndarray):
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


def filter_hits_in_cylinder(hits_np, center, axis, radius=0.01, height=0.05):
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


def main():
    """
    env_data: Dict[int, Dict[str, Any]]
    ├── int (env_id): Index of each simulated environment (e.g., 0, 1, 2, ...)
    │
    └── Dict[str, Any]: Per-environment data, including:
        ├── "skull_points": np.ndarray of shape (N, 3)
        │     → Sampled surface points from the skull mesh (in world coordinates)
        │
        ├── "tumor_centroid": np.ndarray of shape (3,)
        │     → Centroid of the tumor mesh (in world coordinates)
        │
        ├── "vessel_points": np.ndarray of shape (M, 3)
        │     → Sampled surface points from the vessel mesh (in world coordinates)
        │
        ├── "entry_points": List[np.ndarray] or np.ndarray of shape (K, 3)
        │   → Computed entry points projected toward the tumor center
        │
        ├──"entry_poses": List[np.ndarray] or np.ndarray of shape (K, 3)
        │   → Computed entry pose for the robot end effector
    """
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, use_fabric=True)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 2.0], [0.0, 0.0, 0.5])
    scene_cfg = MinimalSceneCfg(num_envs=1, env_spacing=ENV_SPACING)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    mesh_prim = sim_utils.find_matching_prims(prim_path_regex="/World/envs/env_.*/Tumor")
    for prim in mesh_prim:
        print("[INFO] Mesh Prim: ", prim, prim.GetTypeName())
        print("[INFO] Mesh Prim Children: ", prim.GetAllChildren())
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.015, 0.015, 0.015)
    needle_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/needle"))
    camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera"))
    num_envs = scene.num_envs
    robot = scene["robot"]
    print("[INFO] Before loading path Robot: ", robot.body_names)
    needle_index = robot.data.body_names.index("tooltip")
    needle_pose_w = robot.data.body_state_w[:, needle_index, 0:7]
    needle_pos = needle_pose_w[:, :3]
    needle_quat = needle_pose_w[:, 3:7]
    print("Needle position:", needle_pos)
    print("Needle orientation (quat):", needle_quat)
    camera = scene["raycast_camera"]
    tumor = scene["tumor"]
    print("[LOG] Tumor Type: ", type(tumor))
    env_data = {}
    print(camera._view)
  # returns torch.Tensor of positions
    if MODE == "RAYCAST":
        skull_meshes = get_trimesh_mesh("skull")
        for i, (skull_mesh, _, _, _) in enumerate(skull_meshes):
            skull_points, _ = sample_even_fit_mesh(skull_mesh, n_spheres=20000, sphere_radius=0.005)
            env_data[i] = {"skull_points": skull_points}

        vessel_meshes = get_trimesh_mesh("vessel")
        for i, (vessel_mesh, _, vessel_vertices, vessel_prim) in enumerate(vessel_meshes):
            vessel_points, _ = sample_even_fit_mesh(vessel_mesh, n_spheres=20000, sphere_radius=0.005)
            if i not in env_data:
                env_data[i] = {}
            env_data[i]["vessel_points"] = vessel_points

        tumor_meshes = get_trimesh_mesh("tumor")
        for i, (tumor_mesh, tumor_centroid, tumor_vertices, tumor_prim) in enumerate(tumor_meshes):
            tumor_points, _ = sample_even_fit_mesh(tumor_mesh, n_spheres=20000, sphere_radius=0.005)
            if i not in env_data:
                env_data[i] = {}
            env_data[i]["tumor_centroid"] = tumor_centroid
            env_data[i]["tumor_points"] = tumor_points

        for i, data in env_data.items():
            try:
                entry_pts = get_entry_points(data["skull_points"], data["tumor_centroid"])
                data["entry_points"] = entry_pts
                #print(f"[ENV {i}] Entry points found: {len(entry_pts)}")
            except Exception as e:
                print(f"[ERROR] ENV {i}: Failed to compute entry points: {e}")
                data["entry_points"] = []

        for i, data in env_data.items():
            if "entry_points" in data and len(data["entry_points"]) > 0:
                draw_points(data["entry_points"], color=(1.0, 0.0, 0.0, 1.0), size=4.0)
            if "tumor_centroid" in data and len(data["tumor_centroid"]) > 0:
                draw_points([data["tumor_centroid"]], color=(0.0, 0.0, 1.0, 1.0), size=8.0)

        for i, data in env_data.items():
            if "entry_points" in data and len(data["entry_points"]) > 0:
                entry_pts = data["entry_points"]
                tumor_center = data["tumor_centroid"]
                vessel_points = data["vessel_points"]
                scored_paths = []
                for entry_pt in entry_pts:
                    collision_score, collision_ratio = collision_scoring(entry_pt, tumor_center, vessel_points)
                    scored_paths.append({"entry_point": entry_pt,
                                        "score": collision_score,
                                        "ratio": collision_ratio})

                top_paths = sorted(scored_paths, key=lambda x: x["score"])[:10]
                # env_data[i]["top_entry_points"] = top_paths
                data["top_entry_points"] = top_paths

        for i, data in env_data.items():
            if "top_entry_points" in data and len(data["top_entry_points"]) > 0:
                for entry_pt in data["top_entry_points"]:
                    draw_points([entry_pt["entry_point"]], color=(0.0, 1.0, 0.0, 1.0), size=8.0)
                    draw_points([data["tumor_centroid"]], color=(1.0, 1.0, 0.0, 1.0), size=8.0)
                    p1 = np.array(entry_pt["entry_point"]) 
                    p2 = np.array(data["tumor_centroid"])
                    draw_lines(p1, p2, color="green")
                    print(f"[ENV {i}] Drawing line from {p1} to {p2}")

        entry_points_np = np.array([data["top_entry_points"][0]["entry_point"] for data in env_data.values()])
        tumor_centroids_np = np.array([data["tumor_centroid"] for data in env_data.values()])
        entry_points = torch.tensor(entry_points_np, dtype=torch.float32)
        tumor_centroids = torch.tensor(tumor_centroids_np, dtype=torch.float32)
        print("[LOG]: Entry points: ", entry_points)
        print("[LOG]: Tumor centroids: ", tumor_centroids)

        positions, quaternions = batch_get_start_poses(entry_points, tumor_centroids)
        positions = positions.to(device=sim.device)
        quaternions = quaternions.to(device=sim.device)
        print("[LOG]: POSE : ", positions, quaternions)
    
    else:
        tumor_positions = []
        tumor_quaternions = []
        tumor_centroids = []
        entry_points = []
        top_entry_points = []    
        start_positions = []
        start_quaternions = []
        # Read Tumor Dataset Pickle and get entry points and start poses
        data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/tumor_dataset_100_cleaned.pkl")    
        for i in range(min(num_envs, len(data))):
            print(f"[INFO] Loading data for env {i}")
            try:
                env_data = data[i]
                for key in ["tumor_position", "tumor_quat", "tumor_centroid", "entry_points", "top_entry_points", "start_pose"]:
                    assert key in env_data, f"[ERROR] Missing key '{key}' in entry {i}"
                tumor_positions.append(env_data["tumor_position"])
                tumor_quaternions.append(env_data["tumor_quat"])
                tumor_centroids.append(env_data["tumor_centroid"])
                entry_points.append(env_data["entry_points"])
                top_entry_points.append(env_data["top_entry_points"])
                start_positions.append(env_data["start_pose"]["position"])
                start_quaternions.append(env_data["start_pose"]["quaternion"])
            except KeyError as e:
                print(f"[ERROR] ENV {i}: Missing key {e}")
                continue
            except AssertionError as e:
                print(f"[ERROR] ENV {i}: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] ENV {i}: Failed to load data: {e}")
                continue
        
        print("Entry points at env 0:", entry_points[0])
        print("Length of start positions:", len(start_positions), num_envs)
        cloner = GridCloner(spacing=ENV_SPACING)
        offsets, _ = cloner.get_clone_transforms(num_envs)
        
        for env_id, points in enumerate(entry_points):
            #print(f"[INFO] Entry points for env {env_id}: {points}")
            try:
                # Apply offset to each point in current env
                offset = offsets[env_id]  # shape (3,)
                offset_points = points + offset  # (N, 3) + (3,) → (N, 3)
                #for point in offset_points:
                    #draw_points([point], color=(1.0, 0.0, 0.0, 1.0), size=4.0)
                    #print(f"[INFO] Drawing offset entry point {point} for env {env_id}")
            except Exception as e:
                print(f"[ERROR] Failed to draw entry points for env {env_id}: {e}")
                continue

        try:
            for env_id, points in enumerate(tumor_centroids):
                offset = offsets[env_id]  # shape (3,)
                offset_points = points + offset  # (N, 3) + (3,) → (N, 3)
                draw_points([offset_points], color=(0.0, 0.0, 1.0, 1.0), size=4.0)        
        except Exception as e:
            print(f"[ERROR] Failed to draw tumor centroids: {e}")
            pass

        try:
            for env_id, top_points in enumerate(top_entry_points):
                for i in range(len(top_points)):
                    offset = offsets[env_id]  # shape (3,)
                    offset_points = top_points[i]["entry_point"] + offset  # (N, 3) + (3,) → (N, 3)
                    draw_points([offset_points], color=(0.0, 1.0, 0.0, 1.0), size=4.0)
        except Exception as e:
            print(f"[ERROR] Failed to draw top entry points because: {e}")

        try:
            for env_id, top_points in enumerate(top_entry_points):
                tumor_center = tumor_centroids[env_id]
                offset = offsets[env_id]  # shape (3,)

                for i in range(len(top_points)):
                    entry_point = top_points[i]["entry_point"]
                    p1 = entry_point + offset
                    p2 = tumor_center + offset

                    draw_lines(p1, p2, color="green")
                    #print(f"[INFO] Drawing line from {p1} to {p2} in env {env_id}")
        except Exception as e:
            print(f"[ERROR] Failed to draw lines: {e}")

        try:
            start_positions = apply_scene_offsets(start_positions, num_envs)
            #print("New Start positions:", start_positions)
        except Exception as e:
            print(f"[ERROR] ENV {i}: Failed to apply scene offsets: {e}")

        try:
            print("-------------------")
            tumor_positions_tensor = torch.tensor(np.array(tumor_positions), dtype=torch.float32)
            tumor_quaternions_tensor = torch.tensor(np.array(tumor_quaternions), dtype=torch.float32)
            print("Tumor positions:", tumor_positions_tensor.shape)
            assert tumor_positions_tensor.shape[0] == num_envs
            assert tumor_quaternions_tensor.shape[0] == num_envs
            poses_pos, poses_quat = tumor.get_local_poses()
            #print("Tumor Pose Positions Shape:", poses_pos.shape)
            #print("Tumor Pose Quaternions Shape:", poses_quat.shape)
            #print("Tumor Position Tensor:", poses_pos)
            tumor.set_local_poses(tumor_positions_tensor, tumor_quaternions_tensor)
            print(tumor.get_local_poses())
            print("-------------------")
        except Exception as e:
            print(f"[ERROR] ENV {i}: Failed to set tumor pose: {e}")
            pass
    
    if INCLUDE_SHIFT:
        brain_shift_data = []
        shift_data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/precomputed_brain_deformations_50_1env.pkl")
        print("[INFO] Loaded brain shift data with length:", len(shift_data))
        for env_id in range(min(1, len(shift_data))):
            try:
                print(f"[INFO] Extracting top-1 shift steps for env {env_id}")
                env = shift_data[env_id]
                top_entry = env[0]  # top-ranked entry out of 10
                for step_id in range(50):
                    step = top_entry[0][0][step_id]
                    print(f"[DEBUG] Step {step_id} shape: {step.shape}")
                    brain_shift_data.append(step)
            except Exception as e:
                print(f"[ERROR] Failed to extract for env {env_id}: {e}")
        if len(brain_shift_data) < 2:
            raise ValueError("Not enough shift steps in brain_shift_data.")

    holder_pos_trch = torch.zeros((num_envs, 3), dtype=torch.float64, device=sim.device)
    holder_quat_trch = torch.zeros((num_envs, 4), dtype=torch.float64, device=sim.device)
    #visualise_new_paths(env_data[0]["vessel_points"])
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tooltip"])
    robot_entity_cfg.resolve(scene)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera_lab", date)
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
    shift_step = 0
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        camera.update(sim_dt)
        needle_index = robot.data.body_names.index("tooltip")
        needle_pose_w = robot.data.body_state_w[:, needle_index, 0:7]
        needle_pos = needle_pose_w[:, :3]
        needle_quat = needle_pose_w[:, 3:7]
        #needle_marker.visualize(needle_pos, needle_quat)
        # print("Needle position:", needle_pos)
        # print("Needle orientation (quat):", needle_quat)
        # print("camera pose: ", camera.data.pos_w, camera.data.quat_w_world)
        transforms = camera._view.get_transforms()
        pos_w, quat_w = transforms[:, :3], transforms[:, 3:]
        #camera_marker.visualize(pos_w, quat_w)
        # print("Camera parent position:", pos_w)
        # print("Camera parent orientation (quat):", quat_w)
        # print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        distances = camera.data.output["distance_to_camera"]
        H, W = 420, 640
        center_distance = distances[0, H//2, W//2, 0]  # (N, H, W, C) shape
        print("Euclidean distances (sample):", center_distance)  # center pixel
        valid = ~torch.isinf(distances)
        num_valid = valid.sum().item()
        total_rays = distances.numel()
        print(f"[DEBUG] Valid rays: {num_valid}/{total_rays}")
        mean_distance = distances[valid].mean()
        print("Mean distance (valid hits):", mean_distance.item())
        #print(scene["raycast_sensor"])
        hits = scene["raycast_sensor"].data.ray_hits_w  # (N, R, 3) shape
        valid_mask = torch.isfinite(hits).all(dim=-1)  # Shape: (N, R)
        valid_hits = hits[valid_mask]  # Shape: (V, 3), where V is number of valid rays
        #print("Valid ray hit positions:\n", valid_hits)
        print(f"Number of valid hits: {valid_hits.shape[0]}")
        valid_hits_np = valid_hits.cpu().numpy()
        #draw_points(valid_hits_np, color=(1.0, 0.0, 0.0, 1.0), size=4.0)
        # Tooltip position and insertion direction
        needle_center = needle_pos[0].cpu().numpy()  # (3,) position of tooltip
        insertion_axis = np.array([0.0, 0.0, 1.0])  # Replace with true orientation if neededd

        # Filter hits within a cylinder around the needle
        filtered_hits = filter_hits_in_cylinder(
            hits_np=valid_hits_np,
            center=needle_center,
            axis=insertion_axis,
            radius=0.01,
            height=0.05
        )
        # Visualize filtered hits
        draw_points(filtered_hits, color=(0.0, 1.0, 1.0, 1.0), size=4.0)
        print(f"[INFO] Hits inside cylinder: {filtered_hits.shape[0]}/{valid_hits_np.shape[0]}")

        if valid_hits_np.shape[0] != 0:
            print("Valid hits shape:", valid_hits_np.shape)
            save_point_cloud_ply("/home/sanjay/thesis_replications/forked/IsaacLab/custom/raycast_hits_maxdist.ply", valid_hits_np)
        with open(os.path.join(output_dir, "depth_info.csv"), "a") as f:
            f.write(f"{count},{mean_distance.item()},{center_distance.item()},{num_valid/total_rays},{100.0 * num_valid / total_rays:.2f}%\n")
        #print("-------------------------------")
        if CAMERA_SAVE and count % 25 == 0:
            camera_index = 0
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            single_cam_info = camera.data.info[camera_index]

            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                data = np.rot90(data, k=1)
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}

            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            # Unproject to camera frame
            depth = camera.data.output["distance_to_image_plane"]
            points_3d_cam = unproject_depth(depth, camera.data.intrinsic_matrices)  # (N, H, W, 3)

            # env_id = 0
            # points_cam = points_3d_cam[env_id].reshape(-1, 3)  # (H*W, 3)

            # # Apply world transform
            # points_rotated = quat_apply(camera.data.quat_w_world[env_id].unsqueeze(0), points_cam)  # (H*W, 3)
            # points_world = points_rotated + camera.data.pos_w[env_id]  # (H*W, 3)

            # # Filter valid depth
            # depth_valid = depth[env_id].squeeze(-1).reshape(-1) > 1e-4
            # points_world = points_world[depth_valid]

            # dists = torch.norm(points_world - needle_pos, dim=1)

            # mask = dists < 0.05  # distance threshold in meters
            # nearby_points = points_world[mask]

            # if nearby_points.shape[0] > 100:
            #     idx = torch.randperm(nearby_points.shape[0])[:100]
            #     nearby_points = nearby_points[idx]

            # draw_points(nearby_points.tolist(), color=(1.0, 0.0, 0.0, 1.0), size=4.0)

        if INCLUDE_SHIFT and count % 20 == 0:
            print(f"[INFO] Count: {count}, Shift Step: {shift_step}")
            if shift_step < 10:
                print(f"[INFO] Applying shift step {shift_step}")
                deformed_vertices = brain_shift_data[shift_step]
                print(f"[INFO] Deformed vertices shape: {len(deformed_vertices)}")
                if len(deformed_vertices) == 0:
                    print(f"[ERROR] No deformed vertices available for shift step {shift_step}")
                    shift_step += 1
                    continue
                for i in range(num_envs):
                    try:
                        stage = stage_utils.get_current_stage()
                        raw_prim = stage.GetPrimAtPath(f"/World/envs_{i}/Vessel")
                        prim = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/envs_{i}/Vessel"))
                        print(f"[INFO] Found prim at /env_{i}/Vessel: {type(prim),type(raw_prim)}")
                        if not prim or not prim.GetPrim().IsA(UsdGeom.Mesh):
                            print(f"[ERROR] Prim at env_{i}/Vessel is not a valid UsdGeom.Mesh!")
                            continue
                        print(f"[INFO] Found prim: {prim}")
                    except Exception as e:
                        print(f"[ERROR] Failed to get stage or prim: {e}")
                        continue
                    
                    # Get reference to the points attribute of your mesh
                    # Example: points_attr = UsdGeom.Mesh(mesh_paths[i]).GetPointsAttr()
                    points_attr = prim.GetPointsAttr()
                    if not points_attr.IsDefined():
                        print(f"[ERROR] points attribute is not defined on /env_{i}/Vessel")
                        continue

                    # Convert to Gf.Vec3f and assign
                    usd_points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in deformed_vertices]
                    points_attr.Set(usd_points)
                    points_np = np.array([[v[0], v[1], v[2]] for v in deformed_vertices], dtype=np.float32)
                    #draw.clear_points()
                    draw_points(points_np, color=(1.0, 1.0, 1.0, 1.0), size=26.0)
                print(f"[INFO] Deformed vertices applied for shift step {shift_step} in all environments.")
                shift_step += 1
            else:
                print("[INFO] No more shift steps available.")
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        if count % 150 == 0:
            # reset counter
            count = 0
            tooltip_offset_pos = torch.tensor([0.02464, -0.00005, -0.0265], dtype=torch.float64)
            tooltip_offset_quat = torch.tensor(
                R.from_euler("xyz", [0, 0, 1.5707]).as_quat()
            ) 

            # Build transform holder → tooltip, then invert
            T_holder_to_tooltip = make_transform(tooltip_offset_pos, tooltip_offset_quat)
            T_tooltip_to_holder = torch.linalg.inv(T_holder_to_tooltip)
            
            for i in range(num_envs):
                tooltip_pos = start_positions[i]
                tooltip_quat = start_quaternions[i]
                tooltip_pos_world = tooltip_pos
                T_world_tooltip = make_transform(tooltip_pos_world, tooltip_quat)

                # Compute holder base pose
                T_world_holder = T_world_tooltip @ T_tooltip_to_holder
                holder_pos, holder_quat = extract_pose_from_transform(T_world_holder)
                holder_pos_trch[i] = holder_pos
                holder_quat_trch[i] = holder_quat

            root_state = robot.data.default_root_state.clone()
            #print("[INFO]: Default root state: ", root_state) 
            holder_pos_trch.to(device=sim.device)
            holder_quat_trch.to(device=sim.device)
            root_state[:, :3] = holder_pos_trch
            root_state[:, 3:7] = holder_quat_trch
            # set really slow velocity
            root_state[:, 7:] = torch.zeros_like(root_state[:, 7:])
            #print("[INFO]: Root state: ", root_state[0])
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            #print("[INFO]: Updated Root state: ", root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * -1.0
            joint_vel = 0.00000
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            camera.update(sim_dt)   
            
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


if __name__ == "__main__":

    main()
    simulation_app.close()
