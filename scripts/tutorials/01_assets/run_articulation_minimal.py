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
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
from pxr import UsdGeom, UsdPhysics
import omni.log
import omni.physics.tensors.impl.api as physx
import trimesh
from typing import List, Tuple
import isaacsim.core.utils.prims as prim_utils
from isaacsim.util.debug_draw import _debug_draw
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab_assets import UR5N_CFG, UR5H_CFG, FRANKA_PANDA_CFG, UR10_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG, UR5_CFG  # isort:skip
draw = _debug_draw.acquire_debug_draw_interface()
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

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.0, 1.0, 1.0)
        ),
    )

    skull = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Skull",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/skull.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.50), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    vessel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Vessel",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/vessels.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.50), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    tumor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tumor",
        spawn=sim_utils.MeshFileCfg(
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/tumor.obj"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.50), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )
    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def get_entry_points(points_world, tumor_center, safety_offset=0.0):
    if isinstance(points_world, torch.Tensor):
        points_world = points_world.cpu().numpy()
    if isinstance(tumor_center, torch.Tensor):
        tumor_center = tumor_center.cpu().numpy()
    radius = 0.03   # 3 cm disk
    height_min = 0.00
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


def get_trimesh_mesh(flag="skull"):
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
            return tm , tm.centroid
        except Exception as e:
            print("[ERROR] Failed to convert to mesh prim due to trimesh mesh")
            pass


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
        print("[INFO] Mesh Prim: ", prim, prim.GetTypeName())
        print("[INFO] Mesh Prim Children: ", prim.GetAllChildren())
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


def draw_points(points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):
    point_list = [tuple(p) for p in points_np]
    colors = [color] * len(point_list)
    sizes = [size] * len(point_list)
    draw.draw_points(point_list, colors, sizes)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 2.0], [0.0, 0.0, 0.5])
    scene_cfg = MinimalSceneCfg(num_envs=2, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    skull = scene["skull"]
    vessel = scene["vessel"]
    tumor = scene["tumor"]

    skull_trimesh, _ = get_trimesh_mesh("skull")
    _, tumor_centroid = get_trimesh_mesh("tumor")
    skull_trimesh_sampled = sample_even_fit_mesh(mesh=skull_trimesh, n_spheres=5000, sphere_radius=0.005)
    try:
        entry_pts = get_entry_points(skull_trimesh_sampled[0], tumor_centroid)
        draw_points(entry_pts, color=(1.0, 0.0, 0.0, 1.0), size=4.0)
    except Exception as e:
        print("[ERROR] Failed to get entry points: ", e)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tooltip"])
    robot_entity_cfg.resolve(scene)
    #print("[INFO]: Setup complete...")

    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 150 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            # Tilt 90° about X-axis => quaternion: [w, x, y, z] = [cos(θ/2), sin(θ/2)*axis]
            q_tilt = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]], device=sim.device)  # 90° around X
            # tilt with some noise
            q_tilt += torch.rand_like(q_tilt) * 0.5
            q_tilt = q_tilt / q_tilt.norm(dim=1, keepdim=True)  # normalize

            root_state[:, :3] = scene.env_origins # set root position
            root_state[:, 3:7] = q_tilt  # set root orientation
            #print("[INFO]: Updated Root state: ", root_state)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
 
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.05
            #print("[INFO]: Joint pos: ", joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            robot.reset()
            print("[INFO]: Resetting robot state...")
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


if __name__ == "__main__":

    main()
    simulation_app.close()
