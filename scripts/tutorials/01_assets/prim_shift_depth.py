# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse

from isaaclab.app import AppLauncher
# create argparser
parser = argparse.ArgumentParser(description="Tutorial on testing distance based deformation.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.io import dump_pickle, load_pickle
from curobo.util.usd_helper import UsdHelper
import isaacsim.core.utils.stage as stage_utils     
from pxr import UsdGeom, UsdPhysics, Gf
import numpy as np
import torch
import omni
import trimesh
import asyncio
import omni
from omni.kit.async_engine import run_coroutine
import omni.replicator.core as rep
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns, RayCaster
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth
from isaaclab_assets import UR5_CFG
from isaaclab.assets import Articulation
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
# === Globals for animation
precomputed_deformations = {}
entry_point = None
insertion_vector = None
INCLUDE_SHIFT = True  # Set to False to disable brain shift deformation
DATA = False  # Set to False to disable RBF deformation
TEST_SHIFT = False  # Set to True to test a simple shift instead of RBF deformation


def rbf_deformation(tm):
    """Precompute RBF deformations for animation"""
    import trimesh
    from scipy.interpolate import RBFInterpolator

    global precomputed_deformations, entry_point, insertion_vector

    print("[INFO] Starting RBF deformation precomputation...")
    mesh_trimesh = tm
    num_samples = 3000
    points, _ = trimesh.sample.sample_surface(mesh_trimesh, num_samples)
    vertices = mesh_trimesh.vertices.copy()
    center = vertices.mean(axis=0)
    direction = np.array([1.0, 1.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    offset = 0.25 * direction
    craniotomy_center = np.array([center[0], center[1], vertices[:, 2].max()])# + offset

    influence_radius = 0.04
    dists = np.linalg.norm(points - craniotomy_center, axis=1)
    control_points = points[dists < influence_radius]

    print(f"[INFO] Control points before downsampling: {len(control_points)}")

    if len(control_points) == 0:
        raise RuntimeError("No control points found within influence radius. Aborting deformation.")

    if len(control_points) > 1000:
        idx = np.random.choice(len(control_points), 1000, replace=False)
        control_points = control_points[idx]

    max_shift = 0.015
    max_steps = 50
    original_vertices = mesh_trimesh.vertices.copy()
    for step in range(max_steps + 1):
        shift_ratio = step / max_steps
        displacements = np.zeros_like(control_points)

        for i, pt in enumerate(control_points):
            dist = np.linalg.norm(pt - craniotomy_center)
            falloff = np.exp(-dist**2 / (2 * (influence_radius / 2)**2))
            displacements[i, 2] = -max_shift * falloff * shift_ratio

        try:
            control_points = np.asarray(control_points)
            displacements = np.asarray(displacements)
            rbf = RBFInterpolator(control_points, displacements, kernel="thin_plate_spline", smoothing=1e-5)
            deformation = rbf(original_vertices)
            precomputed_deformations[step] = original_vertices + deformation
            disp = np.linalg.norm(deformation, axis=1)
            deformed_vertices = original_vertices + deformation
            displacement = np.linalg.norm(deformed_vertices - original_vertices, axis=1)
            print(f"[DEBUG] Step {step}: max disp = {displacement.max():.6f}, mean = {displacement.mean():.6f}")
            print(f"[DEBUG] Step {step}: max disp = {np.max(disp):.6f}, mean = {np.mean(disp):.6f}")

        except Exception as e:
            print(f"[ERROR] RBFInterpolator failed at step {step}: {e}")
            continue

    if not precomputed_deformations:
        raise RuntimeError("RBF deformation failed — no deformations computed.")

    print(f"[INFO] Precomputed {len(precomputed_deformations)} deformation steps.")
    try:
        deformation_list = [precomputed_deformations[i] for i in range(max_steps + 1)]
        deformation_wrapped = [[[deformation_list]]]
        output_path = "/home/sanjay/thesis_replications/forked/IsaacLab/custom/new_nwworld.pkl"
        dump_pickle(output_path, deformation_wrapped)
        entry_point = craniotomy_center + np.array([0.065, 0.12, 0.01])
        tumor_position = craniotomy_center - np.array([0.0, 0.0, 0.1])
        insertion_vector = tumor_position - entry_point
    except Exception as e:
        print(f"[ERROR] Failed to save precomputed deformations: {e}")
        raise RuntimeError("Failed to save precomputed deformations.")


def rbf_deform():
    try:
        current_stage = stage_utils.get_current_stage()
        prim_path = "/World/Vessel/geometry/mesh"
        raw_prim = current_stage.GetPrimAtPath(prim_path)
        prim = UsdGeom.Mesh(raw_prim)
        points = np.asarray(prim.GetPointsAttr().Get())
        print(f"[INFO] Loaded mesh with {len(points)} vertices.")
        # Apply world transform (rotation + translation)
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        faces = np.array(prim.GetFaceVertexIndicesAttr().Get())
        counts = np.array(prim.GetFaceVertexCountsAttr().Get()) 
        if not np.all(counts == 3):
            raise ValueError("Mesh contains non-triangular faces. You need to triangulate first.")
        faces = faces.reshape((-1, 3))
        tm = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        #tm.export("/home/sanjay/thesis_replications/forked/IsaacLab/custom/vessel_usd.obj")
        print("[INFO] Loaded vessel mesh for RBF deformation.")
        rbf_deformation(tm)
    except Exception as e:
        print("Error loading current stage: ", e)


def draw_points(points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):
    draw.clear_points()
    point_list = [tuple(p) for p in points_np]
    # random_colors = np.random.rand(len(points_np), 3)
    # colors = [(r, g, b, 1.0) for r, g, b in random_colors]
    colors = [color] * len(point_list)
    sizes = [size] * len(point_list)
    draw.draw_points(point_list, colors, sizes)


def define_sensor() -> RayCasterCamera:
    """Defines the ray-cast camera sensor to add to the scene."""
    # Camera base frames
    # In contras to the USD camera, we associate the sensor to the prims at these locations.
    # This means that parent prim of the sensor is the prim at this location.
    # Setup camera sensor
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Robot/needle_tool/tooltip",
        mesh_prim_paths=["/World/Tumor"],
        update_period=0.1,
        max_distance=0.01,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=5.0,
            horizontal_aperture=20.955,
            height=48,
            width=48,
        ),
    )
    # Create camera
    camera = RayCasterCamera(cfg=camera_cfg)

    caster_cfg = RayCasterCfg(
        prim_path="/World/Robot/needle_tool/tooltip",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["/World/Tumor", "/World/Vessel"],
        attach_yaw_only=False,
        max_distance=1,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-60, 60], horizontal_fov_range=[-60, 60], horizontal_res=1.0
        )
    )
    caster = RayCaster(cfg=caster_cfg)
    return camera, caster


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # cfg_mesh = sim_utils.MeshFileCfg(
    #     file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/vessels.obj"
    #     #file_path="/home/sanjay/thesis_replications/forked/IsaacLab/deformed_step_0.obj",
    # )
    # cfg_mesh.func("/World/Vessel", cfg_mesh, translation=(0.0, 0.0, 0.20))

    cfg_mesh = sim_utils.UsdFileCfg(usd_path="/home/sanjay/thesis_replications/forked/Vessels.usd")
    cfg_mesh.func("/World/Vessel", cfg_mesh, translation=(0.0, 0.0, 0.20))

    cfg_mesh = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
    cfg_mesh.func("/World/Cube", cfg_mesh, translation=(0.0, 0.0, 0.25))

    cfg_mesh = sim_utils.MeshFileCfg(
        file_path="/home/sanjay/thesis_replications/curobo/src/curobo/content/assets/scene/tumr.obj"
    )
    cfg_mesh.func("/World/Tumor", cfg_mesh, translation=(0.0185, 0.045, 0.20))

    ur5_cfg = UR5_CFG.copy()
    ur5_cfg.prim_path = "/World/Robot"
    minimal = Articulation(cfg=ur5_cfg)

    if DATA:
        rbf_deform()

    camera, caster = define_sensor()
    scene_entities = {"camera": camera, "caster": caster, "robot": minimal}
    return scene_entities
    

def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Main function."""
    # Create the tool tip (small green sphere)
    #prim_utils.create_sphere(prim_path=tool_prim_path, radius=0.002, color=(0.0, 1.0, 0.0))
    camera: RayCasterCamera = scene_entities["camera"]
    robot = scene_entities["robot"]
    caster = scene_entities["caster"]
    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
    print("[INFO]: Setup complete...")

    max_frames = 50
    max_steps = 50

    stage = stage_utils.get_current_stage()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            print(f"[DEBUG] Found mesh prim: {prim.GetPath()}")
    prim = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Vessel/Vessels/Vessels"))
    points_attr = prim.GetPointsAttr()
    print(f"[INFO] Loaded mesh with {len(points_attr.Get())} vertices.")
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    cube_matrix = np.array(omni.usd.get_world_transform_matrix(cube_prim)).T
    cube_pos = cube_matrix[:3, 3]
    tumor_prim = stage.GetPrimAtPath("/World/Tumor")
    tumor_matrix = np.array(omni.usd.get_world_transform_matrix(tumor_prim)).T
    tumor_position = tumor_matrix[:3, 3]
    distance = np.linalg.norm(cube_pos - tumor_position)
    current_step = 0
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.015, 0.015, 0.015)
    camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera"))
    if INCLUDE_SHIFT:
        brain_shift_data = []
        shift_data = load_pickle("/home/sanjay/thesis_replications/forked/IsaacLab/precomputed_brain_deformations_50_1env.pkl")
        #print(shift_data)
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
        
    async def task():
        await stage_utils.update_stage_async()

    root_state = robot.data.default_root_state.clone()
    print(root_state.device)
    root_state[:, :3] += torch.tensor([0.0, 0.0, 0.320], dtype=torch.float32, device=sim.device) 
    robot.write_root_pose_to_sim(root_state[:, :7])  # Slightly lift the robot
    last_distance = None
    while simulation_app.is_running():
        cube_prim = stage.GetPrimAtPath("/World/Cube")
        cube_matrix = np.array(omni.usd.get_world_transform_matrix(cube_prim)).T
        cube_pos = cube_matrix[:3, 3]
        tumor_prim = stage.GetPrimAtPath("/World/Tumor")
        tumor_matrix = np.array(omni.usd.get_world_transform_matrix(tumor_prim)).T
        tumor_position = tumor_matrix[:3, 3]
        distance = np.linalg.norm(cube_pos - tumor_position)
        print(f"[DEBUG] Cube position: {cube_pos}, Tumor position: {tumor_position}, Distance: {distance:.4f}")
        if last_distance is not None and np.isclose(distance, last_distance, rtol=1e-5):
            # Skip updating mesh
            sim.step()
            continue
        last_distance = distance

        # 4. Check distance bounds
        if distance > 0.10:
            # Out of range → no deformation
            sim.step()
            continue

        depth_ratio = 1.0 - (distance / 0.10)  # 1.0 at 0m → 0.0 at 10cm
        step_f = depth_ratio * (len(brain_shift_data) - 1)
        low = int(np.floor(step_f))
        high = min(int(np.ceil(step_f)), len(brain_shift_data) - 1)
        alpha = step_f - low
        # if current_step <= max_frames:
        #     depth_ratio = current_step / max_frames
        #     step_f = depth_ratio * (len(brain_shift_data) - 1 if INCLUDE_SHIFT else max_steps)
        #     low = int(np.floor(step_f))
        #     high = min(int(np.ceil(step_f)), len(brain_shift_data) - 1 if INCLUDE_SHIFT else max_steps)
        #     alpha = step_f - low
        if not DATA:
            if INCLUDE_SHIFT:
                try:
                    verts_low = brain_shift_data[low]
                    verts_high = brain_shift_data[high]
                    print(f"verts_low shape: {verts_low.shape}")
                    print(f"verts_high shape: {verts_high.shape}")

                except IndexError:
                    print(f"[ERROR] Shift data out of bounds at step {low}/{high}")
                    break
            else:
                verts_low = precomputed_deformations[low]
                verts_high = precomputed_deformations[high]
            original_np = np.asarray(prim.GetPointsAttr().Get())
            if TEST_SHIFT:
                shifted = original_np + np.array([0.01, 0, 0]) 
                new_np = np.asarray(shifted, dtype=np.float32)
                usd_pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in shifted]
                points_attr.Set(usd_pts)
            else:
                interpolated_vertices = (1 - alpha) * verts_low + alpha * verts_high
                new_np = interpolated_vertices.astype(np.float32)
                assert new_np.shape == original_np.shape, f"Shape mismatch: {new_np.shape} vs {original_np.shape}"
                """ 
                ----Needed only if you want to set points in local space when dataset collected from this script----
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(prim)).T
                inv_transform = np.linalg.inv(transform_matrix)
                new_np_local = (new_np - transform_matrix[:3, 3]) @ inv_transform[:3, :3] 
                #usd_pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in new_np_local]"""
                usd_pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in interpolated_vertices]
                points_attr.Set(usd_pts)
                stage_utils.update_stage()
            
            displacement = np.linalg.norm(new_np - original_np, axis=1)
            changed_mask = displacement > 1e-5
            changed_percent = 100.0 * np.sum(changed_mask) / displacement.shape[0]
            print(f"[INFO] Vertices changed: {changed_percent:.2f}%")
            print(f"[INFO] Mean displacement: {np.mean(displacement):.6f}")
            print(f"[INFO] Max displacement:  {np.max(displacement):.6f}")
        else:
            continue
        
        transforms = camera._view.get_transforms()
        pos_w, quat_w = transforms[:, :3], transforms[:, 3:]
        camera_marker.visualize(pos_w, quat_w)
        #print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        distances = camera.data.output["distance_to_camera"]
        H, W = 48, 48
        center_distance = distances[0, H//2, W//2, 0]  # (N, H, W, C) shape
        #print("Euclidean distances (sample):", center_distance)  # center pixel
        valid = ~torch.isinf(distances)
        num_valid = valid.sum().item()
        total_rays = distances.numel()
        #print(f"[DEBUG] Valid rays: {num_valid}/{total_rays}")
        mean_distance = distances[valid].mean()
        #print("Mean distance (valid hits):", mean_distance.item())
        if current_step % 100 == 0:
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * -0.05
            joint_vel += torch.rand_like(joint_vel) * -0.000001
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        hits = caster.data.ray_hits_w  # (N, R, 3) shape
        valid_mask = torch.isfinite(hits).all(dim=-1)  # Shape: (N, R)
        valid_hits = hits[valid_mask]  # Shape: (V, 3), where V is number of valid rays
        #print("Valid ray hit positions:\n", valid_hits)
        #print(f"Number of valid hits: {valid_hits.shape[0]}")
        valid_hits_np = valid_hits.cpu().numpy()
        draw_points(valid_hits_np, color=(1.0, 0.0, 0.0, 1.0), size=4.0)
        if args_cli.save:
            # Extract camera data
            camera_index = 0
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            # Pointcloud in world frame
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )

            # Check methods are valid
            # im_height, im_width = camera.image_shape
            # # -- project points to (u, v, d)
            # reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            # reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            # sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            # torch.testing.assert_close(reproj_depths, sim_depths)
        print(f"[INFO] Current step: {current_step}/{max_frames}")

        sim.step()

        #tip_pos = entry_point + depth_ratio * insertion_vector

        # Optionally move the tool tip if you're rendering it:
        # prim_utils.set_prim_world_pose(tool_prim_path, tip_pos)
        current_step += 1


def main():
    """Main function."""
    # Load kit helper
    sim = sim_utils.SimulationContext()
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    main()