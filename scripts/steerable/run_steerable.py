import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on Preop Path Planning with simulated brainshift and Intraoperative Vision")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--type", type=str, default="greedy", help="Type of the scene to spawn.")
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
if not args_cli.headless:
    import omni.log
    import omni.physics.tensors.impl.api as physx
    import omni.replicator.core as rep
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
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
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import Usd, UsdGeom, Gf
# torch.manual_seed(42)  # for reproducibility
# np.random.seed(42)  # for reproducibility
# Parameters
insertion_depth = 0.05  # mm per segment
num_bins = 16  # Number of bins per segment
bin_angle_deg = 22.5  # Narrow bin angle
bin_angle_rad = np.deg2rad(bin_angle_deg)
num_steps = 10

U1 = torch.tensor(0.01, device=args_cli.device, dtype=torch.float32)  # mm/s
U2 = torch.tensor(0.1, device=args_cli.device, dtype=torch.float32)  # mm/s
REB = torch.tensor(0.001, device=args_cli.device, dtype=torch.float32)  # mm
PHI_Deg = torch.tensor(30, device=args_cli.device, dtype=torch.float32)  # degrees
PHI_Rad = torch.deg2rad(PHI_Deg)  # radians
CURV = torch.tensor(0.2, device=args_cli.device, dtype=torch.float32)  # mm
dt = torch.tensor(0.5, device=args_cli.device, dtype=torch.float32)  # seconds


@configclass
class SteerableSceneCfg(InteractiveSceneCfg):
    """Configuration for a steerable scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
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
            file_path="/home/sanjay/thesis_replications/curobo_thesis_fork/src/curobo/content/assets/scene/tumor.obj",
            scale=(1, 1, 1)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20), rot=(0.70710, 0.70710, 0.0, 0.0)),
    )

    # articulation
    needle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        spawn=sim_utils.CylinderCfg(
            radius=0.002,
            height=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    #raycaster 
    raycast_camera_vessel = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        mesh_prim_paths=["{ENV_REGEX_NS}/Vessel"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.20), rot=(0, 0.0, 0.0, 1.0) ,convention="world"),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=False,
        max_distance=0.2,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=420,
            width=640,
        ),
    )

    raycast_tumor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.20), rot=(0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Tumor"],
        attach_yaw_only=True,
        max_distance=0.2,
        debug_vis=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=50, vertical_fov_range=[-60, 60], horizontal_fov_range=[-20, 20], horizontal_res=1.0
        )
    )


def draw_points(points_np, color=(0.2, 0.8, 0.2, 1.0), size=4.0):
    #draw.clear_points()
    point_list = [tuple(p) for p in points_np]
    colors = [color] * len(point_list)
    sizes = [size] * len(point_list)
    draw.draw_points(point_list, colors, sizes)


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


# Function to generate bin directions
def generate_bin_directions(base_dir, num_bins=16, angle=bin_angle_rad):
    directions = []
    test_tensor = torch.tensor([0, 0, 1], device=base_dir.device, dtype=base_dir.dtype)
    angle_tensor = torch.tensor(angle, device=base_dir.device, dtype=base_dir.dtype)
    for i in range(num_bins):
        theta = 2 * torch.pi * i / torch.tensor(num_bins)
        if torch.allclose(base_dir, test_tensor):
            ortho1 = torch.tensor([1, 0, 0])
        else:
            ortho1 = torch.cross(base_dir, test_tensor)
            ortho1 /= torch.norm(ortho1)
        ortho2 = torch.cross(base_dir, ortho1)
        dir_vector = (
            torch.cos(angle_tensor) * base_dir +
            torch.sin(angle_tensor) * (torch.cos(theta) * ortho1 + torch.sin(theta) * ortho2)
        )
        directions.append(dir_vector / torch.norm(dir_vector))
    return directions


def skew(v):
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], device=v.device, dtype=v.dtype)


def twist_to_matrix(v, w):
    mat = torch.zeros((4, 4), device=v.device, dtype=v.dtype)
    mat[:3, :3] = skew(w)
    mat[:3, 3] = v
    return mat


def generate_needle_path(start, goal, u1=0.02, u2=0.0, phi_deg=30.0, r=0.2, reb=0.001, dt=1.0, num_steps=100):
    """Generates a needle path based on twist kinematics using SE(3)."""
    phi = torch.deg2rad(torch.tensor(phi_deg, device=start.device, dtype=start.dtype))

    poses = [start.clone()]
    g_current = start.clone()

    for _ in range(num_steps):
        tip_pos = g_current[:3, 3]
        to_goal = goal - tip_pos
        to_goal = to_goal / torch.linalg.norm(to_goal)
        current_dir = g_current[:3, 2]
        angle_diff = torch.acos(torch.clamp(torch.dot(current_dir, to_goal), -1, 1))
        direction_changed = angle_diff > torch.deg2rad(torch.tensor(10.0, device=start.device))

        v = torch.tensor([0, -u1 * torch.sin(phi), u1 * torch.cos(phi)], device=start.device, dtype=start.dtype)
        w = torch.tensor([u1 / r, 0, u2], device=start.device, dtype=start.dtype)
        xi_hat = twist_to_matrix(v, w)

        if direction_changed:
            t_mod = reb / u1
            g_partial = g_current @ torch.linalg.matrix_exp(xi_hat * (dt - t_mod))
            z_axis = g_partial[:3, 2]
            trans = torch.eye(4, device=start.device, dtype=start.dtype)
            trans[:3, 3] = z_axis * reb

            theta = torch.atan2(to_goal[1], to_goal[0]) - torch.atan2(z_axis[1], z_axis[0])
            rot_z = torch.eye(4, device=start.device, dtype=start.dtype)
            rot_z[:3, :3] = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
            ], device=start.device, dtype=start.dtype)
            g_current = g_partial @ trans @ rot_z
        else:
            g_current = g_current @ torch.linalg.matrix_exp(xi_hat * dt)

        poses.append(g_current.clone())

    return poses


def rotation_between(vec1, vec2):
    vec1 = vec1 / torch.norm(vec1)
    vec2 = vec2 / torch.norm(vec2)
    dot = torch.dot(vec1, vec2)

    if dot > 0.9999:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=vec1.device)
    elif dot < -0.9999:
        orthogonal = torch.tensor([1.0, 0.0, 0.0], device=vec1.device)
        if torch.allclose(vec1, orthogonal, atol=1e-3):
            orthogonal = torch.tensor([0.0, 1.0, 0.0], device=vec1.device)
        axis = torch.cross(vec1, orthogonal)
        axis = axis / torch.norm(axis)
        return torch.cat((torch.tensor([0.0], device=vec1.device), axis))
    else:
        axis = torch.cross(vec1, vec2)
        axis = axis / torch.norm(axis)
        angle = torch.acos(dot)
        s = torch.sin(angle / 2)
        w = torch.cos(angle / 2)
        xyz = axis * s
        return torch.cat((w.unsqueeze(0), xyz))
        

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, origins):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    num_envs = scene.num_envs
    scene_origins = scene.env_origins
    robot = scene["needle"]
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.005, 0.005, 0.005)
    camera_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/camera"))
    raycaster = scene["raycast_camera_vessel"]
    raycaster.update(dt=sim.get_physics_dt(), force_recompute=True)
    raycast_sensor = scene["raycast_tumor"]
    print(raycast_sensor)
    # Print camera info
    print(raycaster)
    print("Received shape of depth image: ", raycaster.data.output["distance_to_image_plane"].shape)
    print("-------------------------------")
    current_gravity_status = robot.root_physx_view.get_disable_gravities()   # https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidBodyView.get_disable_gravities
    print("[INFO]: Current gravity status:", current_gravity_status, current_gravity_status[0])  
    if current_gravity_status[0] == 0:
        print("[INFO]: Disabling gravity for the robot.")
        robot.root_physx_view.set_disable_gravities(1, num_envs)
    origins = [torch.tensor(o, device=sim.device, dtype=torch.float32) for o in origins]
    # TODO: Add perturbance to start and goal points
    start_points = torch.stack([origins[0] + scene_origins[i] for i in range(num_envs)])
    print(f"Start Points after perturbance: {start_points}")
    goal_points = torch.stack([origins[1] + scene_origins[i] for i in range(num_envs)])
    print(f"Goal Points after perturbance: {goal_points}")
    curved_paths = []

    #Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    origins = [torch.tensor(o, device=sim.device, dtype=torch.float32) for o in origins]

    for env_id in range(num_envs):
        # Calculate goal direction
        goal_direction = goal_points[env_id] - start_points[env_id]
        goal_direction = goal_direction / torch.norm(goal_direction)
        target = start_points[env_id] + goal_direction * insertion_depth * num_steps

        # Generate path and bin vectors
        path = [start_points[env_id]]
        current_pos = start_points[env_id].clone()
        heading = goal_direction.clone()

        if args_cli.type == "greedy":
            for _ in range(num_steps):
                bins = generate_bin_directions(heading, num_bins, bin_angle_rad)
                best_bin = min(bins, key=lambda b: torch.linalg.norm(current_pos + insertion_depth * b - target))
                current_pos += insertion_depth * best_bin
                heading = best_bin
                path.append(current_pos.clone())
        elif args_cli.type == "steerable":
            start_pose = torch.eye(4, device=sim.device, dtype=torch.float32)
            start_pose[:3, 3] = start_points[env_id]
            poses = generate_needle_path(
                start=start_pose,
                goal=goal_points[env_id],
                num_steps=num_steps*2
            )
            curved_path = torch.stack([pose[:3, 3] for pose in poses])  # shape (11, 3)
            curved_paths.append(curved_path)

            # P_i = A + (B-A) * i / N where i in [0, N]
            straight_path = torch.stack([
                start_points[env_id] + (goal_points[env_id] - start_points[env_id]) * i / 100
                for i in range(100 + 1)
            ])  # shape (101, 3)

            # Compute lateral distances and draw lines
            if not args_cli.headless:
                for i in range(len(curved_path)):
                    curved_pt = curved_path[i]
                    # Compute distances to all straight points
                    dists = torch.norm(straight_path - curved_pt[None, :], dim=1)
                    min_idx = torch.argmin(dists)
                    nearest_straight_pt = straight_path[min_idx]
                    draw_points(straight_path.cpu().numpy(), color=(0.8, 0.2, 0.2, 1.0), size=2.0)
                    draw_lines(curved_pt, nearest_straight_pt, color="yellow")
                    lateral_distance = dists[min_idx].item()
                    print(f"[env {env_id} | step {i}] Lateral Distance = {lateral_distance:.6f}")
            
            # Compute normalized distance from each curved point to the goal point
            distance_to_goal = torch.norm(curved_path - goal_points[env_id], dim=1)
            print(f"[env {env_id}] Distance to goal: {distance_to_goal}")

        else:
            raise ValueError(f"Unknown type: {args_cli.type}. Supported types are 'greedy' and 'steerable'.")
        curved_paths.append(torch.stack(path))

    path_idx = torch.zeros(num_envs, dtype=torch.int32, device=sim.device)

    if not args_cli.headless:
        for i in range(num_envs):
            draw_points(curved_paths[i].cpu().numpy(), color=(0.2, 0.8, 0.2, 1.0), size=4.0)
            #draw_lines(start_points[i].cpu(), goal_points[i].cpu(), color="green")

    count = 0
    while simulation_app.is_running():
        distances = raycaster.data.output["distance_to_camera"]
        actual_distances = raycaster.data.output.get("distance_to_camera", None)
        if actual_distances is not None:
            print(f"[INFO] Actual raycast distances: {actual_distances}")
        if distances is None or distances.shape[0] == 0:
            print("[WARN] Raycast distances not yet populated.")
        else:
            print(f"[INFO] Raycast distances shape: {distances.shape}")

        hits = raycast_sensor.data.ray_hits_w
        # for env_id in range(num_envs):
        #     hits_env = hits[env_id]  # shape (R, 3)
        #     valid_mask = torch.isfinite(hits_env).all(dim=-1)  # shape (R,)
        #     valid_hits = hits_env[valid_mask]  # shape (V, 3)
        #     if valid_hits.shape[0] > 0:
        #         # Draw the valid hits
        #         draw_points(valid_hits.cpu().numpy(), color=(1.0, 0.0, 0.0, 1.0), size=4.0)
        #         print(f"[INFO] Valid raycast hits for environment {env_id}: {valid_hits.shape[0]}")
        #     else:
        #         print(f"[WARN] No valid raycast hits for environment {env_id}.")
        if count % 50 == 0:
            # lateral drift from the curved path to straight path
            root_state = robot.data.default_root_state.clone()
            for i in range(num_envs):
                path = curved_paths[i]
                idx = path_idx[i].item()
                pos = path[idx]
                next_pos = path[idx + 1] if idx < len(path) - 1 else path[idx]
                direction = next_pos - pos
                direction = direction / torch.norm(direction)
                root_state[i, :3] = pos
                needle_forward = torch.tensor([0.0, 0.0, 1.0], device=sim.device, dtype=direction.dtype)
                rotation = rotation_between(needle_forward, direction)
                root_state[i, 3:7] = rotation
                path_idx[i] = (idx + 1) % len(path)
                camera_marker.visualize(raycaster.data.pos_w, raycaster.data.quat_w_world)

            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
        
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = SteerableSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    origins = [[0.0, 0.0, 0.0], [0.5, 0.3, 0.5]]
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator1
    run_simulator(sim, scene, origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()











