import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on Preop Path Planning with simulated brainshift and Intraoperative Vision")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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
# Parameters
insertion_depth = 0.05  # mm per segment
num_bins = 16  # Number of bins per segment
bin_angle_deg = 22.5  # Narrow bin angle
bin_angle_rad = np.deg2rad(bin_angle_deg)
num_steps = 100

@configclass
class SteerableSceneCfg(InteractiveSceneCfg):
    """Configuration for a steerable scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    needle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/needle",
        spawn=sim_utils.ConeCfg(
            radius=0.002,
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
def generate_bin_directions(base_dir, num_bins=8, angle=bin_angle_rad):
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, origins):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = scene["needle"]
    print(type(robot))
    current_gravity_status = robot.root_physx_view.get_disable_gravities()    # https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidBodyView.get_disable_gravities
    print("[INFO]: Current gravity status:", current_gravity_status)  
    if current_gravity_status == 0:
        print("[INFO]: Disabling gravity for the robot.")
        robot.root_physx_view.set_disable_gravities(1, 1)
    #Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Start and goal setup
    start_point = torch.tensor(origins[0], device=sim.device, dtype=torch.float32)
    goal_point = torch.tensor(origins[1], device=sim.device, dtype=torch.float32)
    print("[INFO]: Start Point:", start_point)
    print("[INFO]: Goal Point:", goal_point)
    # Calculate goal direction
    goal_direction = goal_point - start_point
    # Normalize goal direction
    if torch.norm(goal_direction) == 0:
        raise ValueError("Start and goal points cannot be the same.")
    goal_direction = goal_direction / torch.norm(goal_direction)
    goal_point = start_point + goal_direction * insertion_depth * num_steps

    # Generate path and bin vectors
    curved_path = [start_point]
    bin_vectors_per_step = []

    current_pos = start_point.clone()
    heading = goal_direction.clone()

    for _ in range(num_steps):
        bins = generate_bin_directions(heading, num_bins, bin_angle_rad)
        bin_vectors_per_step.append((current_pos.clone(), bins))
        best_bin = min(bins, key=lambda b: torch.linalg.norm(current_pos + insertion_depth * b - goal_point))
        current_pos = current_pos + insertion_depth * best_bin
        heading = best_bin
        curved_path.append(current_pos.clone())

    print("[INFO]: Curved Path:", curved_path)
    curved_path = torch.stack(curved_path)
    straight_path = [start_point + goal_direction * insertion_depth * i for i in range(num_steps + 1)]
    straight_path = torch.stack(straight_path)
    if not args_cli.headless:
        draw_points(curved_path.cpu().numpy(), color=(0.2, 0.8, 0.2, 1.0), size=4.0)
        draw_lines(
            start_point.cpu().numpy(),
            goal_point.cpu().numpy(),
            color="green"
        )
    
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

    path_idx = 0
    count = 0

    while simulation_app.is_running():
        if count % 50 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            path_pos = curved_path[path_idx]
            root_state[:, :3] = path_pos

            # --- Compute orientation so needle tip faces forward ---
            if path_idx < len(curved_path) - 1:
                next_pos = curved_path[path_idx + 1]
            else:
                next_pos = curved_path[path_idx]

            direction = next_pos - path_pos
            direction = direction / torch.norm(direction)

            # Assuming needle tip points in +Z
            needle_forward = torch.tensor([0.0, 0.0, 1.0], device=direction.device)
            rotation = rotation_between(needle_forward, direction)
            root_state[:, 3:7] = rotation  # quaternion (w, x, y, z)

            print(f"[INFO]: Pos = {path_pos.tolist()}, Rotation = {rotation.tolist()}")

            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()

            path_idx = (path_idx + 1) % len(curved_path)

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
    scene_cfg = SteerableSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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











