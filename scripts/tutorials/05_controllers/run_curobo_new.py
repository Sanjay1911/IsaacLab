from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

try:
    # Third Party
    import isaacsim
except ImportError:
    pass
import sys
import torch
import torch.nn.functional as F
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

from curobo.types.robot import JointState
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.geom.sdf.world import CollisionCheckerType
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab_assets import UR5N_CFG, UR5H_CFG, FRANKA_PANDA_CFG, UR10_CFG
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaacsim.core.api.world import World
from helper import add_robot_to_scene, add_extensions
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )
    robot = UR5N_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def quat_diff(q1, q2):
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    # Quaternion dot product (cosine of half the angle)
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    angle = 2 * torch.acos(torch.abs(dot))  # in radians
    return torch.rad2deg(angle)  # convert to degrees


def save_ee_plot(ee_current_positions, ee_goal_positions, time_steps, step):
    ee_current_np = np.array(ee_current_positions)
    ee_goal_np = np.array(ee_goal_positions)
    time_np = np.array(time_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(time_np, ee_current_np[:, 0], label="EE_Current_X")
    plt.plot(time_np, ee_current_np[:, 1], label="EE_Current_Y")
    plt.plot(time_np, ee_current_np[:, 2], label="EE_Current_Z")
    plt.plot(time_np, ee_goal_np[:, 0], '--', label="EE_Goal_X")
    plt.plot(time_np, ee_goal_np[:, 1], '--', label="EE_Goal_Y")
    plt.plot(time_np, ee_goal_np[:, 2], '--', label="EE_Goal_Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("End Effector Position Over Time")
    plt.legend()
    plt.grid()
    print(f"[DEBUG] Saving EE plot at step {step}")
    print(f"Data points: {len(ee_current_positions)}")
    plt.savefig(f"ee_position_plot_step_{step}.png")
    plt.close()

def main():
    # Init sim and scene
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda", use_fabric=False)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 1.0, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = MinimalSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # my_world = World(stage_units_in_meters=1.0)
    # my_world._scene = scene
    robot = scene["robot"]
    robot_prim_path = "/World/envs/env_0/Robot"
    print("Robot type:", type(robot), robot_prim_path)
    try:
        robot_wrapper = Robot(prim_path=robot_prim_path, name="robot")
        robot_wrapper.initialize()
        print("Robot Wrapper: ", robot_wrapper)
    except Exception as e:
        print(f"[ERROR] Failed to create robot wrapper: {e}")
        pass
    tensor_args = TensorDeviceType()

    try:
        articulation_controller = robot_wrapper.get_articulation_controller()
        if articulation_controller is None:
            raise ValueError("Articulation controller is None")
        print("[INFO] Articulation controller is available")
    except Exception as e:
        print(f"[ERROR] Failed to get articulation controller: {e}")

    # Load CuRobo config
    robot_cfg_path = join_path(get_robot_configs_path(), "ur5e_biopsy_needle_lock.yml")
    robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world, position=[0, 0, 0.5])
    #Isaac Lab
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tooltip"])
    robot_entity_cfg.resolve(scene)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg=robot_cfg,
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        trajopt_tsteps=32,
        optimize_dt=True,
        collision_cache={"obb": 30, "mesh": 10},
        use_cuda_graph=False,
        maximum_trajectory_dt=1.0,
    )

    motion_gen = MotionGen(motion_gen_cfg)
    motion_gen.warmup(enable_graph=True,warmup_js_trajopt=False)
    print("[INFO] MotionGen is ready")

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=4,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
        check_start_validity=True
    )

    # Create a target pose (in world frame)
    goal_pose = Pose(
        position=torch.tensor([[0.4, 0.4, 0.7]], device="cuda"),
        quaternion=torch.tensor([[1.0, 0.0, 0, 0]], device="cuda"),
    )

    # Warm-up sim
    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    sim_dt = sim.get_physics_dt()
    step = 0
    plan = None
    traj_len = 0
    reset_interval = 150
    joint_tol = 0.007
    ee_current_positions = []
    ee_goal_positions = []
    time_steps = []

    USE_CUROBO_EXECUTION = False
    while simulation_app.is_running():
        # Reset plan every N steps
        if step % reset_interval == 0:
            try:
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
            except Exception as e:
                print(f"[ERROR] Failed to reset robot joint state: {e}")
            
            pos = torch.tensor(robot.data.joint_pos[0], device="cuda").unsqueeze(0)
            vel = torch.tensor(robot.data.joint_vel[0], device="cuda").unsqueeze(0)
            acc = torch.tensor(robot.data.joint_acc[0], device="cuda").unsqueeze(0)

            cu_js = JointState(
                position=pos[0][:6].reshape(1,6),
                velocity=vel[0][:6].reshape(1,6),
                acceleration=acc[0][:6].reshape(1,6),
                jerk=acc[0][:6].reshape(1,6),
                joint_names=j_names[:6],
            )
            # Plan to target
            print("[INFO] Current joint state:", cu_js.joint_names)
            print("[INFO] Planning to goal pose:", goal_pose)
            result = motion_gen.plan_single(cu_js, goal_pose, plan_config)
            if not result.success.item():
                print("[ERROR] Planning failed:", result.status)
                break

            print("[INFO] Planning succeeded.")
            plan = result.get_interpolated_plan()
            cmd_plan = motion_gen.get_full_js(plan)
            print("[INFO] Command plan:", cmd_plan.joint_names)
            sim_js_names = j_names[:6]
            print("[INFO] Robot joint names:", sim_js_names)
            idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in cmd_plan.joint_names:
                    idx_list.append(robot_wrapper.get_dof_index(x))
                    common_js_names.append(x)
            print("[INFO] Common joint names:", common_js_names)
            cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
            cmd_idx = 0
            traj_len = plan.position.shape[0]
            step = 0  # reset trajectory step index
            #print("[INFO] Plan:", plan.position)

        if USE_CUROBO_EXECUTION:
            if cmd_plan is not None:
                cmd_state = cmd_plan[cmd_idx]
                past_cmd = cmd_state.clone()
                print("=== DEBUG ACTION INPUTS ===")
                print("Position:", type(cmd_state.position), cmd_state.position.device)
                print("Velocity:", type(cmd_state.velocity), cmd_state.velocity.device)
                print("Position NumPy:", type(cmd_state.position.cpu().numpy()))
                print("Velocity NumPy:", type(cmd_state.velocity.cpu().numpy()))
                print("Joint Indices:", idx_list, type(idx_list), [type(i) for i in idx_list])
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    #cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )
                articulation_controller.apply_action(art_action)

                for _ in range(2):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)

                cmd_idx += 1
                if cmd_idx >= len(cmd_plan.position):
                    print("Reached target pose, applying pneumatic...")
                    pneumatic_action = ArticulationAction(np.array([-0.145]), joint_indices=[6])
                    articulation_controller.apply_action(pneumatic_action)
                    for _ in range(5):
                        scene.write_data_to_sim()
                        sim.step()
                        scene.update(sim_dt)
                    cmd_idx = 0
                    cmd_plan = None
                    past_cmd = None

        elif plan is not None and step < traj_len:
            ee_pos = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3].cpu().numpy()[0]
            ee_rot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7].cpu().numpy()[0]
            goal_pos = goal_pose.position.cpu().numpy()[0]
            goal_rot = goal_pose.quaternion.cpu().numpy()[0]
            pos_error = np.linalg.norm(ee_pos - goal_pos)
            q_current = torch.tensor([ee_rot], device='cuda')
            q_goal = torch.tensor([goal_rot], device='cuda')
            angle_error = quat_diff(
                q_current,
                q_goal,
            )
            if pos_error < joint_tol :
                print(f"[INFO] Reached goal position with error: {pos_error:.4f} m")
                if angle_error < 2.0:
                    print(f"[INFO] Reached goal orientation with error below 2 degrees")
                else:
                    print(f"[INFO] Orientation error above 2 degrees")
                plan = None
                step = 0
                continue
            robot.set_joint_position_target(plan.position[step], joint_ids=robot_entity_cfg.joint_ids[:6])
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        ee_pos = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3].cpu().numpy()[0]
        goal_pos = goal_pose.position.cpu().numpy()[0]

        ee_current_positions.append(ee_pos)
        ee_goal_positions.append(goal_pos)
        time_steps.append(step * sim_dt)

        df = pd.DataFrame({
            "time": time_steps,
            "ee_x": [p[0] for p in ee_current_positions],
            "ee_y": [p[1] for p in ee_current_positions],
            "ee_z": [p[2] for p in ee_current_positions],
            "goal_x": [p[0] for p in ee_goal_positions],
            "goal_y": [p[1] for p in ee_goal_positions],
            "goal_z": [p[2] for p in ee_goal_positions],
        })
        df.to_csv("ee_trajectory.csv", index=False)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Update markers
        ee_marker.visualize(
            robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3],
            robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7],
        )
        goal_marker.visualize(
            goal_pose.position,
            goal_pose.quaternion,
        )
        # print("wrapper joint pos: ", robot_wrapper.get_joint_positions())
        # print("Joint positions: ", robot.data.joint_pos[0])
        print("EE_Current: ", robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3], robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7])  
        print("EE_Goal: ", goal_pose.position , goal_pose.quaternion)  

        # Now increment step
        if plan is not None and step < traj_len:
            step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
