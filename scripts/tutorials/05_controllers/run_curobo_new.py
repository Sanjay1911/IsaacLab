from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import matplotlib.pyplot as plt
import numpy as np
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
    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



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
    
    print("Scene" , scene)
    robot = scene["robot"]
    tensor_args = TensorDeviceType()
    try:
        articulation_controller = robot.get_articulation_controller()
        if articulation_controller is None:
            raise ValueError("Articulation controller is None")
        print("[INFO] Articulation controller is available")
    except Exception as e:
        print(f"[ERROR] Failed to get articulation controller: {e}")

    # Load CuRobo config
    robot_cfg_path = join_path(get_robot_configs_path(), "ur10e.yml")
    robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]

    #Isaac Lab
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
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
        position=torch.tensor([[0.5, 0.5, 0.7]], device="cuda"),
        quaternion=torch.tensor([[0.707, 0, 0.707, 0]], device="cuda"),
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
    joint_tol = 1e-3 
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
                joint_names=robot_cfg["kinematics"]["cspace"]["joint_names"],
            )
            print("Joint names in YAML:", robot_cfg["kinematics"]["cspace"]["joint_names"])
            print("Joint names in sim:", robot.data.joint_names)
            # Plan to target
            print("[INFO] Current joint state:", cu_js.position)
            print("[INFO] Planning to goal pose:", goal_pose)
            result = motion_gen.plan_single(cu_js, goal_pose, plan_config)
            if not result.success.item():
                print("[ERROR] Planning failed:", result.status)
                break

            print("[INFO] Planning succeeded.")
            plan = result.get_interpolated_plan()
            traj_len = plan.position.shape[0]
            step = 0  # reset trajectory step index

        if USE_CUROBO_EXECUTION:
            print("[INFO] Executing cuRobo plan immediately (blocking)...")
            for exec_step in range(traj_len):
                joint_target = plan.position[exec_step].unsqueeze(0)
                robot.set_joint_position_target(joint_target, joint_ids=robot_entity_cfg.joint_ids[:6])
            plan = None         
        elif plan is not None and step < traj_len:
            # ee_error = np.linalg.norm(ee_pos - goal_pos)
            # if ee_error < 1e-2:
            #     print(f"[INFO] EE is close to goal. Stopping joint commands. Error: {ee_error:.4f}")
            #     plan = None  # Stop executing plan
            # else:
            robot.set_joint_position_target(plan.position[step], joint_ids=robot_entity_cfg.joint_ids[:6])
        # ee_pos = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3].cpu().numpy()[0]
        # goal_pos = goal_pose.position.cpu().numpy()[0]

        # ee_current_positions.append(ee_pos)
        # ee_goal_positions.append(goal_pos)
        # time_steps.append(step * sim_dt)

        # df = pd.DataFrame({
        #     "time": time_steps,
        #     "ee_x": [p[0] for p in ee_current_positions],
        #     "ee_y": [p[1] for p in ee_current_positions],
        #     "ee_z": [p[2] for p in ee_current_positions],
        #     "goal_x": [p[0] for p in ee_goal_positions],
        #     "goal_y": [p[1] for p in ee_goal_positions],
        #     "goal_z": [p[2] for p in ee_goal_positions],
        # })
        # df.to_csv("ee_trajectory.csv", index=False)
        # Step sim
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
        print("Joint positions: ", robot.data.joint_pos[0])
        print("EE_Current: ", robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3], robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7])  
        print("EE_Goal: ", goal_pose.position , goal_pose.quaternion)  

        # Now increment step
        if plan is not None and step < traj_len:
            step += 1



if __name__ == "__main__":
    main()
    simulation_app.close()
