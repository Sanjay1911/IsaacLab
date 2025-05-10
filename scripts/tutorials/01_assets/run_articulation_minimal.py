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

import isaacsim.core.utils.prims as prim_utils

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

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG, UR5_CFG  # isort:skip

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
    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 2.0], [0.0, 0.0, 0.5])
    scene_cfg = MinimalSceneCfg(num_envs=32, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tooltip"])
    robot_entity_cfg.resolve(scene)
    print("[INFO]: Setup complete...")

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
            print("[INFO]: Root state: ", root_state)
            # Tilt 90° about X-axis => quaternion: [w, x, y, z] = [cos(θ/2), sin(θ/2)*axis]
            q_tilt = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]], device=sim.device)  # 90° around X
            # tilt with some noise
            q_tilt += torch.rand_like(q_tilt) * 0.5
            q_tilt = q_tilt / q_tilt.norm(dim=1, keepdim=True)  # normalize

            root_state[:, :3] = scene.env_origins # set root position
            root_state[:, 3:7] = q_tilt  # set root orientation
            print("[INFO]: Updated Root state: ", root_state)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
 
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.05
            print("[INFO]: Joint pos: ", joint_pos)
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
