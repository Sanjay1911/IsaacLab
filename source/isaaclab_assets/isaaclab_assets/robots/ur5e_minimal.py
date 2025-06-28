# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR5e.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/czlocal/sanjay_isaac/forked/minimal_needle.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True
        ), 
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "holder_needle_slider": -0.015, 
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=1.0,
            effort_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
)