# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import numpy as np

from isaaclab_tasks.direct.biopsy_skrl.biopsy_env import BiopsyDirectEnvCfg

NUM_POSES = 10
NUM_TRIALS = 5


###
# Observation space as Dict & Action space as Box
###


@configclass
class DictBoxEnvCfg(BiopsyDirectEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        tooltip_pose      Box with shape (7,)  # x, y, z, qw, qx, qy, qz
        raycaster         Point Cloud 
        trial             Discrete with NUM_TRIALS elements
        depth_tumor       Box with shape (1,)
        # History of Previous Poses and their depth values
        ================  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===                  ===
        Idx                  Action
        ===                  ===
        Pose Offset          {X, Y, Z, W, X, Y, Z}
        ===  ===
    """

    # spaces
    observation_space = spaces.Dict({
        "tooltip_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),  # x, y, z
        "tooltip_quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),  # quaternion  qw, qx, qy, qz
        "raycaster": spaces.Box(low=-np.inf, high=np.inf, shape=(64, 3)),  # raw PCD
        #"trial": spaces.Discrete(NUM_TRIALS),
        "depth_tumor": spaces.Box(low=0.0, high=float("inf"), shape=(1,)),
        # History of Previous Poses and their depth values
    })
    # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))

### 
# Observation space as Dict & Action space as Discrete
###

@configclass
class DictDiscreteEnvCfg(BiopsyDirectEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        tooltip_pose      Box with shape (7,)  # x, y, z, qw, qx, qy, qz
        raycaster         Point Cloud 
        trial             Discrete with NUM_TRIALS elements
        depth_tumor       Box with shape (1,)
        # History of Previous Poses and their depth values
        ================  ===

    * Action space (``~gymnasium.spaces.Discrete`` with NUM_POSES elements)

        ===      ===
        N        Action
        ===      ===
        Pick     {P1, P2 .....Pn}
        ===      ===
    """

    # spaces
    observation_space = spaces.Dict({
        "tooltip_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),  # x, y, z
        "tooltip_quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),  # quaternion  qw, qx, qy, qz
        "raycaster": spaces.Box(low=-np.inf, high=np.inf, shape=(64, 3)),  # raw PCD
        #"trial": spaces.Discrete(NUM_TRIALS),
        "depth_tumor": spaces.Box(low=0.0, high=float("inf"), shape=(1,)),
        # History of Previous Poses and their depth values
    })
    # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Discrete(NUM_POSES)  # or for simplicity: {2}


###
# Observation space as Dict & Action space as Dict {Box, Discrete}
###
@configclass
class DictMixedEnvCfg(BiopsyDirectEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        tooltip_pose      Box with shape (7,)  # x, y, z, qw, qx, qy, qz
        raycaster         Point Cloud 
        trial             Discrete with NUM_TRIALS elements
        depth_tumor       Box with shape (1,)
        # History of Previous Poses and their depth values
        ================  ===

    * Action space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ==================  ===
        Key                  Action
        ==================  ===
        Pick               {P1, P2 .....Pn}
        Pose Offset        {X, Y, Z, W, X, Y, Z}
        ==================  ===
    """

    # spaces
    observation_space = spaces.Dict({
        "tooltip_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),  # x, y, z
        "tooltip_quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),  # quaternion  qw, qx, qy, qz
        "raycaster": spaces.Box(low=-np.inf, high=np.inf, shape=(64, 3)),  # raw PCD
        "trial": spaces.Discrete(NUM_TRIALS),
        "depth_tumor": spaces.Box(low=0.0, high=float("inf"), shape=(1,)),
        # History of Previous Poses and their depth values
    })
    # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Dict({
        "offset": spaces.Box(low=-1.0, high=1.0, shape=(6,)),  # or for simplicity: 1 or [1]
        "pick": spaces.Discrete(NUM_POSES),  # or for simplicity: {2}
    })
    # or for simplicity: {"value": 1, "direction": 1}