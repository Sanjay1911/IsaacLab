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
NUM_BINS = 16  # Number of bins per segment

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
        Twist    {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5}
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
    action_space = spaces.Discrete(NUM_BINS)  # or for simplicity: {2}


###
# Observation space as Dict & Action space as Dict {Box, Discrete}
###

@configclass
class DictMultiDiscreteEnvCfg(BiopsyDirectEnvCfg):
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

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 constituent spaces)

        ==================  ===
        Key                  Action
        ==================  ===
        Insertion           {0.01, 0.02, 0.03}
        Twist               {0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5}
        ==================  ===
    """

    # spaces
    observation_space = spaces.Dict({
        "tooltip_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),  # x, y, z
        "tooltip_quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),  # quaternion  qw, qx, qy, qz
        "raycaster": spaces.Box(low=-np.inf, high=np.inf, shape=(64, 3)),  # raw PCD
        "normalized_depth": spaces.Box(low=0.0, high=1.0, shape=(1,)),  # Normalized depth values for each env
        "deviation": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # Deviation values for each env
        # History of Previous Poses and their depth values
    })
    action_space = spaces.MultiDiscrete([3, 16])
