# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Biopsy Replanning Environment.
"""

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Biopsy-Direct-v0",
    entry_point=f"{__name__}.biopsy_direct:BiopsyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biopsy_direct:BiopsyDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

###
# Observation space as Dict & Action space as Discrete  
###
gym.register(
    id="Isaac-Biopsy-Direct-Dict-Discrete-v0",
    entry_point=f"{__name__}.biopsy_env:BiopsyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biopsy_env_cfg:DictDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_discrete_ppo_cfg.yaml",
    },
)

###
# Observation space as Dict & Action space as Multi Discrete  
###
gym.register(
    id="Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0",
    entry_point=f"{__name__}.biopsy_env:BiopsyDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biopsy_env_cfg:DictMultiDiscreteEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dict_multi_discrete_ppo_cfg.yaml",
    },
)
