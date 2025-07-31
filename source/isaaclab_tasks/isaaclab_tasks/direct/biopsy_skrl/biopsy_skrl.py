import torch
import torch.nn as nn
import torch.nn.functional as F
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model, CategoricalMixin, MultiCategoricalMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space  # https://skrl.readthedocs.io/en/latest/api/utils/spaces.html#skrl.utils.spaces.torch.unflatten_tensorized_space 
from gymnasium import spaces
# seed for reproducibility
set_seed(42)  
DEBUG = False  

class PointNetExtractor(nn.Module):
    def __init__(self, point_channel=3, output_dim=256):
        super(PointNetExtractor, self).__init__()

        print("Initialized PointNetExtractor")

        self.local_mlp = nn.Sequential(
            nn.Linear(point_channel, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, C] where C is usually 3 (xyz)
        """
        x = self.local_mlp(x)               # [B, N, 256]
        x = torch.max(x, dim=1)[0]          # [B, 256]
        return x


class ContinouosActionPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.pointnet = PointNetExtractor(point_channel=3, output_dim=256)  # your observation must be [B, N, 3]

        self.actor = nn.Sequential(
            nn.Linear(264, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(0.5 * torch.ones(self.num_actions))
    
    def compute(self, inputs, role):
        states = inputs["states"]
        #print("States keys:", states, len(states[0]))  # Debugging line to check available keys
        states = unflatten_tensorized_space(self.observation_space, states)  # https://github.com/Toni-SM/skrl/discussions/205
        print("Unflattened states shape:", {k: v.shape for k, v in states.items()})  # Debugging line to check shapes
        print("States keys:", states.keys())  # Debugging line to check available keys
        # Process raycaster point cloud [B, 64, 3]
        pcd = states["raycaster"]
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0)  # Ensure batch dimension exists

        pointnet_features = self.pointnet(pcd)  # -> [B, 256]

        # Other inputs (make sure all are [B, D])
        depth_tumor = states["depth_tumor"]
        if depth_tumor.ndim == 1:
            depth_tumor = depth_tumor.unsqueeze(0)

        tooltip_pos = states["tooltip_position"]
        if tooltip_pos.ndim == 1:
            tooltip_pos = tooltip_pos.unsqueeze(0)

        tooltip_quat = states["tooltip_quaternion"]
        if tooltip_quat.ndim == 1:
            tooltip_quat = tooltip_quat.unsqueeze(0)

        # trial = states["trial"]
        # if trial.ndim == 0:
        #     trial = trial.unsqueeze(0)
        # one_hot_trial = F.one_hot(trial.long(), num_classes=5).float()

        # Concatenate all features
        other_features = torch.cat([depth_tumor, tooltip_pos, tooltip_quat], dim=-1)  # [B, 1 + 7 = 8]
        x = torch.cat([pointnet_features, other_features], dim=-1)  # [B, 256 + 8]
        if DEBUG:
            print("Raycaster shape:", pcd.shape)  # Should be [B, 64, 3]
            print(f"PointNet features: {pointnet_features.shape}")
            print("Other features shapes:",depth_tumor.shape, tooltip_pos.shape, tooltip_quat.shape)  # Debug
            print(f"Other features: {other_features.shape}")
            print(f"Input to actor: {x.shape}")
            print("Concatenated features shape:", x.shape)

        x = self.actor(x)
        # raw_mean = self.mean_layer(x)
        # mean = torch.tanh(raw_mean)  # ensure action is in [-1, 1]
        # print("Policy raw output before tanh:", raw_mean)
        # print("Policy mean (after tanh):", mean)
        # return mean, self.log_std_parameter, {}
        raw_mean = self.mean_layer(x)
        mean = torch.tanh(raw_mean)
        noise = torch.randn_like(raw_mean)
        std = torch.exp(self.log_std_parameter)
        sampled_action = raw_mean + std * noise
        squashed_action = torch.tanh(sampled_action)  # squashing the action to be in [-1, 1] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html , https://www.reddit.com/r/reinforcementlearning/comments/1hwau8q/clipping_vs_squashed_tanh_for_rescaling_actions/- Why should I normalize actions?
        if DEBUG:
            print("Policy raw output before tanh:", raw_mean)
            #print("Policy mean (after tanh):", mean)
            print("Policy sampled action (before squashing):", sampled_action)
            print("Policy squashed action (after tanh):", squashed_action)
        return mean, self.log_std_parameter, {}


# define the model
class DiscreteActionPolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        self.pointnet = PointNetExtractor(point_channel=3, output_dim=256)  # your observation must be [B, N, 3]

        self.actor = nn.Sequential(
            nn.Linear(264, 128),  # 256 from PointNet + 8 from other features
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions)
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        #print("States keys:", states, len(states[0]))  # Debugging line to check available keys
        states = unflatten_tensorized_space(self.observation_space, states)  # https://github.com/Toni-SM/skrl/discussions/205
        #print("Unflattened states shape:", {k: v.shape for k, v in states.items()})  # Debugging line to check shapes
        # Process raycaster point cloud [B, 64, 3]
        pcd = states["raycaster"]
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0)  # Ensure batch dimension exists

        pointnet_features = self.pointnet(pcd)  # -> [B, 256]

        # Other inputs (make sure all are [B, D])
        depth_tumor = states["depth_tumor"]
        if depth_tumor.ndim == 1:
            depth_tumor = depth_tumor.unsqueeze(0)

        tooltip_pos = states["tooltip_position"]
        if tooltip_pos.ndim == 1:
            tooltip_pos = tooltip_pos.unsqueeze(0)

        tooltip_quat = states["tooltip_quaternion"]
        if tooltip_quat.ndim == 1:
            tooltip_quat = tooltip_quat.unsqueeze(0)

        # trial = states["trial"]
        # if trial.ndim == 0:
        #     trial = trial.unsqueeze(0)
        # one_hot_trial = F.one_hot(trial.long(), num_classes=5).float()

        # Concatenate all features
        other_features = torch.cat([depth_tumor, tooltip_pos, tooltip_quat], dim=-1)  # [B, 1 + 7 = 8]
        x = torch.cat([pointnet_features, other_features], dim=-1)  # [B, 256 + 8]
        if DEBUG:
            print("Raycaster shape:", pcd.shape)  # Should be [B, 64, 3]
            print(f"PointNet features: {pointnet_features.shape}")
            print("Other features shapes:",depth_tumor.shape, tooltip_pos.shape, tooltip_quat.shape)  # Debug
            print(f"Other features: {other_features.shape}")
            print(f"Input to actor: {x.shape}")
            print("Concatenated features shape:", x.shape)
        x = self.actor(x)
        return x, {}


class MultiDiscreteActionPolicy(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction)
        self.pointnet = PointNetExtractor(point_channel=3, output_dim=256)  # your observation must be [B, N, 3]
        self.actor = nn.Sequential(
            nn.Linear(262, 128),  # 256 from PointNet + 6 from other features
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions)
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        #print("States keys:", states, len(states[0]))  # Debugging line to check available keys - len(states[0]) = 64*3 + 1 + 1 + 2 + 1 + 1= 198
        states = unflatten_tensorized_space(self.observation_space, states)  # https://github.com/Toni-SM/skrl/discussions/205
        #print("Unflattened states shape:", {k: v.shape for k, v in states.items()})  # Debugging line to check shapes
        #print("States keys:", states.keys())
        # Process raycaster point cloud [B, 64, 3]
        pcd = states["raycaster"]
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0)  # Ensure batch dimension exists

        pointnet_features = self.pointnet(pcd)  # -> [B, 256]

        # Other inputs (make sure all are [B, D])
        normalized_depth_t = states["normalized_depth_t"]
        if normalized_depth_t.ndim == 1:
            normalized_depth_t = normalized_depth_t.unsqueeze(0)

        deviation_t = states["deviation_t"]
        if deviation_t.ndim == 1:
            deviation_t = deviation_t.unsqueeze(0)

        normalized_depth_t_ndt = states["normalized_depth_t_ndt"]
        if normalized_depth_t_ndt.ndim == 1:
            normalized_depth_t_ndt = normalized_depth_t_ndt.unsqueeze(0)
        
        deviation_t_ndt = states["deviation_t_ndt"]
        if deviation_t_ndt.ndim == 1:
            deviation_t_ndt = deviation_t_ndt.unsqueeze(0)

        current_action = states["current_action"]
        if current_action.ndim == 1:
            current_action = current_action.unsqueeze(0)

        # trial = states["trial"]
        # if trial.ndim == 0:
        #     trial = trial.unsqueeze(0)
        # one_hot_trial = F.one_hot(trial.long(), num_classes=5).float()

        # Concatenate all features
        other_features = torch.cat([current_action, normalized_depth_t, normalized_depth_t_ndt, deviation_t, deviation_t_ndt], dim=-1)  # [B, 1 + 7 = 8]
        x = torch.cat([pointnet_features, other_features], dim=-1)  # [B, 256 + 8]
        if DEBUG:
            print("Raycaster shape:", pcd.shape)  # Should be [B, 64, 3]
            print(f"PointNet features: {pointnet_features.shape}")
            print("Other features shapes:", normalized_depth_t.shape, normalized_depth_t_ndt.shape, deviation_t.shape, deviation_t_ndt.shape, current_action.shape, pcd.shape)  # Debug
            print(f"Other features: {other_features.shape}")
            print(f"Input to actor: {x.shape}")
            print("Concatenated features shape:", x.shape)
        x = self.actor(x)
        return x, {}


class ValueModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(198, 256),  # has to be 198 because of the Dict observation space (match len(states) in compute method)
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# load and wrap the Isaac Lab environment 
env = load_isaaclab_env(task_name="Isaac-Biopsy-Direct-Dict-MultiDiscrete-v0")  # change here for different action space
env = wrap_env(env)

device = env.device
if DEBUG:
    if isinstance(env.action_space, spaces.Discrete):
        print("Discrete action space detected, using CategoricalMixin")
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        print("Continuous action space detected, using MultiCategoricalMixin")
    else:
        raise NotImplementedError("Unknown action space type - only Discrete and MultiDiscrete are supported for this Task")

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
#CategoricalMixin is used for discrete action spaces, GaussianMixin for continuous action spaces
models = {}

if isinstance(env.action_space, spaces.Discrete):
    print("Using CategoricalMixin for discrete action space")
    models["policy"] = DiscreteActionPolicy(env.observation_space, env.action_space, device)
elif isinstance(env.action_space, spaces.MultiDiscrete):
    print("Using MultiCategoricalMixin for MultiDiscrete action space")
    models["policy"] = MultiDiscreteActionPolicy(env.observation_space, env.action_space, device)

models["value"] = ValueModel(env.observation_space, env.action_space, device)  # separate value model
if DEBUG:
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  # 16 * 2048 / 4096
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 80
cfg["experiment"]["checkpoint_interval"] = 800
cfg["experiment"]["directory"] = "runs/torch/Isaac-Biopsy-Direct-Dict-Discrete-v0"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 16000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()