import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln=False) -> None:
        super().__init__()
        self.q_model = self.build_q_network(state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln)

    def build_q_network(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln):
        layers = [nn.Linear(state_dim + action_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
            if use_ln:
                layers += [nn.LayerNorm(hidden_layer_dim)]
        layers += [nn.Linear(hidden_layer_dim, 1)]
        return nn.Sequential(*layers)

    def forward(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q_model(sa)


"""SAC, SCQ, td3+bc"""
class Critic_Discrete(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln=False) -> None:
        super().__init__()
        self.output_dim = action_dim
        self.q_model = self.build_q_network(state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln)

    def build_q_network(self, state_dim, action_dim, num_hidden_layers, hidden_layer_dim, use_ln):
        layers = [nn.Linear(state_dim + action_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(inplace=True)]
            if use_ln:
                layers += [nn.LayerNorm(hidden_layer_dim)]
        layers += [nn.Linear(hidden_layer_dim, self.output_dim)]
        return nn.Sequential(*layers)

    def forward(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q_model(sa)


"""SACN"""
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, action_dim, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action)
        return q_values


"""CQL"""
def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 512),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(512, action_dim))

        self.network = nn.Sequential(*layers)

        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

        input_tensor = torch.cat([observations, actions], dim=1)
        q_values = self.network(input_tensor)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
