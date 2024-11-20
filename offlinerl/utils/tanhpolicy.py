import torch
from torch import nn as nn
from torchrl.modules import TanhNormal

from typing import Tuple
import numpy as np
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from typing import Callable, Optional, Tuple


class TanhGaussianPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 num_hidden_layers,
                 hidden_layer_dim,
                 conditioned_sigma: bool = True,
                 log_sig_max=2.0,
                 log_sig_min=-5.0):
        super().__init__()

        self.max_action = max_action
        self.conditioned_sigma = conditioned_sigma
        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        model = [nn.Linear(state_dim, hidden_layer_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            model += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU()]
        self.preprocess = nn.Sequential(*model)

        self.mean = nn.Linear(hidden_layer_dim, action_dim)
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_dim, action_dim)
        else:
            self.sigma = nn.Parameter(torch.zeros(action_dim, 1))

    def forward(self, state: torch.Tensor):
        """
        :param obs: Observation
        """
        logits = self.preprocess(state)

        action = self.mean(logits)

        if self.conditioned_sigma:
            log_std = torch.clamp(self.sigma(logits), min=self.log_sig_min, max=self.log_sig_max)
            std = log_std.exp()
        else:
            shape = [1] * len(action.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(action))
            std = log_std.exp()

        return TanhNormal(action, std)


"""sac, scq, sacn, td3+bc"""


class PolicyNet_Discrete(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_hidden_layers,
                 hidden_layer_dim):
        super().__init__()

        model = [nn.Linear(state_dim, hidden_layer_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            model += [nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU()]
        self.preprocess = nn.Sequential(*model)

        self.mean = nn.Linear(hidden_layer_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """
        :param obs: Observation
        """
        logits = self.preprocess(state)

        probs = nn.functional.softmax(self.mean(logits), dim=1)

        return probs


"""cql"""


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
            self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float = 1.0,
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            orthogonal_init: bool = False,
            no_tanh: bool = True,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, 5, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return nn.functional.softmax(actions, dim=1), log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


"""iql"""


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 512,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))

        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
