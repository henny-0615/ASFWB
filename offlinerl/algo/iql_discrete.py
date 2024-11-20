import copy
import wandb

import torch
import numpy as np
from torch import optim
from typing import Tuple
import torch.nn.functional as F
from torch.distributions import Distribution

from offlinerl.utils.tanhpolicy import MLP, GaussianPolicy


class TwinQ(torch.nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 512, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 512, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class IQLD:
    def __init__(self, state_shape, action_shape, max_action, args):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.args = args

        max_steps = args.max_timesteps

        self.qf = TwinQ(state_shape, action_shape).to(args.device)
        self.vf = ValueFunction(state_shape).to(args.device)
        self.actor = GaussianPolicy(state_shape, action_shape).to(args.device)

        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=3e-4,
                                    weight_decay=args.weight_decay)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4,
                                    weight_decay=args.weight_decay)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate,
                                    weight_decay=args.weight_decay)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=int(max_steps))

        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(args.device)

        self.log_alpha = None
        self.alpha_opt = None
        self.target_entropy = None
        if args.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=args.actor_learning_rate)

        self.num = args.vae_sampling_num
        self.lam = args.critic_penalty_coef

    def sync_weight(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def get_action(self, states):
        states = torch.FloatTensor(states.reshape(1, -1)).to(self.args.device)
        with torch.no_grad():
            dist = self.actor(states).cpu().data.numpy().flatten()
            a_prob = dist.rsample()
            return np.argmax(a_prob)

    def sample_actions(self, obs, requires_grad=False):
        if requires_grad:
            tanh_normal: Distribution = self.actor(obs)
            action = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(action)
        else:
            with torch.no_grad():
                tanh_normal: Distribution = self.actor(obs)
                action = tanh_normal.sample()
                log_prob = tanh_normal.log_prob(action)

        std = tanh_normal.scale
        return action, log_prob, std

    def update_actor(self, obs, actions, adv, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
        policy_out = self.actor(obs)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict

    def update_critic(self, obs, actions, next_obs, rewards, terminals):
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_obs)
        with torch.no_grad():
            target_q = self.q_target(obs, actions)

        v = self.vf(obs)
        adv = target_q - v
        v_loss = torch.mean(torch.abs(self.iql_tau - (adv < 0).float()) * adv ** 2)

        rewards = rewards.squeeze(dim=-1)
        terminals = terminals.squeeze(dim=-1)

        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(obs, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        log_dict["value_loss"] = v_loss.item()
        log_dict["q_loss"] = q_loss.item()

        return adv, log_dict

    def train(self, states, actions, next_states, rewards, dones, step):
        """
        Step1. Update critic
        """
        adv, log_dict = self.update_critic(states, actions, next_states, rewards, dones)

        """
        Step2. Update actor
        """
        log_dict = self.update_actor(states, actions, adv, log_dict)

        if step % self.args.log_interval == 0:
            logging_data = {**log_dict}
            logging_data['total_step'] = step
            wandb.log(logging_data)

        """
        Step3. Soft Updates target network
        """
        self.sync_weight(self.q_target, self.qf, self.args.soft_target_tau)
        # self.sync_weight(self.critic2_target, self.critic2, self.args.soft_target_tau)
