import copy
import wandb

import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.distributions import Distribution

from offlinerl.utils.critic import FullyConnectedQFunction
from offlinerl.utils.tanhpolicy import TanhGaussianPolicy, Scalar


class CQLD:
    def __init__(self, state_shape, action_shape, max_action, args):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.cql_n_actions = 10
        self.cql_importance_sample = True
        self.cql_temp = 1.0
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_lagrange = False
        self.cql_alpha = 10.0
        self.args = args

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=3e-4,
        )
        self.cql_target_action_gap = -1.0

        max_steps = args.max_timesteps

        self.actor = TanhGaussianPolicy(state_shape, action_shape, orthogonal_init=True
                                        ).to(args.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate,
                                    weight_decay=args.weight_decay)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=int(max_steps))

        self.critic1 = FullyConnectedQFunction(state_shape, action_shape, True, 2).to(args.device)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_learning_rate,
                                    weight_decay=args.weight_decay)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = FullyConnectedQFunction(state_shape, action_shape, True, 2).to(args.device)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_learning_rate,
                                    weight_decay=args.weight_decay)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_clip_grad_norm = args.actor_clip_grad_norm
        self.critic_clip_grad_norm = args.critic_clip_grad_norm

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
            a_prob, _ = self.actor(states)
            a_prob = a_prob.cpu().data.numpy().flatten()
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

    def update_actor(self, obs, actions):
        # sampled_actions, log_pi, _ = self.sample_actions(obs, requires_grad=True)
        sampled_actions, _ = self.actor(obs)
        log_pi = torch.log(sampled_actions + 1e-8)

        # update alpha
        if self.args.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = 0
            alpha = 0.01

        q_actions = torch.min(
            self.critic1(obs, sampled_actions),
            self.critic2(obs, sampled_actions),
        )
        q_actions = q_actions.gather(1, (torch.max(sampled_actions, dim=1))[-1].to(torch.int64).unsqueeze(-1))
        log_pi = (torch.max(log_pi, dim=1))[0]
        q_actions = q_actions.squeeze(-1)
        assert q_actions.shape == (self.args.batch_size,), q_actions.shape
        assert log_pi.shape == (self.args.batch_size,), log_pi.shape

        if self.args.actor_penalty_coef:
            bc_loss = (((torch.max(sampled_actions, dim=1))[-1] - actions.squeeze(-1)) ** 2).sum(-1)
            policy_loss = (alpha * log_pi - q_actions + self.args.actor_penalty_coef * bc_loss).mean()
        else:
            with torch.no_grad():
                bc_loss = (((torch.max(sampled_actions, dim=1))[-1] - actions) ** 2).sum(-1)
            policy_loss = (alpha * log_pi - q_actions).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad_norm or float("inf"))
        policy_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.actor.parameters()]))
        self.actor_opt.step()

        if self.args.use_actor_scheduler:
            self.actor_scheduler.step()

        actor_update_data = {"train/actor q": q_actions.mean(),
                             "train/actor loss": policy_loss.mean(),
                             "train/bc loss": bc_loss.sum(-1).mean(),
                             "train/actor gradient": policy_grad_norm,
                             "train/alpha": alpha}

        return actor_update_data

    def update_critic(self, obs, actions, next_obs, rewards, terminals):
        q1_pred = self.critic1(obs, actions.expand(-1, 5))
        q2_pred = self.critic2(obs, actions.expand(-1, 5))
        q1_pred = q1_pred.gather(1, actions.to(torch.int64))
        q2_pred = q2_pred.gather(1, actions.to(torch.int64))
        assert q1_pred.shape == (self.args.batch_size, 1), q1_pred.shape
        assert q2_pred.shape == (self.args.batch_size, 1), q2_pred.shape

        # next_sampled_actions, _, _ = self.sample_actions(next_obs)
        with torch.no_grad():
            next_sampled_actions, _ = self.actor(next_obs)

        target_q_values = torch.min(
            self.critic1_target(next_obs, next_sampled_actions),
            self.critic2_target(next_obs, next_sampled_actions),
        )
        target_q_values = target_q_values.gather(1,
                                                 (torch.max(next_sampled_actions, dim=1))[-1].to(torch.int64).unsqueeze(
                                                     -1))

        assert target_q_values.shape == (self.args.batch_size, 1), target_q_values.shape
        assert rewards.shape == (self.args.batch_size, 1), rewards.shape

        q_target = rewards + (1. - terminals) * self.args.discount * target_q_values.detach()
        assert q_target.shape == (self.args.batch_size, 1), q_target.shape

        # calculate loss for ood actions
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        # CQL
        batch_size = actions.shape[0]
        action_dim = 5
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, action_dim)
        cql_current_actions, cql_current_log_pis = self.actor(
            obs, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_obs, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic1(obs, cql_random_actions)
        cql_q2_rand = self.critic2(obs, cql_random_actions)
        cql_q1_current_actions = self.critic1(obs, cql_current_actions)
        cql_q2_current_actions = self.critic2(obs, cql_current_actions)
        cql_q1_next_actions = self.critic1(obs, cql_next_actions)
        cql_q2_next_actions = self.critic2(obs, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                q1_pred,
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                q2_pred,
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.repeat_interleave(5, dim=-1).detach(),
                    cql_q1_current_actions - cql_current_log_pis.repeat_interleave(5, dim=-1).detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.repeat_interleave(5, dim=-1).detach(),
                    cql_q2_current_actions - cql_current_log_pis.repeat_interleave(5, dim=-1).detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_pred.squeeze(1),
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_pred.squeeze(1),
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = obs.new_tensor(0.0)
            alpha_prime = obs.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        qf_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic1.parameters(), self.critic_clip_grad_norm or float("inf"))
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic2.parameters(), self.critic_clip_grad_norm or float("inf"))
        critic1_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.critic1.parameters()]))
        critic2_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.critic2.parameters()]))
        self.critic1_opt.step()
        self.critic2_opt.step()

        critic_update_data = {
            "train/q1": q1_pred.mean(),
            "train/q2": q2_pred.mean(),
            "train/q target": q_target.mean(),
            "train/max q target": q_target.max(),
            "train/q critic loss": qf_loss,
            "train/critic1 gradient": critic1_grad_norm,
            "train/critic2 gradient": critic2_grad_norm
        }

        return critic_update_data

    def train(self, states, actions, next_states, rewards, dones, step):
        """
        Step1. Update critic
        """
        critic_update_data = self.update_critic(states, actions, next_states, rewards, dones)

        """
        Step2. Update actor
        """
        actor_update_data = self.update_actor(states, actions)

        if step % self.args.log_interval == 0:
            logging_data = {**critic_update_data, **actor_update_data}
            logging_data['total_step'] = step
            wandb.log(logging_data)

        """
        Step3. Soft Updates target network
        """
        self.sync_weight(self.critic1_target, self.critic1, self.args.soft_target_tau)
        self.sync_weight(self.critic2_target, self.critic2, self.args.soft_target_tau)
