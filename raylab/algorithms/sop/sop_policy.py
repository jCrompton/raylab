"""SOP policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.utils.exploration import ParameterNoise
import raylab.utils.pytorch as ptu
import raylab.policy as raypi


class SOPTorchPolicy(raypi.TargetNetworksMixin, raypi.TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for SOP."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.sop.sop import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config.setdefault("critic", {})
        module_config["critic"]["double_q"] = config["clipped_double_q"]
        module_config.setdefault("actor", {})
        module_config["actor"]["perturbed_policy"] = isinstance(
            self.exploration, ParameterNoise
        )
        # pylint:disable=no-member
        return super().make_module(obs_space, action_space, config)

    @override(raypi.TorchPolicy)
    def optimizer(self):
        config = self.config["torch_optimizer"]
        components = "actor critics".split()
        optim_clss = [
            ptu.get_optimizer_class(config[k].pop("type")) for k in components
        ]
        optims = {
            k: cls(self.module[k].parameters(), **config[k])
            for cls, k in zip(optim_clss, components)
        }
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(raypi.TorchPolicy)
    def compute_module_ouput(self, input_dict, state=None, seq_lens=None):
        # pylint:disable=unused-argument
        return input_dict[SampleBatch.CUR_OBS], state

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors, self.module, self.config))
        info.update(self._update_policy(batch_tensors, self.module, self.config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        critic_loss, info = self.compute_critic_loss(batch_tensors, module, config)
        self._optimizer.critics.zero_grad()
        critic_loss.backward()
        grad_stats = {
            "critic_grad_norm": nn.utils.clip_grad_norm_(
                module.critics.parameters(), float("inf")
            )
        }
        info.update(grad_stats)

        self._optimizer.critics.step()
        return info

    def compute_critic_loss(self, batch_tensors, module, config):
        """Compute loss for Q value function."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]

        with torch.no_grad():
            target_values = self._compute_critic_targets(batch_tensors, module, config)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in module.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "td_error": critic_loss.item(),
        }
        return critic_loss, stats

    @staticmethod
    def _compute_critic_targets(batch_tensors, module, config):
        rewards = batch_tensors[SampleBatch.REWARDS]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_acts = module.target_actor(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.target_critics], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    def _update_policy(self, batch_tensors, module, config):
        policy_loss, info = self.compute_policy_loss(batch_tensors, module, config)
        self._optimizer.actor.zero_grad()
        policy_loss.backward()
        info.update(self.extra_policy_grad_info())

        self._optimizer.actor.step()
        return info

    @staticmethod
    def compute_policy_loss(batch_tensors, module, config):
        """Compute loss for deterministic policy gradient."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.actor(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values)

        stats = {
            "policy_loss": max_objective.neg().item(),
            "qpi_mean": max_objective.item(),
        }
        return max_objective.neg(), stats

    def extra_policy_grad_info(self):
        """Return dict of extra info on policy gradient."""
        return {
            "policy_grad_norm": nn.utils.clip_grad_norm_(
                self.module.actor.parameters(), float("inf")
            ),
        }
