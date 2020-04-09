"""SVGMaxEnt policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.modules import RewardFn
import raylab.utils.pytorch as ptu
from .svg_base_policy import SVGBaseTorchPolicy


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "model actor critic alpha"
)


class SVGMaxEntTorchPolicy(SVGBaseTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if self.config["target_entropy"] is None:
            self.config["target_entropy"] = -action_space.shape[0]
        assert (
            "target_critic" in self.module
        ), "SVGMaxEnt needs a target Value function!"

    @staticmethod
    @override(SVGBaseTorchPolicy)
    def get_default_config():
        """Return the default config for SVGMaxEnt"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_maxent import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGBaseTorchPolicy)
    def optimizer(self):
        """PyTorch optimizer to use."""
        config = self.config["torch_optimizer"]
        components = "model actor critic alpha".split()
        optim_clss = [
            ptu.get_optimizer_class(config[k].pop("type")) for k in components
        ]
        optims = {
            k: cls(self.module[k].parameters(), **config[k])
            for cls, k in zip(optim_clss, components)
        }
        return OptimizerCollection(**optims)

    @override(SVGBaseTorchPolicy)
    def set_reward_fn(self, reward_fn):
        torch_script = self.config["module"]["torch_script"]
        reward_fn = RewardFn(
            self.observation_space,
            self.action_space,
            reward_fn,
            torch_script=torch_script,
        )
        self.reward = torch.jit.script(reward_fn) if torch_script else reward_fn

    @override(SVGBaseTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_importance_sampling_ratios(batch_tensors)

        info.update(self._update_model_and_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        info.update(self._update_alpha(batch_tensors))

        info.update(self.extra_grad_info(batch_tensors))
        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    def _update_model_and_critic(self, batch_tensors):
        model_value_loss, info = self.compute_joint_model_value_loss(batch_tensors)

        self._optimizer.model.zero_grad()
        self._optimizer.critic.zero_grad()
        model_value_loss.backward()
        self._optimizer.model.step()
        self._optimizer.critic.step()

        return info

    @override(SVGBaseTorchPolicy)
    def _compute_value_targets(self, batch_tensors):
        _, logp = self.module.actor.sample(batch_tensors[SampleBatch.CUR_OBS])
        augmented_rewards = batch_tensors[SampleBatch.REWARDS] - logp

        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        next_vals = self.module.target_critic(next_obs).squeeze(-1)

        gamma = self.config["gamma"]
        targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            augmented_rewards,
            augmented_rewards + gamma * next_vals,
        )
        return targets

    def _update_actor(self, batch_tensors):
        svg_loss, info = self.compute_stochastic_value_gradient_loss(batch_tensors)

        self._optimizer.actor.zero_grad()
        svg_loss.backward()
        self._optimizer.actor.step()

        return info

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute bootstrapped Stochatic Value Gradient loss."""
        is_ratios = batch_tensors[self.IS_RATIOS]
        td_targets = self._compute_policy_td_targets(batch_tensors)
        svg_loss = -torch.mean(is_ratios * td_targets)
        return svg_loss, {"loss(actor)": svg_loss.item()}

    def _compute_policy_td_targets(self, batch_tensors):
        _acts, _logp = self.module.actor.reproduce(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        _next_obs, _ = self.module.model.reproduce(
            batch_tensors[SampleBatch.CUR_OBS],
            _acts,
            batch_tensors[SampleBatch.NEXT_OBS],
        )
        _rewards = self.reward(batch_tensors[SampleBatch.CUR_OBS], _acts, _next_obs)
        _augmented_rewards = _rewards - _logp
        _next_vals = self.module.critic(_next_obs).squeeze(-1)

        gamma = self.config["gamma"]
        return torch.where(
            batch_tensors[SampleBatch.DONES],
            _augmented_rewards,
            _augmented_rewards + gamma * _next_vals,
        )

    def _update_alpha(self, batch_tensors):
        alpha_loss, info = self.compute_alpha_loss(batch_tensors)

        self._optimizer.alpha.zero_grad()
        alpha_loss.backward()
        self._optimizer.alpha.step()

        return info

    def compute_alpha_loss(self, batch_tensors):
        """Compute entropy coefficient loss."""
        target_entropy = self.config["target_entropy"]

        with torch.no_grad():
            _, logp = self.module.actor.rsample(batch_tensors[SampleBatch.CUR_OBS])

        alpha = self.module.alpha()
        entropy_diff = torch.mean(-alpha * logp - alpha * target_entropy)
        info = {"loss(alpha)": entropy_diff.item(), "curr_alpha": alpha.item()}
        return entropy_diff, info

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):
        """Return statistics right after components are updated."""
        fetches = {
            f"grad_norm({k})": nn.utils.clip_grad_norm_(
                self.module[k].parameters(), float("inf")
            )
            for k in "model actor critic alpha".split()
        }
        _, logp = self.module.actor.sample(batch_tensors[SampleBatch.CUR_OBS])
        fetches["entropy"] = -logp.mean().item()
        return fetches
