"""Policy for MAPO using PyTorch."""
import collections

import torch
import torch.nn as nn
from torch._six import inf
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch

import raylab.policy as raypi

import raylab.utils.pytorch as torch_util
from raylab.modules import RewardFn
from raylab.modules.catalog import get_module


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "policy critic model"
)


class MAPOTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        config.setdefault("module", {})
        config["module"]["torch_script"] = False
        super().__init__(observation_space, action_space, config)
        self.reward = None
        self.transition = None

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.mapo.mapo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["double_q"] = config["clipped_double_q"]
        for key in (
            "exploration",
            "exploration_gaussian_sigma",
            "smooth_target_policy",
            "target_gaussian_sigma",
        ):
            module_config[key] = config[key]
        module = get_module(
            module_config["name"], obs_space, action_space, module_config
        )
        self.check_model(module.model.sampler)
        return module

    @override(raypi.TorchPolicy)
    def optimizer(self):
        policy_optim_name = self.config["policy_optimizer"]["name"]
        policy_optim_cls = torch_util.get_optimizer_class(policy_optim_name)
        policy_optim_options = self.config["policy_optimizer"]["options"]
        policy_optim = policy_optim_cls(
            self.module.actor.parameters(), **policy_optim_options
        )

        critic_optim_name = self.config["critic_optimizer"]["name"]
        critic_optim_cls = torch_util.get_optimizer_class(critic_optim_name)
        critic_optim_options = self.config["critic_optimizer"]["options"]
        critic_optim = critic_optim_cls(
            self.module.critics.parameters(), **critic_optim_options
        )

        if self.config["true_model"]:
            model_optim = None
        else:
            model_optim_name = self.config["model_optimizer"]["name"]
            model_optim_cls = torch_util.get_optimizer_class(model_optim_name)
            model_optim_options = self.config["model_optimizer"]["options"]
            model_optim = model_optim_cls(
                self.module.model.parameters(), **model_optim_options
            )

        return OptimizerCollection(
            policy=policy_optim, critic=critic_optim, model=model_optim
        )

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""
        torch_script = self.config["module"]["torch_script"]
        reward_fn = RewardFn(
            self.observation_space,
            self.action_space,
            reward_fn,
            torch_script=torch_script,
        )
        self.reward = torch.jit.script(reward_fn) if torch_script else reward_fn

    def set_transition_fn(self, transition_fn):
        """Set the transition function to use when unrolling the policy and model."""
        torch_script = self.config["module"]["torch_script"]
        transition = EnvTransition(
            self.observation_space,
            self.action_space,
            transition_fn,
            torch_script=torch_script,
        )
        self.transition = torch.jit.script(transition) if torch_script else transition
        self.check_model(self.transition)

    def check_model(self, sampler):
        """Verify that the transition model is appropriate for the desired estimator."""
        if self.config["grad_estimator"] == "score_function":
            obs = self.convert_to_tensor(self.observation_space.sample())[None]
            act = self.convert_to_tensor(self.action_space.sample())[None]
            _, logp = sampler(obs, act.requires_grad_())
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
        if self.config["grad_estimator"] == "pathwise_derivative":
            obs = self.convert_to_tensor(self.observation_space.sample())[None]
            act = self.convert_to_tensor(self.action_space.sample())[None]
            samp, _ = sampler(obs.requires_grad_(), act.requires_grad_())
            samp.mean().backward()
            assert (
                obs.grad is not None and act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noise_free_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.actor.policy[:-1](obs_tensors).numpy()

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noisy_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.actor.behavior[:-1](obs_tensors).numpy()

    @torch.no_grad()
    @override(raypi.TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

        if self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["greedy"]:
            actions = self.module.actor.policy(obs_batch)
        else:
            actions = self.module.actor.behavior(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors, self.module, self.config))
        if not self.config["true_model"]:
            info.update(self._update_model(batch_tensors, self.module, self.config))
        info.update(self._update_policy(batch_tensors, self.module, self.config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        critic_loss, info = self.compute_critic_loss(batch_tensors, module, config)
        self._optimizer.critic.zero_grad()
        critic_loss.backward()
        grad_stats = {
            "critic_grad_norm": nn.utils.clip_grad_norm_(
                module.critics.parameters(), float("inf")
            )
        }
        info.update(grad_stats)

        self._optimizer.critic.step()
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

        next_acts = module.actor.target_policy(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.target_critics], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    def _update_model(self, batch_tensors, module, config):
        if config["model_loss"] == "decision_aware":
            model_loss, info = self.compute_decision_aware_loss(
                batch_tensors, module, config
            )
        elif config["model_loss"] == "mle":
            model_loss, info = self.compute_mle_loss(batch_tensors, module)

        self._optimizer.model.zero_grad()
        model_loss.backward()
        grad_stats = {
            "model_grad_norm": nn.utils.clip_grad_norm_(
                module.model.parameters(), float("inf")
            )
        }
        info.update(grad_stats)

        self._optimizer.model.step()
        return info

    def compute_decision_aware_loss(self, batch_tensors, module, config):
        """Compute policy gradient-aware (PGA) model loss."""
        with self.freeze_nets("model"):
            dpg_loss, dpg_info = self.compute_dpg_loss(batch_tensors, module, config)
            dpg_grads = torch.autograd.grad(dpg_loss, module.actor.parameters())

        madpg_loss, _ = self.compute_madpg_loss(batch_tensors, module, config)
        madpg_grads = torch.autograd.grad(
            madpg_loss, module.actor.parameters(), create_graph=True
        )

        total_norm = self.compute_total_diff_norm(
            dpg_grads, madpg_grads, config["norm_type"]
        )

        info = {"decision_aware_loss": total_norm.item()}
        info.update({"target_" + k: v for k, v in dpg_info.items()})
        return total_norm, info

    @staticmethod
    def compute_dpg_loss(batch_tensors, module, config):
        """Compute loss for deterministic policy gradient."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.actor.policy(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values)

        stats = {
            "policy_loss": max_objective.neg().item(),
            "qpi_mean": max_objective.item(),
        }
        return max_objective.neg(), stats

    def compute_madpg_loss(self, batch_tensors, module, config):
        """Compute loss for model-aware deterministic policy gradient."""
        # pylint: disable=too-many-locals
        gamma = config["gamma"]
        rollout_len = config["model_rollout_len"]
        transition = self.transition if config["true_model"] else module.model.sampler

        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = module.actor.policy(obs)
        n_samples = config["num_model_samples"]
        next_obs, logp = transition(
            obs.expand((n_samples,) + obs.shape),
            actions.expand((n_samples,) + actions.shape),
        )
        rews = [self.reward(obs, actions, next_obs)]

        for _ in range(config["model_rollout_len"] - 1):
            obs = next_obs
            actions = module.actor.policy(obs)
            next_obs, _ = transition(obs, actions)
            rews.append(self.reward(obs, actions, next_obs))

        rews = (torch.stack(rews).T * gamma ** torch.arange(rollout_len).float()).T
        critic = module.critics[0](next_obs, module.actor.policy(next_obs)).squeeze(-1)
        values = rews.sum(0) + gamma ** rollout_len * critic

        if config["grad_estimator"] == "score_function":
            baseline = (module.critics[0](obs, actions).squeeze(-1) - rews) / gamma
            loss = torch.mean(logp * (values - baseline).detach(), dim=0).mean().neg()
        elif config["grad_estimator"] == "pathwise_derivative":
            loss = torch.mean(values, dim=0).mean().neg()
        return (
            loss,
            {
                "model_aware_loss": loss.item(),
                "mb_values": values.mean(dim=0).mean().item(),
            },
        )

    @staticmethod
    def compute_total_diff_norm(atensors, btensors, norm_type):
        """Compute the norm of the difference of tensors as a flattened vector."""
        if norm_type == inf:
            total_norm = max((a - b).abs().max() for a, b in zip(atensors, btensors))
        else:
            total_norm = 0
            for atensor, btensor in zip(atensors, btensors):
                norm = (atensor - btensor).norm(norm_type)
                total_norm += norm ** norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        return total_norm

    @staticmethod
    def compute_mle_loss(batch_tensors, module):
        """Compute Maximum Likelihood Estimation (MLE) model loss."""
        avg_logp = module.model.logp(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.NEXT_OBS],
        ).mean()
        loss = avg_logp.neg()
        info = {"mle_loss": loss.item()}
        return loss, info

    def _update_policy(self, batch_tensors, module, config):
        policy_loss, info = self.compute_madpg_loss(batch_tensors, module, config)
        self._optimizer.policy.zero_grad()
        policy_loss.backward()
        info.update(self.extra_policy_grad_info())

        self._optimizer.policy.step()
        return info

    def extra_policy_grad_info(self):
        """Return dict of extra info on policy gradient."""
        info = {
            "policy_grad_norm": nn.utils.clip_grad_norm_(
                self.module.actor.parameters(), float("inf")
            ),
        }
        if self.config["exploration"] == "parameter_noise":
            info["param_noise_stddev"] = self.curr_param_stddev
        return info


class EnvTransition(nn.Module):
    """Wrapper module around existing env transition function."""

    def __init__(self, obs_space, action_space, transition_fn, torch_script=False):
        super().__init__()
        if torch_script:
            obs = torch.as_tensor(obs_space.sample())[None]
            action = torch.as_tensor(action_space.sample())[None]
            transition_fn = torch.jit.trace(transition_fn, (obs, action))
        self.transition_fn = transition_fn

    @override(nn.Module)
    def forward(self, obs, action):  # pylint:disable=arguments-differ
        return self.transition_fn(obs, action)
