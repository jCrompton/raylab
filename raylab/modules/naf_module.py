"""Normalized Advantage Function neural network modules."""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override

from raylab.modules import (
    FullyConnected,
    TrilMatrix,
    NormalizedLinear,
    TanhSquash,
    GaussianNoise,
    DistRSample,
)


BASE_CONFIG = {
    "double_q": False,
    "exploration": None,
    "diag_gaussian_stddev": 0.1,
    "units": (32, 32),
    "activation": "ELU",
    "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing
    # function
    "beta": 1.2,
    # === Module Optimization ===
    # Whether to convert the module to a ScriptModule for faster inference
    "torch_script": False,
}


class NAFModule(nn.ModuleDict):
    """Module dict containing NAF's modules"""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        config = merge_dicts(BASE_CONFIG, config)
        self.update(self._make_naf(obs_space, action_space, config))
        self.target_value = nn.ModuleList(
            [self._make_value(obs_space, config) for _ in self.value]
        )
        self.target_value.load_state_dict(self.value.state_dict())
        self.policy = nn.Sequential(self.logits, self.mu, self.squash)
        self.update(self._make_sampler(obs_space, action_space, config))

    def _make_naf(self, obs_space, action_space, config):
        modules = self._make_components(obs_space, action_space, config)
        naf_module = NAF(
            modules["logits"],
            modules["value"],
            modules["tril"],
            nn.Sequential(modules["mu"], modules["squash"]),
        )
        modules["naf"] = nn.ModuleList([naf_module])
        modules["value"] = nn.ModuleList(
            [nn.Sequential(modules["logits"], modules["value"])]
        )
        if config["double_q"]:
            twin_modules = self._make_components(obs_space, action_space, config)
            twin_naf = NAF(
                twin_modules["logits"],
                twin_modules["value"],
                twin_modules["tril"],
                nn.Sequential(twin_modules["mu"], twin_modules["squash"]),
            )
            modules["naf"].append(twin_naf)
            modules["value"].append(
                nn.Sequential(twin_modules["logits"], twin_modules["value"])
            )

        return modules

    def _make_value(self, obs_space, config):
        logits = self._make_encoder(obs_space, config)
        return nn.Sequential(logits, nn.Linear(logits.out_features, 1))

    def _make_components(self, obs_space, action_space, config):
        logits = self._make_encoder(obs_space, config)
        components = {
            "logits": logits,
            "value": nn.Linear(logits.out_features, 1),
            "mu": NormalizedLinear(
                in_features=logits.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            ),
            "squash": TanhSquash(
                torch.as_tensor(action_space.low), torch.as_tensor(action_space.high)
            ),
            "tril": TrilMatrix(logits.out_features, action_space.shape[0]),
        }
        return components

    def _make_sampler(self, obs_space, action_space, config):
        # Configure sampler module based on exploration strategy
        if config["exploration"] == "full_gaussian":
            return {
                "sampler": MultivariateNormalSampler(
                    action_space,
                    self.logits,
                    self.mu,
                    self.tril,
                    script=config["torch_script"],
                )
            }
        if config["exploration"] == "parameter_noise":
            logits_module = self._make_encoder(obs_space, config)
            sampler = nn.Sequential(
                logits_module,
                NormalizedLinear(
                    in_features=logits_module.out_features,
                    out_features=action_space.shape[0],
                    beta=config["beta"],
                ),
                TanhSquash(
                    torch.as_tensor(action_space.low),
                    torch.as_tensor(action_space.high),
                ),
            )
            actor = nn.ModuleDict({"policy": self.policy, "behavior": sampler})
            return {"sampler": sampler, "actor": actor}
        if config["exploration"] == "diag_gaussian":
            expl_noise = GaussianNoise(config["diag_gaussian_stddev"])
            return {
                "sampler": nn.Sequential(self.logits, self.mu, expl_noise, self.squash)
            }
        return {"sampler": self.policy}

    @staticmethod
    def _make_encoder(obs_space, config):
        return FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            layer_norm=config.get(
                "layer_norm", config["exploration"] == "parameter_noise"
            ),
            **config["initializer_options"],
        )


class MultivariateNormalSampler(nn.Sequential):
    """Neural network module mapping inputs to MultivariateNormal parameters.

    This module is initialized to be close to a standard Normal distribution.
    """

    def __init__(self, action_space, logits_mod, mu_mod, tril_mod, *, script=False):
        params_module = MultivariateNormalParams(logits_mod, mu_mod, tril_mod)
        rsample_module = DistRSample(
            dists.MultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )

        if script:
            act_dim = action_space.shape[0]
            rsample_module = rsample_module.traced(
                {
                    "loc": torch.zeros(1, act_dim),
                    "scale_tril": torch.eye(act_dim).unsqueeze(0),
                }
            )
        super().__init__(params_module, rsample_module)


class NAF(nn.Module):
    """Neural network module implementing the Normalized Advantage Function (NAF)."""

    def __init__(self, logits_module, value_module, tril_module, action_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module
        self.advantage_module = AdvantageFunction(tril_module, action_module)

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        best_value = self.value_module(logits)
        advantage = self.advantage_module(logits, actions)
        return advantage + best_value


class MultivariateNormalParams(nn.Module):
    """Neural network module implementing a multivariate gaussian policy."""

    def __init__(self, logits_module, loc_module, scale_tril_module):
        super().__init__()
        self.logits_module = logits_module
        self.loc_module = loc_module
        self.scale_tril_module = scale_tril_module

    @override(nn.Module)
    def forward(self, obs):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        loc = self.loc_module(logits)
        scale_tril = self.scale_tril_module(logits)
        return {"loc": loc, "scale_tril": scale_tril}


class AdvantageFunction(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    def __init__(self, tril_module, action_module):
        super().__init__()
        self.tril_module = tril_module
        self.action_module = action_module

    @override(nn.Module)
    def forward(self, logits, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_module(logits)  # square matrix [..., N, N]
        best_action = self.action_module(logits)  # batch of actions [..., N]
        action_diff = (actions - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -0.5 * vec.transpose(-1, -2).matmul(vec).squeeze(-1)
        return advantage