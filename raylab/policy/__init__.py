"""Collection of custom RLlib Policy classes."""
from raylab.policy.torch_policy import TorchPolicy
from raylab.policy.kl_coeff_mixin import AdaptiveKLCoeffMixin
from raylab.policy.parameter_noise_mixin import AdaptiveParamNoiseMixin
from raylab.policy.pure_exploration_mixin import PureExplorationMixin
from raylab.policy.target_networks_mixin import TargetNetworksMixin
