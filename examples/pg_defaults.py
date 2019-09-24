"""Tune experiment configuration for PG on CartPoleSwingUp.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'PG' --local-dir <experiment dir>
    --config examples/pg_defaults.py --stop timesteps_total 1000000`
"""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "TimeAwareEnv",
        "env_config": {"env_id": "CartPoleSwingUp", "max_episode_steps": 250},
        # Don't set 'done' at the end of the episode. Note that you still need to
        # set this if soft_horizon=True, unless your env is actually running
        # forever without returning done=True.
        "no_done_at_end": False,
        # === PG ===
        # No remote workers by default
        "num_workers": 0,
        # Learning rate
        "lr": 0.001,
        # Use PyTorch as backend
        "use_pytorch": True,
        # === Model ===
        "model": {
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "tanh",
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [64, 64],
            # For control envs, documented in ray.rllib.models.Model
            "free_log_std": True,
        },
    }