from typing import Tuple

import jaxtyping as jt
from flax.training.train_state import TrainState
from gymnax.environments.environment import EnvState
from jaxtyping import PRNGKeyArray

Obs = jt.Float[jt.Array, "*num_envs num_timesteps state_shape"]
Actions = jt.Int[jt.Array, "*num_envs num_timesteps action_shape"]
PerTimestepScalar = jt.Float[jt.Array, "*num_envs num_timesteps"]

WorldState = Tuple[TrainState, EnvState, Obs, PRNGKeyArray]
Params = jt.PyTree[jt.Float[jt.Array, "_"]]
