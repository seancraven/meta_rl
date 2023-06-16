from functools import partial
from typing import NamedTuple, Optional, Tuple, Union

import chex
import gymnax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from distrax import Categorical
from flax import linen as nn
from flax.training import train_state
from gymnax.environments import environment, spaces

from meta_rl._typing import Actions, Obs, PerTimestepScalar, ScanState


class ActorCritic(nn.Module):
    """Actor-critic network."""

    action_space: int
    internal_dim: int = 64

    def __call__(self, obs: Obs) -> Tuple[Categorical, PerTimestepScalar]:
        logits = nn.Dense(self.internal_dim)(obs)
        logits = nn.relu(logits)
        logits = nn.Dense(self.internal_dim)(logits)
        logits = nn.relu(logits)
        logits = nn.Dense(self.action_space)(logits)
        dist = Categorical(logits=logits)

        value = nn.Dense(self.internal_dim)(obs)
        value = nn.relu(value)
        value = nn.Dense(self.internal_dim)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)

        return dist, value


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    # PURE JAX RL RIP
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),  # type: ignore
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class ReplayBuffer(NamedTuple):
    dones: PerTimestepScalar
    obs: Obs
    actions: Actions
    values: PerTimestepScalar
    rewards: PerTimestepScalar
    logprobs: PerTimestepScalar
    next_obs: Obs


def train(train_seed, env_name):
    env, env_params = gymnax.make(env_name)

    env = FlattenObservationWrapper(env)
    env_obs_space = env.observation_space(env_params).shape

    opt = optax.adam(1e-3)
    num_envs = 10

    init_key = jax.random.PRNGKey(train_seed)
    actor_critic = ActorCritic(env.action_space(env_params).n)
    initial_params = actor_critic.init(init_key, jnp.empty((1, env_obs_space)))
    train_state = train_state.TrainState.create(
        apply_fn=actor_critic.apply,
        params=initial_params,
        tx=opt,
    )

    def _update_step(scan_state: ScanState, _):
        def _env_step(scan_state: ScanState, _):
            train_state, env_state, last_obs, key = scan_state
            _, action_key, trainsition_key = jax.random.split(key, 1 + num_envs)
            policy, value = actor_critic.apply(train_state.params, last_obs)
            action = policy.sample(seed=action_key)
            logprob = policy.log_prob(action)
            v_obs, env_state, reward, dones, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(trainsition_key, env_state, action, env_params)
