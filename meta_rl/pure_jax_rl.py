from functools import partial
from typing import NamedTuple, Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
from distrax import Categorical
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnax.environments import environment, spaces

from meta_rl._typing import Actions, Obs, Params, PerTimestepScalar, WorldState


class ActorCritic(nn.Module):
    """Actor-critic network."""

    action_space: int
    internal_dim: int = 64

    @nn.compact
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
    info: jt.Array


BatchData = jt.PyTree[Tuple[ReplayBuffer, PerTimestepScalar, PerTimestepScalar]]


class TrainingHyperparameters(NamedTuple):
    env_name: str
    seed: int
    gamma: float = 0.9
    lam: float = 0.9
    clip_eps: float = 0.1
    max_grad_norm: float = 1
    num_epochs: int = 100
    batch_size: int = 64
    ent_coef: float = 0.01
    v_coef: float = 0.5
    num_envs: int = 10
    num_updates: int = 100
    num_steps: int = 128
    lr: float = 3e-4


def train(tp: TrainingHyperparameters):
    def _update_step(world_sate: WorldState, _):
        """Update the parameters of the actor-critic newtwork on
        vectorised environments."""

        def _env_step(world_state: WorldState, _) -> Tuple[WorldState, ReplayBuffer]:
            train_state, env_state, last_obs, key = world_state
            _, action_key, trainsition_key = jax.random.split(key, 1 + tp.num_envs)

            policy, value = actor_critic.apply(train_state.params, last_obs)
            action = policy.sample(seed=action_key)
            logprob = policy.log_prob(action)

            new_obs, env_state, reward, dones, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(trainsition_key, env_state, action, env_params)

            replay_buffer = ReplayBuffer(
                dones=dones,
                obs=new_obs,
                actions=action,
                values=value,  # type: ignore
                rewards=reward,
                logprobs=logprob,
                info=info,
            )
            _, new_key = jax.random.split(action_key, 2)
            new_scan_state = (
                train_state,
                env_state,
                new_obs,
                new_key,
            )
            return new_scan_state, replay_buffer

        def _calculate_gae(
            replay_buffer: ReplayBuffer, last_val: jt.Array
        ) -> Tuple[PerTimestepScalar, PerTimestepScalar]:
            def _recursive_gae(gae_and_next_obs, trasition):
                gae, next_obs = gae_and_next_obs
                next_value = actor_critic.apply(train_state.params, next_obs)
                delta = (
                    trasition.rewards
                    + tp.gamma * next_value * (1 - trasition.dones)  # type: ignore
                    - trasition.values
                )
                gae = delta + tp.gamma * tp.lam * (1 - trasition.dones) * gae
                return (gae, trasition.next_obs), gae

            _, advantages = jax.lax.scan(
                _recursive_gae,
                (jnp.zeros_like(last_val), last_val),  # adv is zero for the last step
                replay_buffer,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + replay_buffer.values

        def _network_epoch_update(
            update_data: Tuple[
                TrainState,
                ReplayBuffer,
                PerTimestepScalar,
                PerTimestepScalar,
                jt.PRNGKeyArray,
            ],
            _,
        ):
            def _update_minibatch(
                train_state: TrainState,
                batch_data: BatchData,
            ):
                replay_buffer, advantages, targets = batch_data

                def loss(
                    params: Params,
                    r_b: ReplayBuffer,
                    gae: PerTimestepScalar,
                    targets: PerTimestepScalar,
                ):
                    gae = (gae - jnp.mean(gae)) / (jnp.std(gae) + 1e-8)
                    pi, values = actor_critic.apply(params, r_b.obs)
                    logprobs = pi.log_prob(r_b.actions)
                    ratio = jnp.exp(logprobs - r_b.logprobs)
                    clipped_ratio = jnp.clip(ratio, 1 - tp.clip_eps, 1 + tp.clip_eps)
                    actor_loss = jnp.mean(-jnp.minimum(ratio, clipped_ratio) * gae)

                    value_loss = (values - targets) ** 2
                    clip_value_loss = jnp.square(
                        r_b.values
                        + jnp.clip(values - r_b.values, -tp.clip_eps, tp.clip_eps)
                    )
                    value_loss = 0.5 * jnp.mean(
                        jnp.maximum(value_loss, clip_value_loss)
                    )

                    total_loss = (
                        actor_loss
                        + value_loss * tp.v_coef
                        - tp.ent_coef * jnp.mean(pi.entropy())
                    )
                    return total_loss

                loss, grad = jax.value_and_grad(loss, has_aux=True)(
                    train_state.params, replay_buffer, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grad)
                return train_state, loss

            train_state, replay_buf, adv, targets, key = update_data
            _, key = jax.random.split(key)

            full_batch_size = tp.num_steps * tp.num_envs
            num_minibatches = full_batch_size // tp.batch_size
            permutation = jax.random.permutation(key, tp.batch_size)
            batch = (replay_buf, adv, targets)
            # Flatten observations across multiple envs
            batch = jax.tree_map(
                lambda x: x.reshape((full_batch_size, *x.shape[2:])), batch
            )
            shuffle = jax.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            mini_batches = jax.tree_map(
                lambda x: x.reshape([num_minibatches, -1] + list(x.shape[1:])), shuffle
            )

            train_state, losses = jax.lax.scan(
                _update_minibatch, train_state, mini_batches
            )

            scan_state = (train_state, replay_buf, adv, targets, key)
            return scan_state, replay_buf.info

        world_sate, replay_buffer = jax.lax.scan(
            _env_step, world_sate, None, tp.num_steps
        )
        train_state, env_state, last_obs, key = world_sate
        _, last_val = actor_critic.apply(train_state.params, last_obs)

        advantages, targets = _calculate_gae(replay_buffer, last_val)  # type: ignore

        update_data = (train_state, replay_buffer, advantages, targets, key)

        update_data, info = jax.lax.scan(
            _network_epoch_update, update_data, None, tp.num_epochs
        )
        train_state, _, _, _, key = update_data
        world_sate = (train_state, env_state, last_obs, key)
        return world_sate, info

    init_key = jax.random.PRNGKey(tp.seed)
    _, env_key, train_key, param_key = jax.random.split(init_key, 4)

    env, env_params = gymnax.make(tp.env_name)
    env = FlattenObservationWrapper(env)
    env_obs_space = env.observation_space(env_params).shape
    opt = optax.chain(optax.clip_by_global_norm(tp.max_grad_norm), optax.adam(tp.lr))
    actor_critic = ActorCritic(env.action_space(env_params).n)
    initial_params = actor_critic.init(param_key, jnp.empty((1, *env_obs_space)))
    train_state = TrainState.create(
        apply_fn=actor_critic.apply,
        params=initial_params,
        tx=opt,
    )

    obs, env_state = jax.vmap(env.reset, in_axes=(None, 0))(env_key, env_params)

    world_state = (train_state, env_state, obs, train_key)
    world_state, info = jax.lax.scan(_update_step, world_state, None, tp.num_updates)

    return world_state, info


if __name__ == "__main__":
    tp = TrainingHyperparameters("CartPole-v1", 0)
    world_state, info = train(tp)
