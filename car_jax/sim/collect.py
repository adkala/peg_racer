from car_jax.policy import Policy
from car_jax.env import Observation

from jax import lax, numpy as jnp
from flax import struct

from dataclasses import replace
from jaxtyping import Key, Array, PyTree

import jax


@struct.dataclass
class Transition:
    """Stores a single environment transition"""
    obs: PyTree[Array]
    action: PyTree[Array]
    next_obs: PyTree[Array]
    reward: Array
    episode_reward: Array
    terminated: bool
    truncated: bool
    step_count: int
    # metrics
    agent_win: Array

    def __getitem__(self, idx):
        return Transition(
            obs=self.obs[idx],
            action=self.action[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[..., idx],
            episode_reward=self.episode_reward[..., idx],
            terminated=self.terminated,
            truncated=self.truncated,
            step_count=self.step_count,
            #
            agent_win=self.agent_win,
        )


@struct.dataclass
class Carry:
    """State carried through collection loop"""
    obs: Observation
    obs_for_pi: PyTree[Array]
    episode_reward: PyTree[float]
    policy_context: PyTree
    agent_context: PyTree
    sim_context: PyTree
    done: bool
    reset_key: Key  # convenience for reset


@struct.dataclass
class ObsHolder:
    """Holds previous and next observation"""
    obs: Observation
    next_obs: Observation


def collect_factory(
    sim_reset,
    sim_step,
    pi,
    record_indices,
    timesteps,
    debug=False,
    return_obs=False,
    deterministic=False,
):
    """
    Create a collection function for gathering environment transitions

    Args:
        sim_reset: Function to reset simulation
        sim_step: Function to step simulation
        pi: Policy instance (un-jitted)
        record_indices: Indices to record from observations/actions
        timesteps: Number of timesteps to collect
        debug: If True, return carry state
        return_obs: If True, return observations
        deterministic: If True, use deterministic policy sampling

    Returns:
        collect: Function that collects transitions
    """

    assert isinstance(pi, Policy), "pi must be a Policy (ensure it is not jitted)"

    def record_pytree(pytree):
        return [pytree[i] for i in record_indices]

    def step_branch(carry):
        if deterministic:
            action_from_pi, policy_context = pi.deterministic_sample(
                carry.obs_for_pi, carry.policy_context
            )
        else:
            action_from_pi, policy_context = pi(
                carry.obs_for_pi, carry.policy_context, carry.obs.state.rng.policy_key
            )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            next_obs_for_pi,
            next_agent_context,
            next_sim_context,
        ) = sim_step(carry.obs, action_from_pi, carry.agent_context, carry.sim_context)

        transition = Transition(
            obs=record_pytree(carry.obs_for_pi),
            action=record_pytree(action_from_pi),
            next_obs=record_pytree(next_obs_for_pi),
            reward=reward,
            episode_reward=carry.episode_reward + reward,
            terminated=terminated,
            truncated=truncated,
            step_count=carry.obs.timestep,
            #
            agent_win=next_obs.state.agent_win,
        )

        prev_obs = carry.obs

        carry = Carry(
            obs=next_obs,
            obs_for_pi=next_obs_for_pi,
            episode_reward=carry.episode_reward + reward,
            policy_context=policy_context,
            agent_context=next_agent_context,
            sim_context=next_sim_context,
            done=terminated | truncated,
            reset_key=next_obs.state.rng.next_key,
        )

        if debug:
            out = (carry, transition)
        elif return_obs:
            out = (ObsHolder(obs=prev_obs, next_obs=next_obs), transition)
        else:
            out = transition

        return carry, out

    dummy_obs, dummy_obs_for_pi, dummy_agent_context, dummy_sim_context = sim_reset(
        jax.random.key(0)
    )
    dummy_policy_context = pi.init_context(jax.random.key(0))
    action_from_pi, _ = pi(dummy_obs_for_pi, dummy_policy_context, jax.random.key(0))
    _, dummy_reward, _, _, _, _, _ = sim_step(
        dummy_obs, action_from_pi, dummy_agent_context, dummy_sim_context
    )

    def reset_branch(carry):
        sim_key, policy_key = jax.random.split(carry.reset_key, 2)
        obs, obs_for_pi, agent_context, sim_context = sim_reset(sim_key)
        return step_branch(
            Carry(
                obs=obs,
                obs_for_pi=obs_for_pi,
                episode_reward=jnp.zeros_like(dummy_reward),
                policy_context=pi.reset_context(carry.policy_context, policy_key),
                agent_context=agent_context,
                sim_context=sim_context,
                done=False,
                reset_key=obs.state.rng.next_key,
            )
        )

    def collect_step(carry, _):
        """use with lax.scan"""
        return lax.cond(carry.done, reset_branch, step_branch, carry)

    def collect(key: Key, policy_context: PyTree):
        dummy_carry = Carry(
            obs=dummy_obs,
            obs_for_pi=dummy_obs_for_pi,
            episode_reward=jnp.zeros_like(dummy_reward),
            policy_context=policy_context,
            agent_context=dummy_agent_context,
            sim_context=dummy_sim_context,
            done=True,
            reset_key=key,
        )

        _, transitions = lax.scan(collect_step, dummy_carry, None, timesteps)

        return transitions

    return collect


def batch_collect_factory(
    sim_reset,
    sim_step,
    pi,
    record_indices,
    total_timesteps,
    num_envs,
    eval_mode=False,
    return_obs=False,
    deterministic=False,
):
    """
    Create a batched collection function for parallel environment execution

    Args:
        sim_reset: Function to reset simulation
        sim_step: Function to step simulation
        pi: Policy instance (un-jitted)
        record_indices: Indices to record from observations/actions
        total_timesteps: Total number of timesteps to collect
        num_envs: Number of parallel environments
        eval_mode: If True, return observations (doesn't force determinism)
        return_obs: If True, return observations
        deterministic: If True, use deterministic policy sampling

    Returns:
        batch_collect: Function that collects transitions in parallel
    """
    return_obs = eval_mode or return_obs

    timesteps_per_env = total_timesteps // num_envs

    if timesteps_per_env * num_envs != total_timesteps:
        print(
            f"\n\nWARNING: total_timesteps {total_timesteps} is not divisible by num_envs {num_envs}. Will collect {timesteps_per_env * num_envs} timesteps.\n\n"
        )

    collect = collect_factory(
        sim_reset,
        sim_step,
        pi,
        record_indices,
        timesteps_per_env,
        return_obs=return_obs,
        deterministic=deterministic,
    )

    def flatten_leaf(node):
        if isinstance(node, jnp.ndarray):
            return node.reshape(-1, *node.shape[2:])

    def batch_collect(key: Key, policy_context: PyTree):
        policy_key, collection_key = jax.random.split(key, 2)

        policy_keys = jax.random.split(policy_key, num_envs)
        policy_context = jax.vmap(pi.reset_context, in_axes=(None, 0))(
            policy_context, policy_keys
        )

        collection_keys = jax.random.split(collection_key, num_envs)
        batch_transitions = jax.vmap(collect)(collection_keys, policy_context)
        transitions = jax.tree_util.tree_map(flatten_leaf, batch_transitions)

        if return_obs:
            truncated = batch_transitions[1].truncated.at[..., -1].set(True).reshape(-1)
            transitions_obj = replace(transitions[1], truncated=truncated)
            transitions = (transitions[0], transitions_obj)
        else:
            truncated = batch_transitions.truncated.at[..., -1].set(True).reshape(-1)
            transitions = replace(transitions, truncated=truncated)

        return transitions

    return batch_collect
