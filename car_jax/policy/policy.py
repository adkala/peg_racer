from jax import numpy as jnp
from jaxtyping import Array, PyTree, Key
from abc import ABC, abstractmethod

import jax
import gymnasium as gym


class Policy(ABC):
    """Base policy class"""

    @abstractmethod
    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        """
        Sample action from policy

        Args:
            obs_for_pi: Observation in policy format
            context: Policy context (e.g., RNN hidden state)
            key: JAX random key

        Returns:
            action: Sampled action
            context: Updated context
        """
        pass

    def init_context(self, key: Key) -> PyTree:
        """Initialize policy context"""
        return {}

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        """Reset policy context (e.g., at episode boundary)"""
        return context

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        """Sample action deterministically (e.g., mean of distribution)"""
        raise NotImplementedError("deterministic_sample is not implemented")


class PolicyAggregator(Policy):
    """Aggregates multiple policies for multi-agent control"""

    def __init__(self, policies: list[Policy]):
        self.policies = policies

    def __call__(self, obs_for_pi: PyTree, context: PyTree, key: Key):
        keys = jax.random.split(key, len(self.policies))
        actions = []
        contexts = []
        for i in range(len(self.policies)):
            action, policy_context = self.policies[i](
                obs_for_pi[i], context[i], keys[i]
            )

            actions.append(action)
            contexts.append(policy_context)

        return actions, contexts

    def init_context(self, key: Key) -> PyTree:
        keys = jax.random.split(key, len(self.policies))
        return [
            self.policies[i].init_context(keys[i]) for i in range(len(self.policies))
        ]

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        keys = jax.random.split(key, len(self.policies))
        return [
            self.policies[i].reset_context(context[i], keys[i])
            for i in range(len(self.policies))
        ]

    def deterministic_sample(self, obs_for_pi: PyTree, context: PyTree):
        actions = []
        contexts = []
        for i in range(len(self.policies)):
            action, policy_context = self.policies[i].deterministic_sample(
                obs_for_pi[i], context[i]
            )
            actions.append(action)
            contexts.append(policy_context)
        return actions, contexts


class DeterministicWrapper(Policy):
    """Wraps a policy to always use deterministic sampling"""

    def __init__(self, policy: Policy):
        self.policy = policy

    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        return self.policy.deterministic_sample(obs_for_pi, context)

    def init_context(self, key: Key) -> PyTree:
        return self.policy.init_context(key)

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        return self.policy.reset_context(context, key)

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        return self.policy.deterministic_sample(obs_for_pi, context)


class RandomPolicy(Policy):
    """Random policy that samples from action space"""

    def __init__(self, action_space: gym.Space):
        if isinstance(action_space, gym.spaces.Box):
            self.random_fn = lambda key: jax.random.uniform(
                key,
                shape=action_space.shape,
                minval=action_space.low,
                maxval=action_space.high,
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            self.random_fn = lambda key: jax.random.randint(
                key,
                shape=(),
                minval=0,
                maxval=action_space.n,
            )
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key) -> tuple[Array, PyTree]:
        out = self.random_fn(key)
        return out, context

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        """For random policy, deterministic sampling returns zeros (neutral action)"""
        # Return middle of action space
        return jnp.array([0.5, 0.0]), context


class ConstantPolicy(Policy):
    """Policy that always returns the same action"""

    def __init__(self, action: Array):
        self.action = jnp.array(action)

    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        return self.action, context

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        return self.action, context


class TrajCategoricalPolicy(Policy):
    """
    Policy that samples a trajectory at episode start and follows it
    Per-trajectory categorical sampling
    """

    def __init__(self, probs: Array):
        self.probs = probs

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key):
        out = context["current_choice"]
        return out, context

    def reset_context(self, context: PyTree, key: Key):
        key, subkey = jax.random.split(key)
        current_choice = jax.random.choice(subkey, a=len(self.probs), p=self.probs)
        return {"key": key, "current_choice": current_choice}

    def init_context(self, key: Key):
        key, subkey = jax.random.split(key)
        current_choice = jax.random.choice(subkey, a=len(self.probs), p=self.probs)
        return {"current_choice": current_choice}

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        out = context["current_choice"]
        return out, context


class ForwardPolicy(Policy):
    """Simple policy that always drives forward"""

    def __init__(self, target_vel: float = 0.8, target_steer: float = 0.0):
        self.target_vel = target_vel
        self.target_steer = target_steer

    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        action = jnp.array([self.target_vel, self.target_steer])
        return action, context

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        action = jnp.array([self.target_vel, self.target_steer])
        return action, context
