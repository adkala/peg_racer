from abc import ABC, abstractmethod
from jaxtyping import Array, PyTree, Key
from car_jax.env import Observation
import jax.numpy as jnp
import jax


class Reward(ABC):
    """
    Reward function wrapper
    """

    @abstractmethod
    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """
        Compute reward

        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            context: Reward context (for stateful rewards)

        Returns:
            reward: Scalar reward
            context: Updated context
        """
        pass

    def init_context(self, key: Key) -> PyTree:
        """
        Initialize reward context

        Args:
            key: JAX random key

        Returns:
            context: Initial context
        """
        return {}


class VelocityReward(Reward):
    """
    Reward based on forward velocity
    """

    def __init__(self, agent_idx: int = 0, scale: float = 1.0):
        self.agent_idx = agent_idx
        self.scale = scale

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Reward = scale * vx"""
        state = next_obs.state
        vx = state.vx[self.agent_idx] if state.vx.ndim > 0 else state.vx
        reward = self.scale * vx
        return reward, context


class ProgressReward(Reward):
    """
    Reward based on progress along a path
    """

    def __init__(self, agent_idx: int = 0, scale: float = 1.0):
        self.agent_idx = agent_idx
        self.scale = scale

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Reward = scale * distance traveled"""
        prev_state = obs.state
        next_state = next_obs.state

        prev_x = prev_state.x[self.agent_idx] if prev_state.x.ndim > 0 else prev_state.x
        prev_y = prev_state.y[self.agent_idx] if prev_state.y.ndim > 0 else prev_state.y

        next_x = next_state.x[self.agent_idx] if next_state.x.ndim > 0 else next_state.x
        next_y = next_state.y[self.agent_idx] if next_state.y.ndim > 0 else next_state.y

        dx = next_x - prev_x
        dy = next_y - prev_y
        distance = jnp.sqrt(dx**2 + dy**2)

        reward = self.scale * distance
        return reward, context


class SurvivalReward(Reward):
    """
    Constant reward for each timestep (encourages agent to survive)
    """

    def __init__(self, reward_value: float = 1.0):
        self.reward_value = reward_value

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Constant reward"""
        return jnp.array(self.reward_value), context


class CompositeReward(Reward):
    """
    Composite reward that combines multiple reward functions
    """

    def __init__(self, rewards: list[Reward], weights: list[float]):
        assert len(rewards) == len(weights), "Number of rewards and weights must match"
        self.rewards = rewards
        self.weights = jnp.array(weights)

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Compute weighted sum of rewards"""
        total_reward = 0.0
        new_contexts = []

        for i, reward_fn in enumerate(self.rewards):
            r, ctx = reward_fn(obs, action, next_obs, context[i])
            total_reward += self.weights[i] * r
            new_contexts.append(ctx)

        return total_reward, new_contexts

    def init_context(self, key: Key) -> PyTree:
        """Initialize contexts for all sub-rewards"""
        keys = jax.random.split(key, len(self.rewards))
        return [self.rewards[i].init_context(keys[i]) for i in range(len(self.rewards))]
