from abc import ABC, abstractmethod
from jaxtyping import Array, PyTree, Key
from car_jax.env import Observation


class Agent(ABC):
    """
    Agent wrapper that transforms observations and actions between
    environment space and policy space
    """

    @abstractmethod
    def obs_env2pi(self, obs: Observation, context: PyTree) -> tuple[PyTree, PyTree]:
        """
        Transform observation from environment to policy format

        Args:
            obs: Observation from environment
            context: Agent context (e.g., history, internal state)

        Returns:
            obs_for_pi: Observation in policy format
            context: Updated context
        """
        pass

    @abstractmethod
    def action_pi2env(self, action: PyTree, context: PyTree) -> tuple[Array, PyTree]:
        """
        Transform action from policy to environment format

        Args:
            action: Action from policy
            context: Agent context

        Returns:
            action: Action in environment format
            context: Updated context
        """
        pass

    def init_context(self, key: Key) -> PyTree:
        """
        Initialize agent context

        Args:
            key: JAX random key

        Returns:
            context: Initial context
        """
        return {}


class IdentityAgent(Agent):
    """
    Simple agent that passes observations and actions through without modification
    Extracts single agent observation from multi-agent environment
    """

    def __init__(self, agent_idx: int = 0):
        self.agent_idx = agent_idx

    def obs_env2pi(self, obs: Observation, context: PyTree) -> tuple[Array, PyTree]:
        """Return observation for this agent as array [x, y, psi, vx, vy, omega]"""
        state = obs.state
        obs_for_pi = jnp.array([
            state.x[self.agent_idx] if state.x.ndim > 0 else state.x,
            state.y[self.agent_idx] if state.y.ndim > 0 else state.y,
            state.psi[self.agent_idx] if state.psi.ndim > 0 else state.psi,
            state.vx[self.agent_idx] if state.vx.ndim > 0 else state.vx,
            state.vy[self.agent_idx] if state.vy.ndim > 0 else state.vy,
            state.omega[self.agent_idx] if state.omega.ndim > 0 else state.omega,
        ])
        return obs_for_pi, context

    def action_pi2env(self, action: Array, context: PyTree) -> tuple[Array, PyTree]:
        """Pass action through directly"""
        return action, context


class RelativePositionAgent(Agent):
    """
    Agent that provides relative position observations
    Useful for multi-agent scenarios
    """

    def __init__(self, agent_idx: int, other_agent_indices: list[int]):
        self.agent_idx = agent_idx
        self.other_agent_indices = other_agent_indices

    def obs_env2pi(self, obs: Observation, context: PyTree) -> tuple[Array, PyTree]:
        """
        Return observation with ego state and relative positions to other agents
        Format: [x, y, psi, vx, vy, omega, dx1, dy1, dpsi1, dx2, dy2, dpsi2, ...]
        """
        state = obs.state

        # Ego state
        ego_x = state.x[self.agent_idx] if state.x.ndim > 0 else state.x
        ego_y = state.y[self.agent_idx] if state.y.ndim > 0 else state.y
        ego_psi = state.psi[self.agent_idx] if state.psi.ndim > 0 else state.psi
        ego_vx = state.vx[self.agent_idx] if state.vx.ndim > 0 else state.vx
        ego_vy = state.vy[self.agent_idx] if state.vy.ndim > 0 else state.vy
        ego_omega = state.omega[self.agent_idx] if state.omega.ndim > 0 else state.omega

        ego_obs = jnp.array([ego_x, ego_y, ego_psi, ego_vx, ego_vy, ego_omega])

        # Relative observations
        relative_obs = []
        for other_idx in self.other_agent_indices:
            other_x = state.x[other_idx]
            other_y = state.y[other_idx]
            other_psi = state.psi[other_idx]

            dx = other_x - ego_x
            dy = other_y - ego_y
            dpsi = other_psi - ego_psi

            # Normalize angle difference
            dpsi = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))

            relative_obs.extend([dx, dy, dpsi])

        obs_for_pi = jnp.concatenate([ego_obs, jnp.array(relative_obs)])
        return obs_for_pi, context

    def action_pi2env(self, action: Array, context: PyTree) -> tuple[Array, PyTree]:
        """Pass action through directly"""
        return action, context


# Import jnp for the agents above
import jax.numpy as jnp
