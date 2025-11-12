"""
Racing rewards for multi-agent track racing.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Key
from car_jax.env import Observation
from car_jax.sim.reward import Reward


def wrap_diff(a, b, L):
    """Calculate wrapped difference for circular track"""
    d = a - b
    d = jnp.where(d < -L / 2.0, d + L, d)
    d = jnp.where(d > L / 2.0, d - L, d)
    return d


class RelativeProgressReward(Reward):
    """
    Reward based on relative progress compared to other cars.

    Tracks progress along the track and rewards cars for gaining
    position relative to the car(s) behind them.
    """

    def __init__(self, agent_idx: int, waypoint_generator, track_L: float,
                 num_cars: int = 3, scale: float = 1.0):
        """
        Initialize relative progress reward.

        Args:
            agent_idx: Index of this agent
            waypoint_generator: Waypoint generator to compute track progress
            track_L: Total track length
            num_cars: Total number of cars
            scale: Reward scale factor
        """
        self.agent_idx = agent_idx
        self.wp_gen = waypoint_generator
        self.track_L = track_L
        self.num_cars = num_cars
        self.scale = scale

    def init_context(self, key: Key) -> PyTree:
        """Initialize context with last relative position"""
        return {'last_rel': jnp.array(0.0)}

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """
        Compute reward based on relative progress.

        Reward is the change in position relative to the cars behind,
        encouraging the agent to maintain and improve its position.
        """
        # Compute track progress for all cars in next state
        next_state = next_obs.state
        s_values = []

        for i in range(self.num_cars):
            x = next_state.x[i] if next_state.x.ndim > 0 else next_state.x
            y = next_state.y[i] if next_state.y.ndim > 0 else next_state.y
            psi = next_state.psi[i] if next_state.psi.ndim > 0 else next_state.psi
            vx = next_state.vx[i] if next_state.vx.ndim > 0 else next_state.vx
            vy = next_state.vy[i] if next_state.vy.ndim > 0 else next_state.vy

            obs5 = jnp.array([x, y, psi, vx, vy])
            _, _, s, _ = self.wp_gen(obs5, vx)
            s_values.append(s)

        s_values = jnp.array(s_values)

        # Calculate relative position: distance ahead of the furthest car behind
        other_indices = jnp.array([i for i in range(self.num_cars) if i != self.agent_idx])
        s_self = s_values[self.agent_idx]

        # Get maximum s of other cars (furthest ahead of the others)
        s_others = s_values[other_indices]
        s_max_other = jnp.max(s_others)

        # Relative position is how far ahead we are of the furthest other car
        rel_pos = wrap_diff(s_self, s_max_other, self.track_L)

        # Reward is change in relative position
        reward = self.scale * wrap_diff(rel_pos, context['last_rel'], self.track_L)

        # Update context with new relative position
        new_context = {'last_rel': rel_pos}

        return reward, new_context


class TrackProgressReward(Reward):
    """
    Simple reward based on absolute track progress.
    Rewards forward progress along the track.
    """

    def __init__(self, agent_idx: int, waypoint_generator, track_L: float, scale: float = 1.0):
        """
        Initialize track progress reward.

        Args:
            agent_idx: Index of this agent
            waypoint_generator: Waypoint generator to compute track progress
            track_L: Total track length
            scale: Reward scale factor
        """
        self.agent_idx = agent_idx
        self.wp_gen = waypoint_generator
        self.track_L = track_L
        self.scale = scale

    def init_context(self, key: Key) -> PyTree:
        """Initialize context with last progress"""
        return {'last_s': jnp.array(0.0)}

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Compute reward based on track progress"""
        next_state = next_obs.state

        x = next_state.x[self.agent_idx] if next_state.x.ndim > 0 else next_state.x
        y = next_state.y[self.agent_idx] if next_state.y.ndim > 0 else next_state.y
        psi = next_state.psi[self.agent_idx] if next_state.psi.ndim > 0 else next_state.psi
        vx = next_state.vx[self.agent_idx] if next_state.vx.ndim > 0 else next_state.vx
        vy = next_state.vy[self.agent_idx] if next_state.vy.ndim > 0 else next_state.vy

        obs5 = jnp.array([x, y, psi, vx, vy])
        _, _, s, _ = self.wp_gen(obs5, vx)

        # Reward is progress made
        reward = self.scale * wrap_diff(s, context['last_s'], self.track_L)

        new_context = {'last_s': s}

        return reward, new_context


class LateralErrorPenalty(Reward):
    """Penalty for large lateral errors (going off track)"""

    def __init__(self, agent_idx: int, waypoint_generator, scale: float = 0.1):
        """
        Initialize lateral error penalty.

        Args:
            agent_idx: Index of this agent
            waypoint_generator: Waypoint generator to compute lateral error
            scale: Penalty scale factor
        """
        self.agent_idx = agent_idx
        self.wp_gen = waypoint_generator
        self.scale = scale

    def __call__(
        self,
        obs: Observation,
        action: Array,
        next_obs: Observation,
        context: PyTree,
    ) -> tuple[Array, PyTree]:
        """Compute penalty based on lateral error"""
        next_state = next_obs.state

        x = next_state.x[self.agent_idx] if next_state.x.ndim > 0 else next_state.x
        y = next_state.y[self.agent_idx] if next_state.y.ndim > 0 else next_state.y
        psi = next_state.psi[self.agent_idx] if next_state.psi.ndim > 0 else next_state.psi
        vx = next_state.vx[self.agent_idx] if next_state.vx.ndim > 0 else next_state.vx
        vy = next_state.vy[self.agent_idx] if next_state.vy.ndim > 0 else next_state.vy

        obs5 = jnp.array([x, y, psi, vx, vy])
        _, _, _, e = self.wp_gen(obs5, vx)

        # Penalty is proportional to squared lateral error
        penalty = -self.scale * e**2

        return penalty, context
