"""
Waypoint-following agent that provides track-relative observations.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Key
from car_jax.env import Observation
from car_jax.sim.agent import Agent


class WaypointAgent(Agent):
    """
    Agent that provides waypoint-based observations for track following.

    Observations include:
    - Track progress (s) and lateral error (e)
    - Heading error relative to track
    - Car velocities (vx, vy, omega)
    - Track curvature at current and lookahead positions
    - Relative positions to other cars (optional)
    """

    def __init__(self, agent_idx: int, waypoint_generator, track_L: float,
                 num_cars: int = 3, include_other_cars: bool = True):
        """
        Initialize waypoint agent.

        Args:
            agent_idx: Index of this agent
            waypoint_generator: Waypoint generator instance
            track_L: Total track length
            num_cars: Total number of cars in environment
            include_other_cars: Whether to include other cars in observation
        """
        self.agent_idx = agent_idx
        self.wp_gen = waypoint_generator
        self.track_L = track_L
        self.num_cars = num_cars
        self.include_other_cars = include_other_cars

    def obs_env2pi(self, obs: Observation, context: PyTree) -> tuple[Array, PyTree]:
        """
        Transform environment observation to waypoint-based observation.

        Returns observation vector:
        If include_other_cars=True (multi-agent racing):
            [front_s_diff, front_e, self_e, front_theta_diff, front_vx, front_vy, front_omega,
             self_theta_diff, self_vx, self_vy, self_omega, front_curv, self_curv,
             front_curv_lh, self_curv_lh]  (15-dim)

        If include_other_cars=False (single agent):
            [self_e, self_theta_diff, self_vx, self_vy, self_omega, self_curv, self_curv_lh]  (7-dim)
        """
        state = obs.state

        # Extract features for all cars
        feats = []
        for i in range(self.num_cars):
            x = state.x[i] if state.x.ndim > 0 else state.x
            y = state.y[i] if state.y.ndim > 0 else state.y
            psi = state.psi[i] if state.psi.ndim > 0 else state.psi
            vx = state.vx[i] if state.vx.ndim > 0 else state.vx
            vy = state.vy[i] if state.vy.ndim > 0 else state.vy

            obs5 = jnp.array([x, y, psi, vx, vy])
            tgt, _, s, e = self.wp_gen(obs5, vx)

            theta = tgt[0, 2]
            theta_diff = jnp.arctan2(jnp.sin(theta - psi), jnp.cos(theta - psi))
            curv = tgt[0, 3]
            curv_lh = tgt[-1, 3]

            feats.append(jnp.array([s, e, theta_diff, vx, vy,
                                   state.omega[i] if state.omega.ndim > 0 else state.omega,
                                   curv, curv_lh]))

        feats = jnp.stack(feats)

        if self.include_other_cars and self.num_cars > 1:
            # Multi-agent: include relative position to car in front
            self_feat = feats[self.agent_idx]

            # Find car in front (closest with higher s value, wrapping around track)
            def wrap_diff(a, b):
                d = a - b
                d = jnp.where(d < -self.track_L / 2.0, d + self.track_L, d)
                d = jnp.where(d > self.track_L / 2.0, d - self.track_L, d)
                return d

            # Calculate distances to other cars
            other_indices = jnp.array([i for i in range(self.num_cars) if i != self.agent_idx])

            def get_front_idx():
                # Find car closest ahead
                dists = jax.vmap(lambda i: jnp.abs(wrap_diff(feats[i, 0], self_feat[0])))(other_indices)
                closest_idx = other_indices[jnp.argmin(dists)]
                return closest_idx

            front_idx = get_front_idx()
            front_feat = feats[front_idx]

            # Build observation: relative to front car + self info
            obs_for_pi = jnp.array([
                front_feat[0] - self_feat[0],  # front_s_diff
                front_feat[1],                  # front_e
                self_feat[1],                   # self_e
                front_feat[2],                  # front_theta_diff
                front_feat[3],                  # front_vx
                front_feat[4],                  # front_vy
                front_feat[5],                  # front_omega
                self_feat[2],                   # self_theta_diff
                self_feat[3],                   # self_vx
                self_feat[4],                   # self_vy
                self_feat[5],                   # self_omega
                front_feat[6],                  # front_curv
                self_feat[6],                   # self_curv
                front_feat[7],                  # front_curv_lh
                self_feat[7],                   # self_curv_lh
            ])
        else:
            # Single agent: only self info
            self_feat = feats[self.agent_idx]
            obs_for_pi = jnp.array([
                self_feat[1],   # self_e
                self_feat[2],   # self_theta_diff
                self_feat[3],   # self_vx
                self_feat[4],   # self_vy
                self_feat[5],   # self_omega
                self_feat[6],   # self_curv
                self_feat[7],   # self_curv_lh
            ])

        return obs_for_pi, context

    def action_pi2env(self, action: Array, context: PyTree) -> tuple[Array, PyTree]:
        """
        Transform action from policy to environment format.
        Clips throttle to [0, 1] and steering to [-1, 1].
        """
        # Ensure action is [throttle, steering]
        if action.ndim == 0:
            action = jnp.array([action, 0.0])
        elif action.shape[0] == 1:
            action = jnp.array([action[0], 0.0])

        # Clip to valid ranges
        clipped_action = jnp.array([
            jnp.clip(action[0], 0.0, 1.0),  # throttle
            jnp.clip(action[1], -1.0, 1.0),  # steering
        ])

        return clipped_action, context
