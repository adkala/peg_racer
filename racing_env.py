"""
Multi-agent racing environment using car_jax framework.

This module provides a racing environment that combines:
- JAX-based car dynamics from car_jax
- Waypoint-based track following
- Multi-agent racing with relative progress rewards
"""
import jax
import jax.numpy as jnp
from jax import random
import os
from typing import Optional

from car_jax.env import CarEnv
from car_jax.sim import Sim
from car_jax.sim.waypoint_agent import WaypointAgent
from car_jax.sim.racing_reward import RelativeProgressReward
from car_dynamics.models_jax import DynamicParams
from waypoint_generator import WaypointGenerator


class RacingEnv:
    """
    Multi-agent racing environment with waypoint following.

    Features:
    - 3 cars racing on a track
    - Dynamic bicycle model physics with RK4 integration
    - Waypoint-based observations
    - Relative progress rewards
    - 4-step action delay
    """

    def __init__(
        self,
        params_yaml_path: Optional[str] = None,
        ref_trajs_dir: Optional[str] = None,
        num_cars: int = 3,
        max_steps: int = 500,
        dt: float = 0.1,
    ):
        """
        Initialize racing environment.

        Args:
            params_yaml_path: Path to params YAML (defaults to ./data/params-num.yaml)
            ref_trajs_dir: Directory with reference trajectories (defaults to ./ref_trajs)
            num_cars: Number of cars in the race (default: 3)
            max_steps: Maximum episode length (default: 500)
            dt: Simulation timestep (default: 0.1)
        """
        self.num_cars = num_cars
        self.max_steps = max_steps
        self.dt = dt

        # Create dynamic parameters matching jit_neppo settings
        self.params = DynamicParams(
            num_envs=num_cars,
            DT=dt,
            Sa=0.34,
            Sb=0.0,
            Ta=20.0,
            Tb=0.0,
            mu=0.5,
            delay=4
        )

        # Create waypoint generator
        self.wp_gen = WaypointGenerator(
            params_yaml_path=params_yaml_path,
            ref_trajs_dir=ref_trajs_dir,
            dt=dt,
            H=9,
            speed=1.0,
            scale=6.5
        )
        self.track_L = self.wp_gen.track_L

        # Create environment
        self.env = CarEnv(
            num_cars=num_cars,
            dt=dt,
            max_steps=max_steps,
            dynamics_params=self.params
        )

        # Create agents (one per car)
        self.agents = [
            WaypointAgent(
                agent_idx=i,
                waypoint_generator=self.wp_gen,
                track_L=self.track_L,
                num_cars=num_cars,
                include_other_cars=True
            )
            for i in range(num_cars)
        ]

        # Create rewards (relative progress for each car)
        self.rewards = [
            RelativeProgressReward(
                agent_idx=i,
                waypoint_generator=self.wp_gen,
                track_L=self.track_L,
                num_cars=num_cars,
                scale=1.0
            )
            for i in range(num_cars)
        ]

        # Create simulation wrapper
        self.sim = Sim(
            env=self.env,
            agents=self.agents,
            reward=self.rewards,
            max_steps=max_steps
        )

    def get_spawn_poses(self):
        """Get initial spawn poses for 3 cars (staggered start)"""
        return jnp.array(
            [
                [3.0, 5.0, -jnp.pi / 2 - 0.72],
                [0.0, 0.0, -jnp.pi / 2 - 0.50],
                [-2.0, -6.0, -jnp.pi / 2 - 0.50],
            ],
            dtype=jnp.float32,
        )

    def reset(self, key: jax.Array):
        """
        Reset the environment.

        Args:
            key: JAX random key

        Returns:
            env_obs: Environment observation
            obs_for_pi: List of observations for each policy (15-dim each)
            agent_context: List of agent contexts
            sim_context: List of reward contexts
        """
        # Reset with initial poses
        initial_poses = self.get_spawn_poses()
        key1, key2 = random.split(key)

        # Reset environment with poses
        env_obs, _ = self.env.reset(key1, initial_poses=initial_poses)

        # Reset simulation (gets policy observations and contexts)
        _, obs_for_pi, agent_context, sim_context = self.sim.reset(key2)

        return env_obs, obs_for_pi, agent_context, sim_context

    def step(self, env_obs, actions, agent_context, sim_context):
        """
        Step the environment.

        Args:
            env_obs: Current environment observation
            actions: List of actions for each policy (num_cars, 2)
            agent_context: List of agent contexts
            sim_context: List of reward contexts

        Returns:
            next_env_obs: Next environment observation
            rewards: Array of rewards (num_cars,)
            terminated: Whether episode is done (any car won)
            truncated: Whether episode is truncated (max steps)
            next_obs_for_pi: List of next observations for each policy
            next_agent_context: Updated agent contexts
            next_sim_context: Updated reward contexts
        """
        return self.sim.step(env_obs, actions, agent_context, sim_context)


def build_racing_env(
    params_yaml_path: Optional[str] = None,
    ref_trajs_dir: Optional[str] = None,
    num_cars: int = 3,
    max_steps: int = 500,
):
    """
    Build a racing environment.

    Args:
        params_yaml_path: Path to params YAML (defaults to ./data/params-num.yaml)
        ref_trajs_dir: Directory with reference trajectories (defaults to ./ref_trajs)
        num_cars: Number of cars
        max_steps: Episode length

    Returns:
        RacingEnv instance
    """
    return RacingEnv(
        params_yaml_path=params_yaml_path,
        ref_trajs_dir=ref_trajs_dir,
        num_cars=num_cars,
        max_steps=max_steps,
    )
