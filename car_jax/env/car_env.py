import jax
import jax.numpy as jnp
from jax import random
from flax import struct
from jaxtyping import Array, Key, PyTree
import gymnasium as gym
from typing import Optional

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams, CarState


@struct.dataclass
class RNGState:
    """Random number generator state"""
    policy_key: Key
    next_key: Key


@struct.dataclass
class CarEnvState:
    """Internal state for car environment"""
    # Car physical state
    x: Array
    y: Array
    psi: Array
    vx: Array
    vy: Array
    omega: Array

    # Action buffer for delay
    action_buffer: Array  # shape (num_cars, delay, 2)

    # RNG state
    rng: RNGState

    # Metrics
    agent_win: Array  # shape (num_cars,) - which agents have won
    collision: Array  # shape (num_cars,) - which agents have collided
    out_of_bounds: Array  # shape (num_cars,) - which agents are out of bounds


@struct.dataclass
class Observation:
    """Observation returned by environment"""
    state: CarEnvState
    timestep: int

    def __getitem__(self, idx):
        """Allow indexing to get observations for specific agents"""
        return Observation(
            state=CarEnvState(
                x=self.state.x[idx],
                y=self.state.y[idx],
                psi=self.state.psi[idx],
                vx=self.state.vx[idx],
                vy=self.state.vy[idx],
                omega=self.state.omega[idx],
                action_buffer=self.state.action_buffer[idx],
                rng=self.state.rng,
                agent_win=self.state.agent_win[idx] if self.state.agent_win.ndim > 0 else self.state.agent_win,
                collision=self.state.collision[idx] if self.state.collision.ndim > 0 else self.state.collision,
                out_of_bounds=self.state.out_of_bounds[idx] if self.state.out_of_bounds.ndim > 0 else self.state.out_of_bounds,
            ),
            timestep=self.timestep,
        )


class CarEnv:
    """
    JAX-based car racing environment using Dynamic Bicycle Model
    Supports single or multi-agent racing
    """

    def __init__(
        self,
        num_cars: int = 1,
        dt: float = 0.05,
        max_steps: int = 100,
        dynamics_params: Optional[DynamicParams] = None,
    ):
        self.num_cars = num_cars
        self.dt = dt
        self.max_steps = max_steps

        # Create dynamics parameters
        if dynamics_params is None:
            dynamics_params = DynamicParams(num_envs=num_cars, DT=dt)

        self.dynamics_params = dynamics_params
        self.dynamics = DynamicBicycleModel(dynamics_params)

        # Define action and observation spaces (for gym compatibility)
        import numpy as np
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0]),  # [target_vel, target_steer]
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        # Observation: [x, y, psi, vx, vy, omega] for each car
        obs_dim = 6 * num_cars
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, key: Key, initial_poses: Optional[Array] = None):
        """
        Reset environment to initial state

        Args:
            key: JAX random key
            initial_poses: Optional array of initial poses (num_cars, 3) [x, y, psi]

        Returns:
            obs: Initial observation
            info: Additional info dict
        """
        key, subkey1, subkey2 = random.split(key, 3)

        # Initialize car states
        if initial_poses is None:
            # Default: place cars at origin with random small perturbations
            x = random.normal(subkey1, (self.num_cars,)) * 0.01
            y = random.normal(subkey2, (self.num_cars,)) * 0.01
            psi = jnp.zeros(self.num_cars)
        else:
            x = initial_poses[:, 0]
            y = initial_poses[:, 1]
            psi = initial_poses[:, 2]

        vx = jnp.zeros(self.num_cars)
        vy = jnp.zeros(self.num_cars)
        omega = jnp.zeros(self.num_cars)

        # Initialize action buffer
        action_buffer = jnp.zeros((self.num_cars, self.dynamics_params.delay, 2))

        # Initialize RNG state
        key, policy_key, next_key = random.split(key, 3)
        rng = RNGState(policy_key=policy_key, next_key=next_key)

        # Initialize metrics
        agent_win = jnp.zeros(self.num_cars, dtype=jnp.bool_)
        collision = jnp.zeros(self.num_cars, dtype=jnp.bool_)
        out_of_bounds = jnp.zeros(self.num_cars, dtype=jnp.bool_)

        state = CarEnvState(
            x=x,
            y=y,
            psi=psi,
            vx=vx,
            vy=vy,
            omega=omega,
            action_buffer=action_buffer,
            rng=rng,
            agent_win=agent_win,
            collision=collision,
            out_of_bounds=out_of_bounds,
        )

        obs = Observation(state=state, timestep=0)
        info = {}

        return obs, info

    def step(self, obs: Observation, action: Array):
        """
        Step environment forward by one timestep

        Args:
            obs: Current observation
            action: Action to take (num_cars, 2) [target_vel, target_steer]

        Returns:
            next_obs: Next observation
            info: Additional info dict
        """
        state = obs.state

        # Ensure action has correct shape: (num_cars, 2)
        if action.ndim == 1:
            # Single agent action: reshape to (1, 2)
            action = action.reshape(1, -1)
        elif action.ndim == 0:
            # Scalar action: treat as single dimension
            action = jnp.array([[action, 0.0]])

        # Ensure we have 2D action [target_vel, target_steer]
        if action.shape[-1] == 1:
            # Only target_vel provided, add zero steer
            action = jnp.concatenate([action, jnp.zeros_like(action)], axis=-1)

        # Ensure action matches number of cars
        if action.shape[0] != self.num_cars:
            # Broadcast single action to all cars if needed
            if action.shape[0] == 1 and self.num_cars > 1:
                action = jnp.repeat(action, self.num_cars, axis=0)
            elif action.shape[0] != self.num_cars:
                raise ValueError(f"Action shape {action.shape} doesn't match num_cars {self.num_cars}")

        # Update action buffer (shift and add new action)
        new_buffer = jnp.concatenate([
            action[:, None, :],  # New action at front
            state.action_buffer[:, :-1, :],  # Shift old actions
        ], axis=1)

        # Get delayed action from buffer
        delayed_action = state.action_buffer[:, -1, :]  # (num_cars, 2)

        # Step dynamics
        next_x, next_y, next_psi, next_vx, next_vy, next_omega = self.dynamics.step(
            state.x,
            state.y,
            state.psi,
            state.vx,
            state.vy,
            state.omega,
            delayed_action[:, 0],  # target_vel
            delayed_action[:, 1],  # target_steer
        )

        # Update RNG
        key = state.rng.next_key
        policy_key, next_key = random.split(key, 2)
        new_rng = RNGState(policy_key=policy_key, next_key=next_key)

        # Create next state
        next_state = CarEnvState(
            x=next_x,
            y=next_y,
            psi=next_psi,
            vx=next_vx,
            vy=next_vy,
            omega=next_omega,
            action_buffer=new_buffer,
            rng=new_rng,
            agent_win=state.agent_win,  # Updated by sim
            collision=state.collision,  # Updated by sim
            out_of_bounds=state.out_of_bounds,  # Updated by sim
        )

        next_obs = Observation(state=next_state, timestep=obs.timestep + 1)
        info = {}

        return next_obs, info

    def get_obs_array(self, obs: Observation) -> Array:
        """
        Convert observation to flat array for policy input

        Returns:
            Array of shape (num_cars * 6,) with [x, y, psi, vx, vy, omega] for each car
        """
        state = obs.state
        obs_list = [state.x, state.y, state.psi, state.vx, state.vy, state.omega]
        return jnp.concatenate([o.reshape(-1) for o in obs_list])

    def get_state_dict(self, obs: Observation) -> dict:
        """
        Convert observation to dictionary for easier access
        """
        state = obs.state
        return {
            'x': state.x,
            'y': state.y,
            'psi': state.psi,
            'vx': state.vx,
            'vy': state.vy,
            'omega': state.omega,
            'timestep': obs.timestep,
        }
