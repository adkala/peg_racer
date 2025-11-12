"""
Waypoint-based controller for jit_neppo racing environment.
Follows the track waypoints using simple control laws.
"""

import jax.numpy as jnp
import numpy as np


class WaypointController:
    """
    Simple waypoint-following controller that uses the track waypoints
    to generate steering and throttle commands.
    """

    def __init__(self, lookahead_distance=2.0, target_speed=3.0,
                 k_steer=2.0, k_speed=0.5):
        """
        Initialize waypoint controller.

        Args:
            lookahead_distance: Distance ahead to look for waypoint (meters)
            target_speed: Desired speed (m/s)
            k_steer: Steering gain
            k_speed: Speed gain
        """
        self.lookahead_distance = lookahead_distance
        self.target_speed = target_speed
        self.k_steer = k_steer
        self.k_speed = k_speed

    def get_action(self, obs, car_index=0):
        """
        Compute action based on observation.

        Args:
            obs: Observation array (shape: [n_cars, 15])
            car_index: Which car to control (0, 1, or 2)

        Returns:
            action: Array [throttle, steering]
        """
        # Observation structure (15 dimensions per car):
        # 0: relative arc length to front car
        # 1: front car lateral error
        # 2: self lateral error
        # 3: front car heading error
        # 4-6: front car velocities (vx, vy, omega)
        # 7: self heading error
        # 8-10: self velocities (vx, vy, omega)
        # 11: front car curvature
        # 12: self curvature
        # 13: front car lookahead curvature
        # 14: self lookahead curvature

        car_obs = obs[car_index]

        # Extract relevant features
        lateral_error = float(car_obs[2])     # Self lateral error
        heading_error = float(car_obs[7])     # Self heading error
        vx = float(car_obs[8])                # Self forward velocity
        curvature = float(car_obs[12])        # Path curvature

        # Steering control: correct for lateral and heading errors
        # Use heading error as primary signal, lateral error for damping
        steering = self.k_steer * (heading_error - 0.3 * lateral_error)

        # Add feedforward term based on curvature
        steering += curvature * 0.5

        # Clip steering to [-1, 1]
        steering = np.clip(steering, -1.0, 1.0)

        # Throttle control: maintain target speed
        speed_error = self.target_speed - vx
        throttle = 0.5 + self.k_speed * speed_error

        # Clip throttle to [0, 1]
        throttle = np.clip(throttle, 0.0, 1.0)

        return np.array([throttle, steering], dtype=np.float32)

    def get_actions_batch(self, obs):
        """
        Compute actions for all cars.

        Args:
            obs: Observation array (shape: [n_cars, 15])

        Returns:
            actions: Array [n_cars, 2] with [throttle, steering] for each car
        """
        n_cars = obs.shape[0]
        actions = np.zeros((n_cars, 2), dtype=np.float32)

        for i in range(n_cars):
            actions[i] = self.get_action(obs, i)

        return actions


class AggressiveController(WaypointController):
    """
    More aggressive variant that targets higher speeds and sharper turns.
    """

    def __init__(self):
        super().__init__(
            lookahead_distance=2.5,
            target_speed=4.5,
            k_steer=2.5,
            k_speed=0.7
        )


class ConservativeController(WaypointController):
    """
    More conservative variant that targets lower speeds and smoother turns.
    """

    def __init__(self):
        super().__init__(
            lookahead_distance=1.5,
            target_speed=2.5,
            k_steer=1.5,
            k_speed=0.3
        )
