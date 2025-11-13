"""
MPC-inspired reward function for single-agent racing.

This module provides a modular, configurable reward function based on
the cost function structure used in Model Predictive Control (MPC).

The reward is composed of multiple weighted components:
1. Progress Reward: Encourages forward movement along the track
2. Cross-Track Error Penalty: Penalizes deviation from centerline
3. Velocity Reward: Encourages maintaining target velocity
4. Heading Alignment Reward: Rewards alignment with track direction
5. Control Smoothness Penalty: Penalizes large control changes
6. Safety Penalties: Heavy penalties for constraint violations

This design mirrors the MPC cost function structure:
    J = cost_tracking + cost_actuation + cost_violation
but inverted to be a reward (higher is better).
"""

from typing import NamedTuple, Optional
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class MPCRewardConfig:
    """Configuration for MPC-inspired reward function.

    These weights correspond to the Q, R, P matrices in MPC:
    - progress_weight: Primary objective (tracking cost Q)
    - cross_track_weight: Position error penalty (tracking cost Q)
    - velocity_weight: Velocity tracking
    - heading_weight: Heading alignment (tracking cost Q)
    - control_smooth_weight: Control rate penalty (actuation cost R)
    - off_track_penalty: Constraint violation penalty (soft constraints)
    - collision_penalty: Safety constraint violation

    Typical MPC values:
    - Q = diag([1, 1]) for position tracking
    - R = diag([0.005, 1]) for [accel_rate, steer_rate]
    - P = diag([0, 0]) for terminal cost
    """

    # Primary objective: maximize progress along track
    progress_weight: float = 10.0  # Scale factor for forward progress

    # Tracking costs (analogous to MPC Q matrix)
    cross_track_weight: float = 1.0  # Penalty for lateral deviation
    velocity_weight: float = 0.1  # Reward for maintaining velocity
    heading_weight: float = 0.5  # Reward for heading alignment

    # Actuation costs (analogous to MPC R matrix)
    control_smooth_weight: float = 0.01  # Penalty for control rate changes

    # Constraint violation penalties (analogous to MPC slack variables)
    off_track_penalty: float = 10.0  # Heavy penalty for going off track
    off_track_threshold: float = 2.0  # Distance threshold (meters)
    velocity_penalty_weight: float = 0.05  # Penalty for very low velocity

    # Target values
    target_velocity: float = 2.0  # Target forward velocity (m/s)
    velocity_tolerance: float = 1.0  # Acceptable velocity range


class RewardState(NamedTuple):
    """State information needed to compute rewards across timesteps."""
    last_action: jnp.ndarray  # Previous action for smoothness penalty


def compute_progress_reward(
    s_current: float,
    s_previous: float,
    track_length: float,
    weight: float = 10.0
) -> float:
    """
    Compute reward for progress along the track.

    This corresponds to the tracking cost in MPC for following
    the reference trajectory. We reward forward progress.

    Args:
        s_current: Current progress along track (m)
        s_previous: Previous progress along track (m)
        track_length: Total track length for wraparound handling
        weight: Scaling factor for progress reward

    Returns:
        Progress reward (higher is better)
    """
    # Handle wraparound for cyclic tracks
    delta_s = s_current - s_previous
    delta_s = jnp.where(delta_s < -track_length / 2.0, delta_s + track_length, delta_s)
    delta_s = jnp.where(delta_s > track_length / 2.0, delta_s - track_length, delta_s)

    # Scale progress by weight (analogous to Q matrix in MPC)
    return weight * delta_s


def compute_cross_track_penalty(
    e: float,
    weight: float = 1.0
) -> float:
    """
    Compute penalty for cross-track error.

    This corresponds to the lateral position tracking cost in MPC.
    We want to stay near the centerline.

    Args:
        e: Cross-track error (lateral distance from centerline in meters)
        weight: Scaling factor for penalty

    Returns:
        Cross-track penalty (negative reward)
    """
    # Quadratic penalty, similar to MPC cost: e^T @ Q @ e
    return -weight * jnp.square(e)


def compute_velocity_reward(
    vx: float,
    target_velocity: float = 2.0,
    weight: float = 0.1,
    tolerance: float = 1.0
) -> float:
    """
    Compute reward for maintaining target velocity.

    This encourages the agent to maintain forward speed.

    Args:
        vx: Current forward velocity (m/s)
        target_velocity: Desired velocity (m/s)
        weight: Scaling factor for reward
        tolerance: Acceptable deviation from target

    Returns:
        Velocity reward
    """
    # Bonus for being within tolerance of target
    velocity_error = jnp.abs(vx - target_velocity)

    # Smooth reward: higher when closer to target
    reward = weight * jnp.maximum(0.0, tolerance - velocity_error)

    # Additional small bonus for any forward velocity
    forward_bonus = weight * 0.5 * jnp.clip(vx, 0.0, target_velocity)

    return reward + forward_bonus


def compute_heading_alignment_reward(
    theta_diff: float,
    weight: float = 0.5
) -> float:
    """
    Compute reward for heading alignment with track.

    This corresponds to the orientation tracking cost in MPC.

    Args:
        theta_diff: Difference between car heading and track tangent (radians)
        weight: Scaling factor for reward

    Returns:
        Heading alignment reward
    """
    # Cosine-based reward: 1.0 when aligned, -1.0 when opposite
    # This is smoother than quadratic and more intuitive
    alignment = jnp.cos(theta_diff)
    return weight * alignment


def compute_control_smoothness_penalty(
    action: jnp.ndarray,
    last_action: jnp.ndarray,
    weight: float = 0.01
) -> float:
    """
    Compute penalty for large control changes.

    This corresponds to the actuation cost in MPC (R matrix).
    We penalize the rate of change in control inputs to encourage
    smooth driving behavior.

    Args:
        action: Current action [throttle, steering]
        last_action: Previous action [throttle, steering]
        weight: Scaling factor for penalty (analogous to R matrix)

    Returns:
        Control smoothness penalty (negative reward)
    """
    # Quadratic penalty on control rate: delta_u^T @ R @ delta_u
    delta_action = action - last_action

    # Different weights for throttle vs steering (as in MPC)
    # Typically steering changes are penalized more heavily
    weights = jnp.array([0.005, 1.0])  # [throttle_weight, steering_weight]

    # Weighted quadratic penalty
    penalty = jnp.sum(weights * jnp.square(delta_action))

    return -weight * penalty


def compute_constraint_violation_penalties(
    e: float,
    vx: float,
    off_track_threshold: float = 2.0,
    off_track_penalty: float = 10.0,
    velocity_penalty_weight: float = 0.05
) -> float:
    """
    Compute penalties for constraint violations.

    This corresponds to the soft constraint violations in MPC
    (typically weighted with very high penalties like 1e6).

    Args:
        e: Cross-track error (meters)
        vx: Forward velocity (m/s)
        off_track_threshold: Distance threshold for off-track penalty
        off_track_penalty: Weight for off-track violation
        velocity_penalty_weight: Weight for low velocity penalty

    Returns:
        Total constraint violation penalty (negative reward)
    """
    # Heavy penalty for going off track (analogous to track boundary constraints)
    off_track = jnp.where(
        jnp.abs(e) > off_track_threshold,
        -off_track_penalty,
        0.0
    )

    # Penalty for very low or negative velocity (encourage forward motion)
    velocity_penalty = -velocity_penalty_weight * jnp.maximum(0.0, 0.5 - vx)

    return off_track + velocity_penalty


def compute_mpc_reward(
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
    reward_state: RewardState,
    config: MPCRewardConfig,
    track_length: float
) -> tuple[float, RewardState]:
    """
    Compute complete MPC-inspired reward.

    This function combines all reward components in a structure
    analogous to the MPC cost function:
        J = cost_tracking + cost_actuation + cost_violation

    But inverted to be a reward (higher is better):
        R = reward_tracking - penalty_actuation - penalty_violation

    Args:
        obs: Current observation [s, e, theta_diff, vx, vy, omega, curv, curv_lh]
        action: Current action [throttle, steering]
        next_obs: Next observation (after action applied)
        reward_state: State containing previous action
        config: Reward configuration
        track_length: Total track length for progress calculation

    Returns:
        (total_reward, new_reward_state)
    """
    # Extract observation features
    s_prev = obs[0]
    e_prev = obs[1]

    s_curr = next_obs[0]
    e_curr = next_obs[1]
    theta_diff = next_obs[2]
    vx = next_obs[3]

    # 1. Progress reward (primary objective - tracking)
    progress_reward = compute_progress_reward(
        s_curr, s_prev, track_length, config.progress_weight
    )

    # 2. Cross-track error penalty (tracking cost)
    cross_track_penalty = compute_cross_track_penalty(
        e_curr, config.cross_track_weight
    )

    # 3. Velocity reward
    velocity_reward = compute_velocity_reward(
        vx, config.target_velocity, config.velocity_weight, config.velocity_tolerance
    )

    # 4. Heading alignment reward (tracking cost)
    heading_reward = compute_heading_alignment_reward(
        theta_diff, config.heading_weight
    )

    # 5. Control smoothness penalty (actuation cost)
    control_penalty = compute_control_smoothness_penalty(
        action, reward_state.last_action, config.control_smooth_weight
    )

    # 6. Constraint violation penalties (soft constraints)
    constraint_penalties = compute_constraint_violation_penalties(
        e_curr, vx,
        config.off_track_threshold,
        config.off_track_penalty,
        config.velocity_penalty_weight
    )

    # Total reward (analogous to negative cost in MPC)
    total_reward = (
        progress_reward +
        cross_track_penalty +
        velocity_reward +
        heading_reward +
        control_penalty +
        constraint_penalties
    )

    # Update reward state for next step
    new_reward_state = RewardState(last_action=action)

    return total_reward, new_reward_state


def create_default_reward_config() -> MPCRewardConfig:
    """
    Create default reward configuration based on typical MPC values.

    Returns:
        Default MPCRewardConfig
    """
    return MPCRewardConfig(
        progress_weight=10.0,
        cross_track_weight=1.0,
        velocity_weight=0.1,
        heading_weight=0.5,
        control_smooth_weight=0.01,
        off_track_penalty=10.0,
        off_track_threshold=2.0,
        velocity_penalty_weight=0.05,
        target_velocity=2.0,
        velocity_tolerance=1.0
    )


def create_aggressive_config() -> MPCRewardConfig:
    """
    Create reward configuration that emphasizes speed over accuracy.

    This configuration prioritizes progress and velocity while
    being more lenient on tracking precision.

    Returns:
        Aggressive MPCRewardConfig
    """
    return MPCRewardConfig(
        progress_weight=15.0,  # Higher priority on progress
        cross_track_weight=0.5,  # Less penalty for deviation
        velocity_weight=0.3,  # More reward for speed
        heading_weight=0.3,  # Less strict on heading
        control_smooth_weight=0.005,  # Allow more aggressive control
        off_track_penalty=15.0,  # Still avoid crashes
        off_track_threshold=2.5,  # Slightly more tolerance
        velocity_penalty_weight=0.1,
        target_velocity=3.0,  # Higher target speed
        velocity_tolerance=1.5
    )


def create_conservative_config() -> MPCRewardConfig:
    """
    Create reward configuration that emphasizes accuracy and safety.

    This configuration prioritizes staying on track and smooth control
    over pure speed.

    Returns:
        Conservative MPCRewardConfig
    """
    return MPCRewardConfig(
        progress_weight=5.0,  # Lower priority on speed
        cross_track_weight=2.0,  # Higher penalty for deviation
        velocity_weight=0.05,  # Less emphasis on speed
        heading_weight=1.0,  # Stricter heading alignment
        control_smooth_weight=0.02,  # Smoother control required
        off_track_penalty=20.0,  # Very high crash penalty
        off_track_threshold=1.5,  # Less tolerance for deviation
        velocity_penalty_weight=0.02,
        target_velocity=1.5,  # Lower target speed
        velocity_tolerance=0.8
    )
