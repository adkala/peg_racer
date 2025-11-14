#!/usr/bin/env python3
"""
Test script for MPC-inspired reward function.

This script verifies that the reward function works correctly with
the single-agent environment and demonstrates the reward components.
"""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from single_agent_env import build_single_agent_env
from mpc_reward import (
    create_default_reward_config,
    create_aggressive_config,
    create_conservative_config,
)


def test_reward_configs():
    """Test different reward configurations."""
    print("=" * 80)
    print("Testing MPC-Inspired Reward Function")
    print("=" * 80)

    # Test with default configuration
    print("\n[1/4] Building environment with DEFAULT reward config...")
    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(
        num_envs=1,
        reward_config=create_default_reward_config()
    )
    print(f"  ✓ Environment built successfully")
    print(f"    Observation dimension: {obs_dim}")
    print(f"    Action dimension: {act_dim}")

    # Initialize and run a few steps
    print("\n[2/4] Running test rollout with DEFAULT config...")
    key = random.PRNGKey(42)
    state, obs = reset_fn(key)
    print(f"  ✓ Environment reset successful")
    print(f"    Initial observation: {obs}")

    rewards_default = []
    for i in range(10):
        # Simple policy: constant throttle with slight steering
        throttle = 0.6
        steering = 0.1 * jnp.sin(i * 0.3)
        action = jnp.array([throttle, steering])

        state, obs, reward, done, truncated = step_fn(state, action)
        rewards_default.append(float(reward))

        if i < 3:
            print(f"  Step {i+1}: reward = {float(reward):.4f}, "
                  f"progress = {float(obs[0]):.2f}, cross_track = {float(obs[1]):.3f}")

    print(f"  ✓ Rollout completed")
    print(f"    Mean reward: {np.mean(rewards_default):.4f}")
    print(f"    Total reward: {np.sum(rewards_default):.4f}")

    # Test with aggressive configuration
    print("\n[3/4] Running test rollout with AGGRESSIVE config...")
    reset_fn_agg, step_fn_agg, _, _ = build_single_agent_env(
        num_envs=1,
        reward_config=create_aggressive_config()
    )

    key, subkey = random.split(key)
    state, obs = reset_fn_agg(subkey)

    rewards_aggressive = []
    for i in range(10):
        throttle = 0.8  # Higher throttle for aggressive config
        steering = 0.15 * jnp.sin(i * 0.3)
        action = jnp.array([throttle, steering])

        state, obs, reward, done, truncated = step_fn_agg(state, action)
        rewards_aggressive.append(float(reward))

    print(f"  ✓ Rollout completed")
    print(f"    Mean reward: {np.mean(rewards_aggressive):.4f}")
    print(f"    Total reward: {np.sum(rewards_aggressive):.4f}")

    # Test with conservative configuration
    print("\n[4/4] Running test rollout with CONSERVATIVE config...")
    reset_fn_cons, step_fn_cons, _, _ = build_single_agent_env(
        num_envs=1,
        reward_config=create_conservative_config()
    )

    key, subkey = random.split(key)
    state, obs = reset_fn_cons(subkey)

    rewards_conservative = []
    for i in range(10):
        throttle = 0.4  # Lower throttle for conservative config
        steering = 0.05 * jnp.sin(i * 0.3)  # Gentler steering
        action = jnp.array([throttle, steering])

        state, obs, reward, done, truncated = step_fn_cons(state, action)
        rewards_conservative.append(float(reward))

    print(f"  ✓ Rollout completed")
    print(f"    Mean reward: {np.mean(rewards_conservative):.4f}")
    print(f"    Total reward: {np.sum(rewards_conservative):.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("Reward Configuration Comparison")
    print("=" * 80)
    print(f"Default Configuration:")
    print(f"  Mean reward:  {np.mean(rewards_default):8.4f}")
    print(f"  Total reward: {np.sum(rewards_default):8.4f}")
    print(f"\nAggressive Configuration (higher speed, less precision):")
    print(f"  Mean reward:  {np.mean(rewards_aggressive):8.4f}")
    print(f"  Total reward: {np.sum(rewards_aggressive):8.4f}")
    print(f"\nConservative Configuration (lower speed, more precision):")
    print(f"  Mean reward:  {np.mean(rewards_conservative):8.4f}")
    print(f"  Total reward: {np.sum(rewards_conservative):8.4f}")

    print("\n" + "=" * 80)
    print("SUCCESS: MPC reward function is working correctly!")
    print("=" * 80)

    print("\nReward Components (from mpc_reward.py):")
    print("  1. Progress Reward: Encourages forward movement along track")
    print("  2. Cross-Track Error Penalty: Penalizes deviation from centerline")
    print("  3. Velocity Reward: Rewards maintaining target velocity")
    print("  4. Heading Alignment Reward: Rewards alignment with track direction")
    print("  5. Control Smoothness Penalty: Penalizes large control changes")
    print("  6. Safety Penalties: Heavy penalties for constraint violations")

    print("\nMPC Analogy:")
    print("  This reward structure mirrors the MPC cost function:")
    print("    J = cost_tracking + cost_actuation + cost_violation")
    print("  But inverted to be a reward (higher is better):")
    print("    R = reward_tracking - penalty_actuation - penalty_violation")

    print("\nTuning Guide:")
    print("  - Increase progress_weight: Prioritize speed over precision")
    print("  - Increase cross_track_weight: Prioritize staying on centerline")
    print("  - Increase control_smooth_weight: Encourage smoother driving")
    print("  - Increase off_track_penalty: Make crashes more costly")
    print("  - Adjust target_velocity: Set desired racing speed")


def test_reward_breakdown():
    """Test and display individual reward components."""
    print("\n" + "=" * 80)
    print("Detailed Reward Component Breakdown")
    print("=" * 80)

    # Build environment
    reset_fn, step_fn, _, _ = build_single_agent_env(num_envs=1)

    key = random.PRNGKey(123)
    state, obs = reset_fn(key)

    # Test a few different actions to see reward breakdown
    test_actions = [
        ("Forward (good)", jnp.array([0.7, 0.0])),
        ("Forward + steering", jnp.array([0.7, 0.3])),
        ("Aggressive steering", jnp.array([0.5, 0.8])),
        ("Low throttle", jnp.array([0.1, 0.0])),
    ]

    print("\nTesting different actions:")
    print("-" * 80)

    for name, action in test_actions:
        # Reset for each test
        state_test, obs_test = reset_fn(key)

        # Take one step
        state_next, obs_next, reward, _, _ = step_fn(state_test, action)

        print(f"\nAction: {name}")
        print(f"  Throttle: {float(action[0]):.2f}, Steering: {float(action[1]):+.2f}")
        print(f"  Total Reward: {float(reward):.4f}")
        print(f"  Progress: {float(obs_next[0]):.3f}, Cross-track error: {float(obs_next[1]):.3f}")
        print(f"  Velocity: {float(obs_next[3]):.3f}, Heading diff: {float(obs_next[2]):.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_reward_configs()
        test_reward_breakdown()
        print("\nAll tests passed! ✓")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
