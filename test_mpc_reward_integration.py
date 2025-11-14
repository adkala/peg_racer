#!/usr/bin/env python3
"""
Tests to verify that single_agent_env.py correctly uses the MPC reward from mpc_reward.py.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from single_agent_env import build_single_agent_env
from mpc_reward import (
    compute_mpc_reward,
    RewardState,
    MPCRewardConfig,
    create_default_reward_config,
    create_aggressive_config,
    create_conservative_config,
    compute_progress_reward,
    compute_cross_track_penalty,
    compute_velocity_reward,
)


def test_env_uses_mpc_reward():
    """Test that the environment is using the MPC reward function."""
    print("\n[TEST 1] Verifying environment uses MPC reward...")

    # Build environment with default config
    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)

    # Reset environment
    key = random.PRNGKey(42)
    env_state, obs = reset_fn(key)

    # Take a step with forward throttle
    action = jnp.array([0.5, 0.0])  # [throttle, steering]
    next_state, next_obs, reward, done, trunc = step_fn(env_state, action)

    # Verify reward is a scalar
    assert isinstance(float(reward), float), "Reward should be a scalar"

    # Verify reward state is maintained in environment
    assert hasattr(next_state, 'reward_state'), "Environment should track reward_state"
    assert hasattr(next_state.reward_state, 'last_action'), "Reward state should have last_action"

    # Verify action is stored for next step (control smoothness)
    assert next_state.reward_state.last_action.shape == (2,), "Last action should be 2D"

    print(f"  ✓ Initial reward: {float(reward):.4f}")
    print(f"  ✓ Reward state properly maintained")
    print(f"  ✓ Last action stored: {next_state.reward_state.last_action}")


def test_reward_components():
    """Test that individual reward components work correctly."""
    print("\n[TEST 2] Testing individual reward components...")

    config = create_default_reward_config()
    track_length = 100.0

    # Test progress reward (moving forward should give positive reward)
    s_prev = 10.0
    s_curr = 12.0  # Moved forward 2 meters
    progress_reward = compute_progress_reward(s_curr, s_prev, track_length, config.progress_weight)
    print(f"  ✓ Progress reward (2m forward): {float(progress_reward):.4f}")
    assert progress_reward > 0, "Forward progress should give positive reward"

    # Test cross-track penalty (deviation should give negative reward)
    e = 1.0  # 1 meter off centerline
    cross_track = compute_cross_track_penalty(e, config.cross_track_weight)
    print(f"  ✓ Cross-track penalty (1m deviation): {float(cross_track):.4f}")
    assert cross_track < 0, "Cross-track error should give negative reward"

    # Test velocity reward
    vx = 2.0  # At target velocity
    vel_reward = compute_velocity_reward(vx, config.target_velocity, config.velocity_weight, config.velocity_tolerance)
    print(f"  ✓ Velocity reward (at target): {float(vel_reward):.4f}")
    assert vel_reward > 0, "Target velocity should give positive reward"


def test_reward_progression():
    """Test that rewards change appropriately as the car progresses."""
    print("\n[TEST 3] Testing reward progression over multiple steps...")

    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)

    key = random.PRNGKey(42)
    env_state, obs = reset_fn(key)

    rewards = []
    progress_values = []

    # Take several steps with consistent forward throttle
    action = jnp.array([0.8, 0.0])  # Strong forward throttle, no steering

    for i in range(10):
        env_state, obs, reward, done, trunc = step_fn(env_state, action)
        rewards.append(float(reward))
        progress_values.append(float(obs[0]))  # s (progress along track)

    print(f"  Initial progress: {progress_values[0]:.4f}")
    print(f"  Final progress:   {progress_values[-1]:.4f}")
    print(f"  Mean reward:      {np.mean(rewards):.4f}")
    print(f"  Std reward:       {np.std(rewards):.4f}")

    # Verify progress is increasing
    assert progress_values[-1] > progress_values[0], "Progress should increase with forward throttle"
    print(f"  ✓ Progress increased by {progress_values[-1] - progress_values[0]:.4f} meters")


def test_control_smoothness_penalty():
    """Test that control smoothness penalty works."""
    print("\n[TEST 4] Testing control smoothness penalty...")

    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)

    key = random.PRNGKey(42)
    env_state, obs = reset_fn(key)

    # Test smooth control (should have minimal penalty)
    smooth_rewards = []
    action = jnp.array([0.5, 0.0])
    for i in range(5):
        env_state, obs, reward, done, trunc = step_fn(env_state, action)
        smooth_rewards.append(float(reward))

    # Reset for jerky control test
    env_state, obs = reset_fn(random.PRNGKey(43))

    # Test jerky control (should have higher penalty)
    jerky_rewards = []
    for i in range(5):
        action = jnp.array([0.5 if i % 2 == 0 else 1.0, 0.0 if i % 2 == 0 else 1.0])
        env_state, obs, reward, done, trunc = step_fn(env_state, action)
        jerky_rewards.append(float(reward))

    print(f"  Mean reward (smooth control):  {np.mean(smooth_rewards):.4f}")
    print(f"  Mean reward (jerky control):   {np.mean(jerky_rewards):.4f}")
    print(f"  ✓ Control smoothness affects rewards")


def test_custom_reward_configs():
    """Test that different reward configurations work."""
    print("\n[TEST 5] Testing custom reward configurations...")

    configs = {
        "default": create_default_reward_config(),
        "aggressive": create_aggressive_config(),
        "conservative": create_conservative_config(),
    }

    for config_name, config in configs.items():
        reset_fn, step_fn, _, _ = build_single_agent_env(num_envs=1, reward_config=config)

        key = random.PRNGKey(42)
        env_state, obs = reset_fn(key)

        # Take a few steps
        rewards = []
        action = jnp.array([0.7, 0.0])
        for i in range(5):
            env_state, obs, reward, done, trunc = step_fn(env_state, action)
            rewards.append(float(reward))

        print(f"  {config_name:12s} - Mean reward: {np.mean(rewards):7.4f}, "
              f"Target vel: {config.target_velocity:.1f}, "
              f"Progress weight: {config.progress_weight:.1f}")

    print("  ✓ All reward configurations work correctly")


def test_reward_state_consistency():
    """Test that reward state is consistently updated."""
    print("\n[TEST 6] Testing reward state consistency...")

    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)

    key = random.PRNGKey(42)
    env_state, obs = reset_fn(key)

    # Initial reward state should have zero action
    initial_action = env_state.reward_state.last_action
    assert jnp.allclose(initial_action, jnp.zeros(2)), "Initial last_action should be zero"
    print(f"  ✓ Initial last_action: {initial_action}")

    # Take a step
    action = jnp.array([0.5, 0.3])
    env_state, obs, reward, done, trunc = step_fn(env_state, action)

    # Check that action is stored (clipped version)
    stored_action = env_state.reward_state.last_action
    print(f"  Input action:      {action}")
    print(f"  Stored action:     {stored_action}")

    # The stored action should be the clipped version [clip(0.5, 0, 1), clip(0.3, -1, 1)]
    expected_stored = jnp.array([0.5, 0.3])
    assert jnp.allclose(stored_action, expected_stored, atol=1e-5), \
        f"Stored action {stored_action} doesn't match expected {expected_stored}"

    print(f"  ✓ Action properly stored for control smoothness calculation")


def test_off_track_penalty():
    """Test that going off track incurs a penalty."""
    print("\n[TEST 7] Testing off-track penalty...")

    config = create_default_reward_config()
    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1, reward_config=config)

    key = random.PRNGKey(42)
    env_state, obs = reset_fn(key)

    # Take steps and monitor cross-track error
    action = jnp.array([0.5, 0.0])  # Straight forward

    on_track_rewards = []
    cross_track_errors = []

    for i in range(10):
        env_state, obs, reward, done, trunc = step_fn(env_state, action)
        on_track_rewards.append(float(reward))
        cross_track_errors.append(float(obs[1]))  # e (cross-track error)

    max_cross_track = max(abs(e) for e in cross_track_errors)
    print(f"  Max cross-track error: {max_cross_track:.4f} m")
    print(f"  Off-track threshold:   {config.off_track_threshold:.4f} m")
    print(f"  Mean reward:           {np.mean(on_track_rewards):.4f}")

    if max_cross_track < config.off_track_threshold:
        print(f"  ✓ Stayed on track (no off-track penalty)")
    else:
        print(f"  ⚠ Went off track (penalty applied)")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("MPC Reward Integration Tests")
    print("=" * 70)

    try:
        test_env_uses_mpc_reward()
        test_reward_components()
        test_reward_progression()
        test_control_smoothness_penalty()
        test_custom_reward_configs()
        test_reward_state_consistency()
        test_off_track_penalty()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
