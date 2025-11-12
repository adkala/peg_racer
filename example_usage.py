#!/usr/bin/env python3
"""
Example usage of the isolated jit_neppo module.

This script demonstrates how to use the isolated jit_neppo module
for racing environment simulation.
"""

import jax
import jax.numpy as jnp
from jit_neppo import build_step_and_reset

def example_with_default_paths():
    """
    Example using default data paths.
    Requires ./data/params-num.yaml and ./data/ref_trajs/ to be set up.
    """
    print("Example 1: Using default paths")
    print("-" * 60)

    # Build the environment functions
    # This will look for data in ./data/ by default
    try:
        reset_fn, step_fn = build_step_and_reset(num_envs=1)
        print("✓ Environment built successfully!")

        # Initialize the environment
        key = jax.random.PRNGKey(0)
        state, obs = reset_fn(key)
        print(f"✓ Environment reset, initial observation shape: {obs.shape}")

        # Take a step
        # Action: [throttle (0-1), steering (-1 to 1)]
        action = jnp.array([[0.5, 0.0]])  # 50% throttle, no steering
        state, next_obs, reward, done, truncated, info = step_fn(state, action)
        print(f"✓ Step executed, reward: {reward}")

    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")
        print("\nTo fix this:")
        print("  1. Create ./data/params-num.yaml")
        print("  2. Create ./data/ref_trajs/ directory")
        print("  3. Place your trajectory CSV files in ./data/ref_trajs/")

def example_with_custom_paths():
    """
    Example using custom data paths.
    """
    print("\nExample 2: Using custom paths")
    print("-" * 60)

    # You can specify custom paths for your data
    custom_yaml = "/path/to/your/params.yaml"
    custom_ref_trajs = "/path/to/your/ref_trajs"

    print(f"YAML path: {custom_yaml}")
    print(f"Ref trajs path: {custom_ref_trajs}")
    print("\n(This is just an example - these paths don't exist)")

    # reset_fn, step_fn = build_step_and_reset(
    #     num_envs=1,
    #     params_yaml_path=custom_yaml,
    #     ref_trajs_dir=custom_ref_trajs
    # )

def example_multi_env():
    """
    Example using multiple parallel environments.
    """
    print("\nExample 3: Multiple parallel environments")
    print("-" * 60)

    try:
        # Create 4 parallel environments
        num_envs = 4
        reset_fn, step_fn = build_step_and_reset(num_envs=num_envs)
        print(f"✓ Created {num_envs} parallel environments")

        # Initialize all environments
        key = jax.random.PRNGKey(42)
        state, obs = reset_fn(key)
        print(f"✓ Observation shape: {obs.shape} (envs x obs_dim)")

        # Take actions for all environments in parallel
        actions = jnp.array([
            [0.5, 0.1],   # Env 1: throttle right
            [0.3, -0.1],  # Env 2: throttle left
            [0.7, 0.0],   # Env 3: straight ahead
            [0.0, 0.0],   # Env 4: coast
        ])
        state, next_obs, rewards, done, truncated, info = step_fn(state, actions)
        print(f"✓ Parallel step executed")
        print(f"  Rewards: {rewards}")

    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")

def example_episode():
    """
    Example running a full episode.
    """
    print("\nExample 4: Running a full episode")
    print("-" * 60)

    try:
        reset_fn, step_fn = build_step_and_reset(num_envs=1)

        # Initialize
        key = jax.random.PRNGKey(123)
        state, obs = reset_fn(key)

        # Run episode
        episode_length = 10
        total_reward = 0.0

        print(f"Running episode for {episode_length} steps...")

        for step in range(episode_length):
            # Simple policy: constant throttle, no steering
            action = jnp.array([[0.5, 0.0]])

            # Step environment
            state, obs, reward, done, truncated, info = step_fn(state, action)
            total_reward += float(reward[0])

            if step % 5 == 0:
                print(f"  Step {step}: reward = {float(reward[0]):.4f}")

            if done or truncated:
                print(f"  Episode terminated at step {step}")
                break

        print(f"\n✓ Episode complete")
        print(f"  Total reward: {total_reward:.4f}")

    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")

def main():
    print("=" * 60)
    print("JIT NEPPO Isolated Module - Usage Examples")
    print("=" * 60)

    # Run examples
    example_with_default_paths()
    example_with_custom_paths()
    example_multi_env()
    example_episode()

    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
