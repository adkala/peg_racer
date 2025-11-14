#!/usr/bin/env python3
"""
Create a video of the racing environment.
Can use a trained policy or a simple heuristic.
"""
import jax
import jax.numpy as jnp
from jax import random
import argparse
from single_agent_env import build_single_agent_env
from video_recorder import record_episode, save_video
import pickle
import os


def simple_controller_policy(obs):
    """
    Simple proportional controller for demonstration.

    obs: [s, e, theta_diff, vx, vy, omega, curv, curv_lh]
    """
    # Extract observation components
    s = obs[0]           # progress along track
    e = obs[1]           # cross-track error
    theta_diff = obs[2]  # heading error
    vx = obs[3]          # longitudinal velocity
    curv = obs[6]        # current curvature

    # Target speed based on curvature
    target_speed = 2.0 - 1.5 * jnp.abs(curv)
    target_speed = jnp.clip(target_speed, 0.5, 2.5)

    # Throttle control (proportional to speed error)
    speed_error = target_speed - vx
    throttle = jnp.clip(0.5 + 0.3 * speed_error, 0.0, 1.0)

    # Steering control (proportional to cross-track and heading error)
    steering = -1.5 * e - 0.8 * theta_diff
    steering = jnp.clip(steering, -1.0, 1.0)

    return jnp.array([throttle, steering])


def load_trained_policy(policy_path):
    """Load a trained policy from pickle file."""
    with open(policy_path, 'rb') as f:
        data = pickle.load(f)

    from ppo import ActorCritic

    params = data['params']
    action_dim = data['action_dim']

    # Create network
    network = ActorCritic(action_dim=action_dim)

    # Return policy function
    def policy_fn(obs):
        action_mean, _, _ = network.apply(params, obs)
        return jnp.clip(action_mean, -1.0, 1.0)

    return policy_fn


def main():
    parser = argparse.ArgumentParser(description='Create a racing video')
    parser.add_argument('--output', type=str, default='racing_demo.mp4',
                        help='Output video path')
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to trained policy pickle file (optional)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps to record')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("Racing Video Creator")
    print("=" * 70)

    # Build environment
    print("\n[1/3] Building environment...")
    try:
        reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)
        print(f"  Environment built successfully")
        print(f"  Observation dimension: {obs_dim}")
        print(f"  Action dimension: {act_dim}")
    except Exception as e:
        print(f"  Failed to build environment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load or create policy
    print("\n[2/3] Loading policy...")
    if args.policy is not None and os.path.exists(args.policy):
        print(f"  Loading trained policy from {args.policy}")
        try:
            policy_fn = load_trained_policy(args.policy)
            print("  Trained policy loaded successfully")
        except Exception as e:
            print(f"  Failed to load policy: {e}")
            print("  Falling back to simple controller")
            policy_fn = simple_controller_policy
    else:
        if args.policy is not None:
            print(f"  Policy file not found: {args.policy}")
        print("  Using simple proportional controller")
        policy_fn = simple_controller_policy

    # Record video
    print("\n[3/3] Recording episode...")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Output: {args.output}")
    print(f"  FPS: {args.fps}")

    try:
        rng = random.PRNGKey(args.seed)

        frames, total_reward, episode_length = record_episode(
            env_reset=reset_fn,
            env_step=step_fn,
            policy_fn=policy_fn,
            rng=rng,
            max_steps=args.max_steps,
            save_path=args.output,
            fps=args.fps,
        )

        print(f"\n  Episode completed!")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Frames recorded: {len(frames)}")
        print(f"  Video saved to: {args.output}")

    except Exception as e:
        print(f"\n  Failed to record video: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("SUCCESS: Video created!")
    print("=" * 70)

    # Print viewing instructions
    print("\nTo view the video:")
    print(f"  - Open {args.output} in a video player")
    print(f"  - Or use: open {args.output}  (macOS)")
    print(f"  - Or use: xdg-open {args.output}  (Linux)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
