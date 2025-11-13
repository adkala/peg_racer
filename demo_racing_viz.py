#!/usr/bin/env python3
"""
Demo script for racing environment with visualization.

This script demonstrates:
- Racing environment with 3 cars
- Real-time pygame visualization
- Optional mp4 video recording
"""
import jax
import jax.numpy as jnp
import argparse
from racing_env import build_racing_env
from visualizer import RacingVisualizer


def simple_policy(obs, step):
    """
    Simple policy: constant throttle with sinusoidal steering.

    Args:
        obs: Observation (not used in this simple policy)
        step: Current step number

    Returns:
        action: [throttle, steering]
    """
    throttle = 0.6
    steering = 0.15 * jnp.sin(step * 0.3)
    return jnp.array([throttle, steering])


def main():
    parser = argparse.ArgumentParser(description='Racing environment visualization demo')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps to run')
    parser.add_argument('--record', action='store_true', help='Record video to mp4')
    parser.add_argument('--output', type=str, default='racing_output.mp4', help='Output video path')
    parser.add_argument('--fps', type=int, default=20, help='Visualization FPS')
    args = parser.parse_args()

    print("=" * 70)
    print("car_jax Racing Environment - Visualization Demo")
    print("=" * 70)

    # Build environment
    print("\n[1/3] Building racing environment...")
    env = build_racing_env(num_cars=3, max_steps=args.steps)
    print(f"✓ Environment built")
    print(f"  - Track length: {env.track_L:.2f}m")
    print(f"  - Number of cars: {env.num_cars}")
    print(f"  - Episode length: {env.max_steps} steps")

    # Initialize visualizer
    print("\n[2/3] Initializing visualizer...")
    viz = RacingVisualizer(
        width=1200,
        height=800,
        fps=args.fps,
        record_video=args.record,
        video_path=args.output
    )
    print("✓ Visualizer ready")
    if args.record:
        print(f"  - Recording to: {args.output}")

    # Reset environment
    print("\n[3/3] Running simulation...")
    key = jax.random.PRNGKey(42)
    env_obs, obs_for_pi, agent_ctx, sim_ctx = env.reset(key)

    # Extract initial state
    state = env_obs.state

    # Simulation loop
    paused = False
    step = 0
    total_rewards = jnp.zeros(env.num_cars)

    print("Controls:")
    print("  - SPACE: Pause/Resume")
    print("  - ESC: Quit")
    print("\nStarting simulation...")

    try:
        while step < args.steps:
            # Check for user input
            event = viz.check_quit()
            if event == 'quit':
                print("\nUser quit")
                break
            elif event == 'pause':
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")

            if not paused:
                # Generate actions for all cars (simple policy)
                actions = [simple_policy(obs_for_pi[i], step) for i in range(env.num_cars)]

                # Step environment
                (env_obs, rewards, terminated, truncated,
                 obs_for_pi, agent_ctx, sim_ctx) = env.step(
                    env_obs, actions, agent_ctx, sim_ctx
                )

                state = env_obs.state
                total_rewards += rewards

                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"Step {step:3d}: Rewards = [{', '.join([f'{float(r):+.2f}' for r in rewards])}]")

                step += 1

                # Check if episode ended
                if terminated or truncated:
                    print(f"\nEpisode ended at step {step}")
                    break

            # Render visualization
            viz.render(state, step, rewards if step > 0 else jnp.zeros(env.num_cars), paused)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Summary
    print("\n" + "=" * 70)
    print("Simulation Summary")
    print("=" * 70)
    print(f"Steps completed: {step}")
    print(f"Total rewards:")
    for i in range(env.num_cars):
        print(f"  Car {i+1}: {float(total_rewards[i]):+.2f}")

    # Close visualizer (saves video if recording)
    print("\nClosing visualizer...")
    viz.close()
    print("Done!")


if __name__ == "__main__":
    main()
