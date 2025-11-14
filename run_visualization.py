#!/usr/bin/env python3
"""
Run JIT NEPPO racing environment with visualization.
Demonstrates waypoint-based controllers with different policies.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jit_neppo import build_step_and_reset
from waypoint_controller import WaypointController, AggressiveController, ConservativeController
from visualizer import RacingVisualizer


def main():
    print("=" * 70)
    print("JIT NEPPO - Racing Visualization")
    print("=" * 70)

    # Build environment
    print("\n[1/3] Building environment...")
    reset_fn, step_fn = build_step_and_reset(num_envs=1)
    print("✓ Environment built")

    # Initialize environment
    print("\n[2/3] Initializing environment...")
    key = jax.random.PRNGKey(42)
    state, obs = reset_fn(key)
    print("✓ Environment initialized")
    print(f"  - 3 cars on Berlin 2018 track")
    print(f"  - Observation shape: {obs.shape}")

    # Create controllers for each car
    print("\n[3/3] Setting up controllers...")
    controllers = [
        WaypointController(target_speed=3.5, k_steer=2.0),    # Car 1: Balanced
        AggressiveController(),                               # Car 2: Aggressive
        ConservativeController(),                             # Car 3: Conservative
    ]
    print("✓ Controllers created:")
    print("  - Car 1: Balanced (3.5 m/s target)")
    print("  - Car 2: Aggressive (4.5 m/s target)")
    print("  - Car 3: Conservative (2.5 m/s target)")

    # Create visualizer
    print("\nInitializing visualization...")
    viz = RacingVisualizer(width=1200, height=800, fps=20)
    print("✓ Visualizer ready")

    # Main simulation loop
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("Controls:")
    print("  - SPACE: Pause/Resume")
    print("  - ESC: Quit")
    print("=" * 70 + "\n")

    step = 0
    max_steps = 500
    paused = False
    running = True
    rewards_history = []

    try:
        while running and step < max_steps:
            # Handle events
            event = viz.check_quit()
            if event == 'quit':
                running = False
                break
            elif event == 'pause':
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'} at step {step}")

            if not paused:
                # Get actions from controllers
                actions = np.zeros((3, 2), dtype=np.float32)
                for i, controller in enumerate(controllers):
                    actions[i] = controller.get_action(obs, i)

                # Convert to JAX array
                actions_jax = jnp.array(actions)

                # Step environment
                state, obs, reward, done, truncated, info = step_fn(state, actions_jax)

                # Track rewards
                rewards_history.append(float(reward.mean()))

                step += 1

                # Print progress occasionally
                if step % 50 == 0:
                    avg_reward = np.mean(rewards_history[-50:])
                    print(f"Step {step:3d}/{max_steps} | "
                          f"Avg Reward: {avg_reward:+.4f} | "
                          f"Current: {[f'{float(r):.3f}' for r in reward]}")

                # Check if episode done
                if done or truncated:
                    print(f"\nEpisode finished at step {step}")
                    break

            # Render
            viz.render(state, step, reward, paused)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Summary
        print("\n" + "=" * 70)
        print("Simulation Summary")
        print("=" * 70)
        print(f"Total steps: {step}")
        if rewards_history:
            print(f"Mean reward: {np.mean(rewards_history):.4f}")
            print(f"Total reward: {np.sum(rewards_history):.4f}")
        print("\nClosing visualizer...")
        viz.close()
        print("✓ Done")


if __name__ == "__main__":
    main()
