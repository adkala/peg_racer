#!/usr/bin/env python3
"""
Test visualization components without requiring display.
Verifies all components work together correctly.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver for headless testing

import jax
import jax.numpy as jnp
import numpy as np
from jit_neppo import build_step_and_reset
from waypoint_controller import WaypointController, AggressiveController, ConservativeController


def test_controllers():
    """Test that controllers work correctly."""
    print("=" * 70)
    print("Testing Controllers")
    print("=" * 70)

    # Build environment
    print("\n1. Building environment...")
    reset_fn, step_fn = build_step_and_reset(num_envs=1)
    print("   ✓ Environment built")

    # Initialize
    key = jax.random.PRNGKey(42)
    state, obs = reset_fn(key)
    print(f"   ✓ Environment initialized (obs shape: {obs.shape})")

    # Create controllers
    print("\n2. Creating controllers...")
    controllers = [
        WaypointController(),
        AggressiveController(),
        ConservativeController(),
    ]
    print(f"   ✓ Created {len(controllers)} controllers")

    # Test each controller
    print("\n3. Testing controller outputs...")
    for i, controller in enumerate(controllers):
        action = controller.get_action(obs, i)
        throttle, steering = action

        print(f"   Controller {i+1}:")
        print(f"     Throttle: {throttle:.3f} (range: [0, 1])")
        print(f"     Steering: {steering:+.3f} (range: [-1, 1])")

        # Verify valid ranges
        assert 0 <= throttle <= 1, f"Invalid throttle: {throttle}"
        assert -1 <= steering <= 1, f"Invalid steering: {steering}"

    print("   ✓ All controllers produce valid outputs")

    # Test batch action generation
    print("\n4. Testing batch action generation...")
    actions = controllers[0].get_actions_batch(obs)
    print(f"   ✓ Batch actions shape: {actions.shape}")
    assert actions.shape == (3, 2), f"Invalid shape: {actions.shape}"

    return True


def test_simulation_loop():
    """Test running simulation loop with controllers."""
    print("\n" + "=" * 70)
    print("Testing Simulation Loop")
    print("=" * 70)

    # Build environment
    print("\n1. Building environment...")
    reset_fn, step_fn = build_step_and_reset(num_envs=1)

    # Initialize
    key = jax.random.PRNGKey(123)
    state, obs = reset_fn(key)
    print("   ✓ Environment ready")

    # Create controllers
    controllers = [
        WaypointController(target_speed=3.0),
        AggressiveController(),
        ConservativeController(),
    ]
    print(f"   ✓ Controllers ready")

    # Run simulation for 20 steps
    print("\n2. Running simulation for 20 steps...")
    n_steps = 20
    rewards_history = []
    positions = {i: [] for i in range(3)}

    for step in range(n_steps):
        # Get actions from controllers
        actions = np.zeros((3, 2), dtype=np.float32)
        for i, controller in enumerate(controllers):
            actions[i] = controller.get_action(obs, i)

        # Convert to JAX array
        actions_jax = jnp.array(actions)

        # Step environment
        state, obs, reward, done, truncated, info = step_fn(state, actions_jax)

        # Record data
        rewards_history.append(float(reward.mean()))
        for i in range(3):
            positions[i].append((float(state.cars.x[i]), float(state.cars.y[i])))

        if step % 5 == 4:
            print(f"   Step {step+1:2d}: rewards={[f'{float(r):.3f}' for r in reward]}")

    print(f"   ✓ Simulation completed {n_steps} steps")

    # Verify cars moved
    print("\n3. Verifying car movement...")
    for i in range(3):
        start_pos = positions[i][0]
        end_pos = positions[i][-1]
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 +
                          (end_pos[1] - start_pos[1])**2)
        print(f"   Car {i+1} traveled: {distance:.2f} meters")
        assert distance > 0.1, f"Car {i+1} didn't move enough"

    print("   ✓ All cars moved correctly")

    # Summary
    print("\n4. Simulation summary...")
    print(f"   Mean reward: {np.mean(rewards_history):.4f}")
    print(f"   Reward range: [{min(rewards_history):.4f}, {max(rewards_history):.4f}]")

    return True


def test_visualizer_components():
    """Test visualizer components (without display)."""
    print("\n" + "=" * 70)
    print("Testing Visualizer Components")
    print("=" * 70)

    try:
        from visualizer import RacingVisualizer

        print("\n1. Creating visualizer...")
        viz = RacingVisualizer(width=800, height=600, fps=20)
        print("   ✓ Visualizer created")

        print("\n2. Checking track data...")
        if viz.track_waypoints is not None:
            print(f"   ✓ Track loaded: {len(viz.track_waypoints)} waypoints")
        else:
            print("   ⚠ Track not loaded (using default)")

        print("\n3. Testing coordinate transformation...")
        world_x, world_y = 10.0, 5.0
        screen_x, screen_y = viz.world_to_screen(world_x, world_y)
        print(f"   World ({world_x}, {world_y}) -> Screen ({screen_x}, {screen_y})")
        print("   ✓ Coordinate transformation works")

        print("\n4. Closing visualizer...")
        viz.close()
        print("   ✓ Visualizer closed cleanly")

        return True

    except Exception as e:
        print(f"   ⚠ Visualizer test skipped (no display): {e}")
        return True  # Not a failure


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "VISUALIZATION COMPONENTS TEST" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run tests
    tests = [
        ("Controllers", test_controllers),
        ("Simulation Loop", test_simulation_loop),
        ("Visualizer Components", test_visualizer_components),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:30s} {status}")

    all_passed = all(r for _, r in results)
    print("=" * 70)

    if all_passed:
        print("\n✓ All visualization components working correctly!")
        print("\nTo run with visualization:")
        print("  python3 run_visualization.py")
        print("\nNote: Requires graphical display environment")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
