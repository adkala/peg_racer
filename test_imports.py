#!/usr/bin/env python3
"""
Simple test script to verify that the isolated jit_neppo module works correctly.
This script tests that all imports are resolved and basic functionality works.
"""

import sys
import os

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")

    try:
        # Test jax_waypoint imports
        from jax_waypoint import init_waypoints, generate, WaypointSpec
        print("✓ jax_waypoint imports successful")

        # Test jit_neppo imports
        from jit_neppo import (
            build_step_and_reset,
            build_env_functions,
            DynamicParams,
            CarBatchState,
            EnvState,
            rk4_step,
            wrap_diff
        )
        print("✓ jit_neppo imports successful")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring data files."""
    print("\nTesting basic functionality...")

    try:
        import jax
        import jax.numpy as jnp
        from jax_waypoint import init_waypoints

        # Test creating a simple waypoint spec (circle trajectory)
        spec = init_waypoints(kind='circle', dt=0.1, H=9, speed=1.0)
        print("✓ Waypoint initialization works")

        # Test DynamicParams
        from jit_neppo import DynamicParams
        params = DynamicParams(num_envs=1)
        print("✓ DynamicParams initialization works")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("JIT NEPPO Isolated Module - Import Verification")
    print("=" * 60)

    # Test imports
    imports_ok = test_imports()

    # Test basic functionality
    functionality_ok = test_basic_functionality()

    print("\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("✓ All tests passed!")
        print("\nThe isolated module is working correctly.")
        print("\nNote: To use build_step_and_reset(), you'll need to:")
        print("  1. Place your params-num.yaml in ./data/")
        print("  2. Place your trajectory CSVs in ./data/ref_trajs/")
        print("\nSee README.md for more details.")
    else:
        print("✗ Some tests failed.")
        print("\nPlease check the error messages above.")
    print("=" * 60)

    return 0 if (imports_ok and functionality_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
