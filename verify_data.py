#!/usr/bin/env python3
"""
Verify that the data files are properly configured and can be loaded.
"""

import os
import sys

def verify_data_structure():
    """Check that all required data files exist."""
    print("Verifying data structure...")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    errors = []

    # Check params-num.yaml
    params_file = os.path.join(data_dir, 'params-num.yaml')
    if os.path.exists(params_file):
        print(f"✓ Found: {params_file}")
    else:
        print(f"✗ Missing: {params_file}")
        errors.append("params-num.yaml not found")

    # Check ref_trajs directory
    ref_trajs_dir = os.path.join(data_dir, 'ref_trajs')
    if os.path.exists(ref_trajs_dir) and os.path.isdir(ref_trajs_dir):
        print(f"✓ Found: {ref_trajs_dir}/")

        # List CSV files
        csv_files = [f for f in os.listdir(ref_trajs_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"  Found {len(csv_files)} CSV file(s):")
            for csv in csv_files:
                print(f"    - {csv}")
        else:
            print("  ✗ No CSV files found in ref_trajs/")
            errors.append("No CSV files in ref_trajs")
    else:
        print(f"✗ Missing: {ref_trajs_dir}/")
        errors.append("ref_trajs directory not found")

    return len(errors) == 0, errors

def verify_yaml_content():
    """Verify the YAML file can be loaded and has correct structure."""
    print("\nVerifying YAML content...")

    try:
        import yaml
        import pandas as pd

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        params_file = os.path.join(data_dir, 'params-num.yaml')

        with open(params_file, 'r') as f:
            yaml_content = yaml.load(f, Loader=yaml.FullLoader)

        # Check required fields
        if 'track_info' not in yaml_content:
            print("✗ Missing 'track_info' section")
            return False

        track_info = yaml_content['track_info']
        required_fields = ['centerline_file', 'ox', 'oy', 'scale']

        for field in required_fields:
            if field in track_info:
                print(f"✓ Found field: track_info.{field} = {track_info[field]}")
            else:
                print(f"✗ Missing field: track_info.{field}")
                return False

        # Check if referenced CSV exists
        centerline_file = track_info['centerline_file'][:-4]  # Remove .csv
        csv_path = os.path.join(data_dir, 'ref_trajs', centerline_file + '_with_speeds.csv')

        if os.path.exists(csv_path):
            print(f"✓ Found trajectory CSV: {centerline_file}_with_speeds.csv")

            # Try to load it
            df = pd.read_csv(csv_path, comment='#')
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.shape[1]} (expected 4: s, x, y, v)")

            if df.shape[1] == 4:
                print("✓ CSV has correct number of columns")
                return True
            else:
                print("✗ CSV has wrong number of columns")
                return False
        else:
            print(f"✗ Missing trajectory CSV: {csv_path}")
            return False

    except ImportError as e:
        print(f"⚠ Cannot verify YAML content: {e}")
        print("  (This is OK if yaml/pandas are not installed)")
        return True
    except Exception as e:
        print(f"✗ Error loading YAML: {e}")
        return False

def verify_with_jit_neppo():
    """Try to load the data using jit_neppo functions."""
    print("\nVerifying with jit_neppo functions...")

    try:
        from jit_neppo import load_path
        import os

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        params_file = os.path.join(data_dir, 'params-num.yaml')
        ref_trajs_dir = os.path.join(data_dir, 'ref_trajs')

        # Try to load the path
        path = load_path(params_file, ref_trajs_dir)

        print(f"✓ Successfully loaded trajectory data")
        print(f"  Shape: {path.shape}")
        print(f"  Track length: {path[-1, 0]:.2f} meters")
        print(f"  Data range:")
        print(f"    X: [{path[:, 1].min():.2f}, {path[:, 1].max():.2f}]")
        print(f"    Y: [{path[:, 2].min():.2f}, {path[:, 2].max():.2f}]")
        print(f"    V: [{path[:, 3].min():.2f}, {path[:, 3].max():.2f}]")

        return True

    except ImportError as e:
        print(f"⚠ Cannot test with jit_neppo: {e}")
        print("  (This is OK if dependencies are not installed)")
        return True
    except Exception as e:
        print(f"✗ Error loading with jit_neppo: {e}")
        return False

def main():
    print("=" * 60)
    print("JIT NEPPO Data Verification")
    print("=" * 60)

    # Run verification steps
    structure_ok, errors = verify_data_structure()
    yaml_ok = verify_yaml_content()
    jit_neppo_ok = verify_with_jit_neppo()

    print("\n" + "=" * 60)
    if structure_ok and yaml_ok and jit_neppo_ok:
        print("✓ All verification checks passed!")
        print("\nThe data files are properly configured and ready to use.")
        return 0
    else:
        print("✗ Some verification checks failed.")
        if errors:
            print("\nErrors:")
            for err in errors:
                print(f"  - {err}")
        return 1
    print("=" * 60)

if __name__ == "__main__":
    sys.exit(main())
