# Included Data Files

This document describes the data files included with the isolated jit_neppo module.

## Overview

The module includes example data for the **Berlin 2018 track**, which allows you to test the module immediately without needing to provide your own track data.

## Files Included

### 1. params-num.yaml

**Location:** `data/params-num.yaml`

**Source:** `simulators/params-num.yaml` from the main repository

**Description:** Track configuration file for the Berlin 2018 track

**Key Configuration:**
```yaml
track_info:
  centerline_file: "berlin_2018.csv"
  scale: 2.0
  ox: 0.0
  oy: 0.0
  # ... other track parameters
```

**Parameters:**
- **centerline_file**: Name of the reference trajectory CSV file (without `_with_speeds` suffix)
- **scale**: Track scale factor (2.0 = 2x size)
- **ox, oy**: Track offset coordinates
- **track_width**: Width of the track (1.0 meters)
- **track_texture**: Surface type ("road", "sand", "mud", "grass")
- **vehicle_params**: Vehicle dynamics parameters

### 2. berlin_2018_with_speeds.csv

**Location:** `data/ref_trajs/berlin_2018_with_speeds.csv`

**Source:** `ref_trajs/berlin_2018_with_speeds.csv` from the main repository

**Description:** Reference trajectory for the Berlin 2018 Formula E track

**Format:**
```
# s_m, x_m, y_m, vx_mps
0.0, 6.9785312, 0.1571242, 5.0
0.0299954, 6.9988495, 0.1791899, 5.0
...
```

**Columns:**
1. **s_m**: Arc length along the track (meters) - cumulative distance from start
2. **x_m**: X coordinate in track frame (meters)
3. **y_m**: Y coordinate in track frame (meters)
4. **vx_mps**: Reference velocity at this point (meters per second)

**Track Statistics:**
- **Total Length**: ~75 meters (at scale=1.0, or ~150m at scale=2.0)
- **Number of waypoints**: ~2503 points
- **Sampling rate**: ~3cm between points
- **Reference velocity**: 5.0 m/s (constant in this example)

## Track Information

### Berlin 2018 Formula E Circuit

The Berlin 2018 track is based on the Formula E circuit used at the Berlin ePrix. This is a scaled-down version suitable for 1/10th scale racing vehicles.

**Track Features:**
- Multiple turns and chicanes
- Mix of tight and flowing corners
- Suitable for testing racing algorithms
- Well-suited for multi-agent racing scenarios

## Data Verification

To verify the data files are properly configured, run:

```bash
python3 verify_data.py
```

This will check:
- ✓ Data files exist in correct locations
- ✓ YAML file has required fields
- ✓ CSV file has correct format (4 columns)
- ✓ Referenced files match between YAML and filesystem
- ✓ Data can be loaded by jit_neppo functions

## Using Your Own Data

To replace the example data with your own track:

### Option 1: Replace Files

1. Replace `data/params-num.yaml` with your track configuration
2. Place your trajectory CSV in `data/ref_trajs/`
3. Update the `centerline_file` field in your YAML to match your CSV name

### Option 2: Use Custom Paths

Specify custom paths when building the environment:

```python
from jit_neppo import build_step_and_reset

reset_fn, step_fn = build_step_and_reset(
    num_envs=1,
    params_yaml_path='/path/to/your/params.yaml',
    ref_trajs_dir='/path/to/your/ref_trajs'
)
```

## Creating Your Own Track Data

### YAML File Requirements

Your YAML file must include:

```yaml
track_info:
  centerline_file: "your_track.csv"  # Without _with_speeds suffix
  ox: 0.0                            # X offset
  oy: 0.0                            # Y offset
  scale: 1.0                         # Scale factor
```

### CSV File Requirements

Your CSV file must:
1. Be named `<centerline_file>_with_speeds.csv`
2. Have exactly 4 columns: `s, x, y, v`
3. First row can be a comment (starting with `#`) with column names
4. Values should be in SI units (meters, m/s)
5. Arc length `s` should be cumulative and increasing
6. The track should form a closed loop (last point near first point)

**Example CSV:**
```csv
# s_m, x_m, y_m, vx_mps
0.0, 0.0, 0.0, 5.0
0.1, 0.1, 0.0, 5.0
0.2, 0.2, 0.0, 5.0
...
100.0, 0.0, 0.0, 5.0
```

## Tips for Track Design

1. **Sampling Rate**: Use 2-5cm between waypoints for smooth trajectories
2. **Velocity Profile**: Can be constant or vary based on curvature
3. **Track Length**: 50-200 meters works well for 1/10 scale vehicles
4. **Closed Loop**: Ensure the track forms a proper loop (first ≈ last point)
5. **Smooth Transitions**: Avoid sharp discontinuities in position or velocity

## Data Sources

The included data originates from:
- **Repository**: peg_racer
- **Branch**: sb/neppo_env
- **Files**:
  - `simulators/params-num.yaml`
  - `ref_trajs/berlin_2018_with_speeds.csv`

## License

The data files are part of the peg_racer repository. Refer to the main repository for license information.

## Additional Tracks Available

The main repository includes several other tracks you can use:
- `berlin_2018.csv` - Original Berlin track
- `berlin_2018-large.csv` - Larger version
- `berlin_2018_raceline.csv` - Optimized racing line
- `modena_2019.csv` - Modena Formula E circuit
- `rounded_rectangle.csv` - Simple test track
- `traj_rr.csv` - Another test track

To use a different track, copy it to `data/ref_trajs/` and update `params-num.yaml` accordingly.
