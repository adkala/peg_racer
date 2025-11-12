# JIT NEPPO - Isolated Module

This directory contains an isolated, standalone version of the JIT NEPPO (JAX-based racing environment) module, extracted from the main peg_racer repository.

## Files

- **jit_neppo.py** - Main module containing the racing environment dynamics and functions
- **jax_waypoint.py** - Waypoint generation and trajectory following utilities
- **data/** - Directory for configuration files and reference trajectories (you need to populate this)

## Dependencies

The isolated module requires only standard Python packages:
- `jax` - For JIT compilation and array operations
- `jax.numpy` - JAX's numpy interface
- `numpy` - Standard numerical operations
- `pandas` - For reading CSV trajectory files
- `pyyaml` - For reading YAML configuration files

Install dependencies:
```bash
pip install jax jaxlib numpy pandas pyyaml
```

## Usage

### Basic Usage

```python
from jit_neppo import build_step_and_reset
import jax

# Build the environment functions (with default paths)
reset_fn, step_fn = build_step_and_reset(num_envs=1)

# Initialize the environment
key = jax.random.PRNGKey(0)
state, obs = reset_fn(key)

# Take a step
action = jnp.array([[0.5, 0.0]])  # [throttle, steering] for 1 env
state, next_obs, reward, done, truncated, info = step_fn(state, action)
```

### Custom Data Paths

If you have your own configuration files and trajectories:

```python
from jit_neppo import build_step_and_reset

reset_fn, step_fn = build_step_and_reset(
    num_envs=1,
    params_yaml_path='/path/to/your/params.yaml',
    ref_trajs_dir='/path/to/your/ref_trajs'
)
```

## Data Structure

The module expects the following data structure:

```
jit_neppo_isolated/
├── data/
│   ├── params-num.yaml          # Track configuration
│   └── ref_trajs/
│       └── <centerline>_with_speeds.csv  # Reference trajectory
```

### YAML Configuration Format

The `params-num.yaml` file should have the following structure:

```yaml
track_info:
  centerline_file: "centerline_name.csv"  # Without "_with_speeds" suffix
  ox: 0.0  # X offset
  oy: 0.0  # Y offset
  scale: 1.0  # Scale factor
```

### CSV Trajectory Format

The trajectory CSV file should have 4 columns: `[s, x, y, v]`
- `s`: Arc length along the track
- `x`: X coordinate
- `y`: Y coordinate
- `v`: Reference velocity

## Key Functions

### `build_step_and_reset(num_envs, params_yaml_path=None, ref_trajs_dir=None)`

Builds JIT-compiled reset and step functions for the racing environment.

**Parameters:**
- `num_envs`: Number of parallel environments
- `params_yaml_path`: Path to params YAML file (default: `./data/params-num.yaml`)
- `ref_trajs_dir`: Directory with reference trajectories (default: `./data/ref_trajs`)

**Returns:**
- `reset_jit`: JIT-compiled reset function
- `step_jit`: JIT-compiled step function

### Environment State

The environment manages multiple cars (3 by default) with the following state:
- `x, y`: Position
- `psi`: Heading angle
- `vx, vy`: Velocities
- `omega`: Angular velocity

### Actions

Actions are `[throttle, steering]` pairs where:
- `throttle`: [0, 1] - Forward throttle
- `steering`: [-1, 1] - Steering angle

### Observations

15-dimensional observation vector per car:
1. Relative arc length to front car
2. Front car's lateral error
3. Self lateral error
4. Front car's heading error
5-7. Front car's velocities (vx, vy, omega)
8. Self heading error
9-11. Self velocities (vx, vy, omega)
12. Front car's path curvature
13. Self path curvature
14. Front car's lookahead curvature
15. Self lookahead curvature

### Rewards

Reward is the change in relative arc length position compared to the maximum of the other two cars.

## Changes from Original

The following changes were made to isolate this module:

1. **Import changes**: Changed `from car_dynamics.controllers_jax.jax_waypoint` to `from jax_waypoint`
2. **Path handling**: Made all file paths relative or configurable through function parameters
3. **Documentation**: Added docstrings and this README

## Original Source

- Original location: `car_ros2/car_ros2/rl_env/jit_neppo.py`
- Dependency: `car_dynamics/car_dynamics/controllers_jax/jax_waypoint.py`

## License

Refer to the main peg_racer repository for license information.
