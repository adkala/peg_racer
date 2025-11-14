# Peg Racer - JIT NEPPO Environment & Visualizer

A JAX-based racing environment with real-time visualization.

## Files

- **jit_neppo.py** - Main racing environment with JIT-compiled dynamics
- **jax_waypoint.py** - Waypoint generation and trajectory following
- **visualizer.py** - Pygame-based visualizer for the racing environment
- **waypoint_controller.py** - Waypoint-following controllers
- **run_jit_neppo.py** - Simple test runner for the environment
- **run_visualization.py** - Run environment with visualization
- **data/** - Track data (Berlin 2018 track)
  - **params-num.yaml** - Track configuration
  - **ref_trajs/berlin_2018_with_speeds.csv** - Reference trajectory

## Dependencies

```bash
pip install jax jaxlib numpy pandas pyyaml pygame
```

## Quick Start

### Run with Visualization

```bash
python run_visualization.py
```

This will start the racing environment with 3 cars on the Berlin 2018 track.

**Controls:**
- SPACE - Pause/Resume
- ESC - Quit

### Test Environment Only

```bash
python run_jit_neppo.py
```

## Usage

### Basic Environment

```python
from jit_neppo import build_step_and_reset
import jax
import jax.numpy as jnp

# Build environment
reset_fn, step_fn = build_step_and_reset(num_envs=1)

# Initialize
key = jax.random.PRNGKey(0)
state, obs = reset_fn(key)

# Step
action = jnp.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])  # [throttle, steering] for 3 cars
state, next_obs, reward, done, truncated, info = step_fn(state, action)
```

### With Visualization

```python
from jit_neppo import build_step_and_reset
from visualizer import RacingVisualizer
import jax
import jax.numpy as jnp

# Build environment
reset_fn, step_fn = build_step_and_reset(num_envs=1)

# Initialize
key = jax.random.PRNGKey(0)
state, obs = reset_fn(key)

# Create visualizer
viz = RacingVisualizer(width=1200, height=800, fps=20)

# Simulation loop
for step in range(500):
    # Get action (e.g., from controller or policy)
    action = jnp.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])

    # Step environment
    state, obs, reward, done, truncated, info = step_fn(state, action)

    # Render
    viz.render(state, step, reward)

    # Check for quit
    if viz.check_quit() == 'quit':
        break

viz.close()
```

## Environment Details

**State:** 3 cars racing on a track
- Position: (x, y)
- Heading: psi
- Velocities: (vx, vy, omega)

**Actions:** `[throttle, steering]`
- throttle: [0, 1]
- steering: [-1, 1]

**Observations:** 15-dimensional per car
1. Relative arc length to front car
2. Front car lateral error
3. Self lateral error
4. Front car heading error
5-7. Front car velocities
8. Self heading error
9-11. Self velocities
12. Front car curvature
13. Self curvature
14-15. Lookahead curvatures

**Reward:** Change in relative position along the track

**Physics:** Dynamic bicycle model with RK4 integration, 4-step action delay
