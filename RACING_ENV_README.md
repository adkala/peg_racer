# Racing Environment using car_jax

This implementation provides a multi-agent racing environment built using the `car_jax` framework. It replicates the functionality previously found in `jit_neppo.py` but uses the modular `car_jax` architecture.

## Overview

The racing environment features:
- **3-car racing** on the Berlin 2018 track
- **Dynamic bicycle model physics** with RK4 integration
- **Waypoint-based observations** for track following
- **Relative progress rewards** for competitive racing
- **4-step action delay** for realistic control latency

## Architecture

### Core Components

1. **waypoint_generator.py** (Learning Support Module)
   - Provides waypoint generation from track CSV files
   - Computes track progress (s) and lateral error (e)
   - Calculates track curvature for lookahead planning
   - Location: Root directory (outside `car_jax`)

2. **car_jax/sim/waypoint_agent.py** (Agent Module)
   - `WaypointAgent`: Transforms environment observations to waypoint-based features
   - Provides track-relative observations (15-dim per car)
   - Supports both single-agent and multi-agent configurations
   - Location: `car_jax/sim/`

3. **car_jax/sim/racing_reward.py** (Reward Module)
   - `RelativeProgressReward`: Rewards relative position improvement
   - `TrackProgressReward`: Rewards absolute track progress
   - `LateralErrorPenalty`: Penalizes going off-track
   - Location: `car_jax/sim/`

4. **racing_env.py** (Environment Builder)
   - `RacingEnv`: Complete multi-agent racing environment
   - Integrates car dynamics, waypoint following, and rewards
   - Provides high-level reset/step API
   - Location: Root directory

5. **test_racing_env.py** (Test Script)
   - Validates environment functionality
   - Runs simple policy to verify observations and rewards
   - Location: Root directory

## File Structure

```
peg_racer/
├── waypoint_generator.py          # Waypoint generation (learning support)
├── racing_env.py                  # Main racing environment
├── test_racing_env.py             # Test script
├── data/
│   └── params-num.yaml            # Track configuration
├── ref_trajs/                     # Track CSV files
│   └── berlin_2018_with_speeds.csv
└── car_jax/                       # Environment framework
    ├── env/
    │   └── car_env.py            # Core car environment
    └── sim/
        ├── sim.py                # Simulation wrapper
        ├── agent.py              # Base agent class
        ├── reward.py             # Base reward class
        ├── waypoint_agent.py     # Waypoint-following agent
        └── racing_reward.py      # Racing rewards
```

## Differences from jit_neppo.py

The new implementation improves on `jit_neppo.py` by:

1. **Modularity**: Separates concerns into agents, rewards, and environment
2. **Reusability**: Uses `car_jax` framework which can be extended for other tasks
3. **Clarity**: Clear separation between environment (car_jax) and learning (waypoint_generator)
4. **Flexibility**: Easy to swap agents, rewards, or add new features
5. **Maintainability**: Uses established abstractions instead of monolithic code

## Usage

### Basic Usage

```python
from racing_env import build_racing_env
import jax

# Build environment
env = build_racing_env(num_cars=3, max_steps=500)

# Reset environment
key = jax.random.PRNGKey(42)
env_obs, obs_for_pi, agent_ctx, sim_ctx = env.reset(key)

# Step environment
actions = [jnp.array([0.6, 0.1]) for _ in range(3)]  # [throttle, steering]
(env_obs, rewards, terminated, truncated,
 obs_for_pi, agent_ctx, sim_ctx) = env.step(
    env_obs, actions, agent_ctx, sim_ctx
)
```

### Running Tests

```bash
# Set PYTHONPATH
export PYTHONPATH=/home/user/peg_racer:/home/user/peg_racer/car_dynamics

# Run test
python test_racing_env.py
```

## Observation Space

Each agent receives a 15-dimensional observation vector:

```python
[
    front_s_diff,      # Longitudinal distance to car ahead
    front_e,           # Lateral error of car ahead
    self_e,            # Own lateral error
    front_theta_diff,  # Heading error of car ahead
    front_vx,          # Longitudinal velocity of car ahead
    front_vy,          # Lateral velocity of car ahead
    front_omega,       # Angular velocity of car ahead
    self_theta_diff,   # Own heading error
    self_vx,           # Own longitudinal velocity
    self_vy,           # Own lateral velocity
    self_omega,        # Own angular velocity
    front_curv,        # Track curvature ahead for front car
    self_curv,         # Own track curvature ahead
    front_curv_lh,     # Lookahead curvature for front car
    self_curv_lh,      # Own lookahead curvature
]
```

## Action Space

Each agent outputs a 2-dimensional action:
- `throttle`: [0.0, 1.0] - Acceleration command
- `steering`: [-1.0, 1.0] - Steering angle

Actions are delayed by 4 timesteps to simulate realistic control latency.

## Reward Structure

The default reward is **relative progress**:
- Measures how much further ahead the car moved compared to competitors
- Encourages racing behavior (overtaking and maintaining position)
- Computed using track progress (s-coordinate)
- Wraps around track boundaries

Alternative rewards available:
- `TrackProgressReward`: Simple forward progress
- `LateralErrorPenalty`: Penalty for going off-track
- `CompositeReward`: Weighted combination of multiple rewards

## Parameters

Key environment parameters (from `DynamicParams`):
- `DT = 0.1`: Simulation timestep (100ms)
- `delay = 4`: Action delay buffer size
- `Sa = 0.34`: Steering scale factor
- `Ta = 20.0`: Throttle acceleration scale
- `mu = 0.5`: Friction coefficient

Waypoint parameters:
- `H = 9`: Waypoint horizon (number of lookahead points)
- `scale = 6.5`: Track scale factor
- `dt = 0.1`: Waypoint time step

## Dependencies

- JAX: JIT compilation and automatic differentiation
- Flax: Struct dataclasses
- Gymnasium: Standard RL interface
- PyYAML: Configuration loading
- Pandas: CSV track data loading

## Next Steps

To use this environment for reinforcement learning:

1. **Training**: Integrate with your preferred RL library (e.g., PPO, SAC)
2. **Policies**: Replace simple actions with learned neural network policies
3. **Curriculum**: Start with single-agent, progress to multi-agent
4. **Rewards**: Tune reward weights or use composite rewards
5. **Tracks**: Add more tracks by providing new CSV files

## Comparison to jit_neppo.py

| Feature | jit_neppo.py | racing_env.py (this implementation) |
|---------|-------------|-------------------------------------|
| Physics | Custom RK4 implementation | Uses car_jax/car_dynamics modules |
| Structure | Monolithic | Modular (agents, rewards, env) |
| Observations | Hardcoded computation | Agent abstraction |
| Rewards | Hardcoded relative progress | Pluggable reward functions |
| Extensibility | Limited | High (swap components) |
| Testing | No dedicated test | test_racing_env.py |
| Dependencies | Mixed with learning code | Clear separation |

Both implementations provide identical physics and equivalent observations/rewards.
