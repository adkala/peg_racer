# MPC-Inspired Reward Function Documentation

## Overview

This document describes the single-agent reward function designed for training autonomous racing agents using reinforcement learning with the perl_jax PPO implementation. The reward function is inspired by Model Predictive Control (MPC) cost functions used in optimal control.

## Motivation

Traditional MPC for autonomous racing uses a quadratic cost function with three main components:

```
J = cost_tracking + cost_actuation + cost_violation
```

Where:
- **cost_tracking**: Penalizes deviation from reference trajectory (position, velocity, heading)
- **cost_actuation**: Penalizes large changes in control inputs (steering, throttle)
- **cost_violation**: Heavily penalizes constraint violations (track boundaries, speed limits)

Our reward function mirrors this structure but is inverted for reinforcement learning (higher rewards are better):

```
R = reward_tracking - penalty_actuation - penalty_violation
```

## Architecture

### Files

1. **`mpc_reward.py`**: Core reward computation module
   - Modular reward components
   - Configurable weights
   - Pre-defined configurations (default, aggressive, conservative)

2. **`single_agent_env.py`**: Environment integration
   - JAX-based racing environment
   - Dynamic bicycle model physics
   - MPC reward integration

3. **`test_mpc_reward.py`**: Testing and validation
   - Unit tests for reward components
   - Configuration comparison
   - Reward breakdown analysis

## Reward Components

### 1. Progress Reward (Primary Objective)

**Analogous to**: MPC position tracking cost (Q matrix)

```python
progress_reward = weight * (s_current - s_previous)
```

- **Weight**: `progress_weight` (default: 10.0)
- **Purpose**: Encourages forward movement along the track
- **MPC Equivalent**: `(x - x_ref)^T @ Q @ (x - x_ref)` for position tracking

**Tuning**:
- Higher weight → prioritize speed
- Lower weight → prioritize accuracy

### 2. Cross-Track Error Penalty

**Analogous to**: MPC lateral position tracking cost (Q matrix)

```python
cross_track_penalty = -weight * e^2
```

- **Weight**: `cross_track_weight` (default: 1.0)
- **Purpose**: Penalizes deviation from track centerline
- **MPC Equivalent**: Lateral position error in Q matrix

**Tuning**:
- Higher weight → stay closer to centerline
- Lower weight → allow more deviation for speed

### 3. Velocity Reward

**Analogous to**: MPC velocity tracking cost

```python
velocity_reward = weight * max(0, tolerance - |vx - target_velocity|)
```

- **Weight**: `velocity_weight` (default: 0.1)
- **Target**: `target_velocity` (default: 2.0 m/s)
- **Purpose**: Encourages maintaining target speed
- **MPC Equivalent**: Velocity reference tracking

**Tuning**:
- Higher target_velocity → faster racing
- Higher weight → strict velocity tracking

### 4. Heading Alignment Reward

**Analogous to**: MPC orientation tracking cost (Q matrix)

```python
heading_reward = weight * cos(theta_diff)
```

- **Weight**: `heading_weight` (default: 0.5)
- **Purpose**: Rewards alignment with track direction
- **MPC Equivalent**: `(psi - psi_ref)^2` orientation cost

**Tuning**:
- Higher weight → stricter heading alignment
- Lower weight → allow more drift/sliding

### 5. Control Smoothness Penalty

**Analogous to**: MPC actuation cost (R matrix)

```python
control_penalty = -weight * sum(w_i * (u_i - u_i-1)^2)
```

- **Weight**: `control_smooth_weight` (default: 0.01)
- **Individual weights**: `[0.005, 1.0]` for `[throttle_rate, steering_rate]`
- **Purpose**: Penalizes jerky control inputs
- **MPC Equivalent**: `(u_t - u_t-1)^T @ R @ (u_t - u_t-1)`

**Tuning**:
- Higher weight → smoother, more conservative driving
- Lower weight → allow aggressive maneuvers

### 6. Safety Constraint Penalties

**Analogous to**: MPC soft constraint violations (slack variables with weight ~1e6)

```python
off_track_penalty = -penalty if |e| > threshold else 0
velocity_penalty = -weight * max(0, 0.5 - vx)
```

- **Off-track penalty**: `off_track_penalty` (default: 10.0)
- **Off-track threshold**: `off_track_threshold` (default: 2.0 m)
- **Purpose**: Heavy penalties for constraint violations
- **MPC Equivalent**: Slack variable penalties for soft constraints

**Tuning**:
- Higher penalty → avoid crashes at all costs
- Higher threshold → more tolerance before penalty

## Configuration Profiles

### Default Configuration

Balanced configuration for general racing:

```python
MPCRewardConfig(
    progress_weight=10.0,
    cross_track_weight=1.0,
    velocity_weight=0.1,
    heading_weight=0.5,
    control_smooth_weight=0.01,
    off_track_penalty=10.0,
    off_track_threshold=2.0,
    target_velocity=2.0,
)
```

**Use case**: General training, balanced speed and precision

### Aggressive Configuration

Emphasizes speed over accuracy:

```python
MPCRewardConfig(
    progress_weight=15.0,        # Higher priority on speed
    cross_track_weight=0.5,      # Less penalty for deviation
    velocity_weight=0.3,         # Reward higher speeds
    heading_weight=0.3,          # Less strict on alignment
    control_smooth_weight=0.005, # Allow aggressive maneuvers
    target_velocity=3.0,         # Higher target speed
)
```

**Use case**: Time trials, maximum speed scenarios

### Conservative Configuration

Emphasizes accuracy and safety:

```python
MPCRewardConfig(
    progress_weight=5.0,         # Lower priority on speed
    cross_track_weight=2.0,      # Higher penalty for deviation
    velocity_weight=0.05,        # Less emphasis on speed
    heading_weight=1.0,          # Stricter alignment
    control_smooth_weight=0.02,  # Smoother control required
    target_velocity=1.5,         # Lower target speed
)
```

**Use case**: Learning phase, obstacle avoidance, safety-critical scenarios

## Integration with perl_jax Training

### Usage Example

```python
from single_agent_env import build_single_agent_env
from mpc_reward import create_default_reward_config, MPCRewardConfig

# Option 1: Use default configuration
reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)

# Option 2: Use pre-defined configuration
reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(
    num_envs=1,
    reward_config=create_aggressive_config()
)

# Option 3: Custom configuration
custom_config = MPCRewardConfig(
    progress_weight=12.0,
    cross_track_weight=1.5,
    velocity_weight=0.2,
    # ... other parameters
)
reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(
    num_envs=1,
    reward_config=custom_config
)
```

### Training Script Integration

The reward function is fully compatible with `train_with_perl_ppo.py`:

```python
# In train_with_perl_ppo.py
from single_agent_env import build_single_agent_env
from mpc_reward import create_default_reward_config

# Build environment with MPC reward
reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(
    num_envs=1,
    reward_config=create_default_reward_config()
)

# Train with perl_jax PPO (rest of training script unchanged)
```

## Comparison with Multi-Agent Reward

### Multi-Agent (jit_neppo.py)

The multi-agent reward in `jit_neppo.py` uses **relative progress**:

```python
# Compute relative position vs other agents
rel_after = relative_progress_vs_competitors(feats_after)
reward = wrap_diff(rel_after, state.last_rel, track_L)
```

This encourages **racing** behavior (overtaking, blocking).

### Single-Agent (mpc_reward.py)

Our single-agent reward uses **absolute progress and tracking quality**:

```python
reward = (
    progress_reward +        # Absolute progress along track
    cross_track_penalty +    # Stay on centerline
    velocity_reward +        # Maintain speed
    heading_reward +         # Align with track
    control_penalty +        # Smooth control
    safety_penalties         # Avoid crashes
)
```

This encourages **time trial** behavior (optimal lap times, precision).

## Relationship to MPC Cost Function

### MPC Formulation (NMPC)

From `car_dynamics/bayes_race/mpc/nmpc.py`:

```python
# MPC cost function (minimize)
cost = 0

# Terminal cost (P matrix)
cost += (x[:,-1] - xref[:,-1]).T @ P @ (x[:,-1] - xref[:,-1])

# Running cost (Q matrix)
for t in range(horizon):
    cost += (x[:,t] - xref[:,t]).T @ Q @ (x[:,t] - xref[:,t])

# Actuation cost (R matrix)
for t in range(horizon):
    delta_u = u[:,t] - u[:,t-1]
    cost += delta_u.T @ R @ delta_u

# Soft constraint violations
if track_cons:
    cost += 1e6 * sum(eps[:,t].T @ eps[:,t])
```

### Our RL Reward (Inverted MPC Cost)

```python
# RL reward function (maximize)
reward = 0

# Tracking reward (analogous to -Q cost)
reward += progress_reward           # -position_error^2
reward += cross_track_penalty       # -lateral_error^2
reward += heading_reward            # -heading_error^2
reward += velocity_reward           # -velocity_error^2

# Actuation penalty (analogous to -R cost)
reward += control_smoothness_penalty  # -(u_t - u_t-1)^T @ R @ (u_t - u_t-1)

# Constraint penalties (analogous to -slack penalties)
reward += safety_penalties          # -1e6 * constraint_violations
```

### Matrix Weight Mapping

| MPC Matrix | RL Reward Parameter | Default Value |
|------------|---------------------|---------------|
| Q (position) | `cross_track_weight` | 1.0 |
| Q (progress) | `progress_weight` | 10.0 |
| Q (heading) | `heading_weight` | 0.5 |
| Q (velocity) | `velocity_weight` | 0.1 |
| R (throttle) | `control_smooth_weight * 0.005` | 0.005 |
| R (steering) | `control_smooth_weight * 1.0` | 0.01 |
| Slack penalty | `off_track_penalty` | 10.0 |

## Tuning Guidelines

### For Faster Lap Times

```python
config = MPCRewardConfig(
    progress_weight=15.0,      # ↑ Prioritize speed
    cross_track_weight=0.5,    # ↓ Allow deviation
    velocity_weight=0.3,       # ↑ Reward high speed
    target_velocity=3.0,       # ↑ Higher target
    control_smooth_weight=0.005 # ↓ Allow aggressive control
)
```

### For More Stable Driving

```python
config = MPCRewardConfig(
    progress_weight=8.0,        # ↓ Less speed focus
    cross_track_weight=2.0,     # ↑ Stay on centerline
    heading_weight=1.0,         # ↑ Strict alignment
    control_smooth_weight=0.02, # ↑ Smoother control
    off_track_penalty=20.0      # ↑ Avoid crashes
)
```

### For Learning from Scratch

```python
config = MPCRewardConfig(
    progress_weight=5.0,        # ↓ Easier to get positive reward
    cross_track_weight=1.0,     # Standard
    velocity_weight=0.2,        # ↑ Encourage any forward motion
    target_velocity=1.0,        # ↓ Lower initial target
    off_track_threshold=3.0     # ↑ More tolerance
)
```

## Testing

### Run Tests

```bash
python test_mpc_reward.py
```

### Expected Output

The test script will:
1. Test all three configurations (default, aggressive, conservative)
2. Compare reward values across configurations
3. Show detailed reward breakdowns for different actions
4. Verify MPC reward function correctness

### Sample Output

```
================================================================================
Testing MPC-Inspired Reward Function
================================================================================

[1/4] Building environment with DEFAULT reward config...
  ✓ Environment built successfully
    Observation dimension: 8
    Action dimension: 2

[2/4] Running test rollout with DEFAULT config...
  ✓ Environment reset successful
  Step 1: reward = 0.8234, progress = 0.12, cross_track = 0.003
  Step 2: reward = 0.7891, progress = 0.24, cross_track = 0.005
  Step 3: reward = 0.8102, progress = 0.35, cross_track = 0.002
  ✓ Rollout completed
    Mean reward: 0.7856
    Total reward: 7.8560
```

## Advanced Topics

### Horizon-Based Rewards (Future Work)

MPC optimizes over a receding horizon. Future work could incorporate:

```python
# Multi-step lookahead reward
reward = sum(gamma^t * reward_t for t in range(horizon))
```

This would require modifying the environment to return future predictions.

### Learned Reward Weights (Future Work)

Instead of manual tuning, weights could be learned:

```python
# Meta-learning reward weights
config = learn_reward_weights(
    objective="minimize_lap_time",
    constraints=["stay_on_track", "smooth_control"],
    training_iterations=1000
)
```

### Adaptive Rewards (Future Work)

Rewards could adapt based on agent skill level:

```python
# Curriculum learning
if agent_performance < threshold:
    config = conservative_config  # Focus on basics
else:
    config = aggressive_config    # Push for speed
```

## References

### MPC Implementation Files

- `car_dynamics/bayes_race/mpc/nmpc.py` - Nonlinear MPC with dynamic bicycle model
- `car_dynamics/bayes_race/mpc/gpmpc.py` - GP-augmented MPC
- `car_dynamics/controllers_jax/mppi.py` - MPPI controller with reward function

### Training Scripts

- `train_with_perl_ppo.py` - Single-agent training with perl_jax PPO
- `train_single_agent.py` - Alternative PPO implementation

### Environment Files

- `single_agent_env.py` - Single-agent JAX environment
- `jit_neppo.py` - Multi-agent JAX environment (for comparison)

## Summary

The MPC-inspired reward function provides a principled, tunable approach to training autonomous racing agents. By mirroring the structure of MPC cost functions, it:

1. **Leverages domain knowledge** from optimal control theory
2. **Provides interpretable tuning parameters** with clear physical meanings
3. **Supports multiple racing styles** through configuration profiles
4. **Maintains compatibility** with perl_jax PPO training
5. **Enables transfer learning** between MPC and RL approaches

The modular design allows easy experimentation with different reward compositions while maintaining the proven structure of MPC optimization.
