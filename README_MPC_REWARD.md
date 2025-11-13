# MPC-Inspired Single-Agent Reward Function

## Quick Start

This implementation provides a single-agent reward function for autonomous racing, inspired by Model Predictive Control (MPC) cost functions, compatible with perl_jax PPO training.

### Files

- **`mpc_reward.py`** - Core reward computation module with configurable components
- **`single_agent_env.py`** - JAX environment integration with MPC reward
- **`test_mpc_reward.py`** - Test script to validate reward function
- **`MPC_REWARD_DOCUMENTATION.md`** - Comprehensive documentation

### Usage

```python
from single_agent_env import build_single_agent_env
from mpc_reward import create_default_reward_config

# Build environment with MPC reward
reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(
    num_envs=1,
    reward_config=create_default_reward_config()
)

# Use with perl_jax training
# (compatible with train_with_perl_ppo.py)
```

### Reward Structure

The reward mirrors MPC cost function structure:

```
MPC:  J = cost_tracking + cost_actuation + cost_violation
RL:   R = reward_tracking - penalty_actuation - penalty_violation
```

**Components:**
1. **Progress Reward** - Encourages forward movement (analogous to MPC position tracking)
2. **Cross-Track Penalty** - Penalizes deviation from centerline (analogous to MPC lateral error)
3. **Velocity Reward** - Maintains target speed (analogous to MPC velocity tracking)
4. **Heading Alignment** - Rewards track alignment (analogous to MPC orientation cost)
5. **Control Smoothness** - Penalizes jerky inputs (analogous to MPC R matrix)
6. **Safety Penalties** - Heavy penalties for violations (analogous to MPC slack variables)

### Configurations

**Default** - Balanced speed and precision:
```python
create_default_reward_config()
```

**Aggressive** - Maximum speed:
```python
create_aggressive_config()
```

**Conservative** - Safety and stability:
```python
create_conservative_config()
```

### Key Differences from Multi-Agent Reward

| Aspect | Multi-Agent (jit_neppo.py) | Single-Agent (mpc_reward.py) |
|--------|---------------------------|------------------------------|
| **Objective** | Relative progress vs opponents | Absolute progress along track |
| **Behavior** | Racing (overtaking, blocking) | Time trial (optimal lap times) |
| **Reward** | `wrap_diff(rel_after, last_rel)` | Multi-component MPC-inspired |
| **Components** | 1 (relative position) | 6 (tracking, control, safety) |

### Relationship to MPC

| MPC Component | RL Reward Parameter | Default |
|---------------|---------------------|---------|
| Q matrix (position) | `cross_track_weight` | 1.0 |
| Q matrix (progress) | `progress_weight` | 10.0 |
| R matrix (throttle) | `control_smooth_weight * 0.005` | 0.005 |
| R matrix (steering) | `control_smooth_weight * 1.0` | 0.01 |
| Slack penalty | `off_track_penalty` | 10.0 |

### Testing

```bash
python test_mpc_reward.py
```

### Documentation

See **`MPC_REWARD_DOCUMENTATION.md`** for:
- Detailed component descriptions
- Tuning guidelines
- MPC theory background
- Advanced topics

## Integration with Existing MPC Research

### Comparison with Existing MPC Implementations

From `car_dynamics/bayes_race/mpc/nmpc.py`:
```python
# MPC uses:
COST_Q = np.diag([1, 1])        # Position tracking
COST_P = np.diag([0, 0])        # Terminal cost
COST_R = np.diag([0.005, 1])    # Actuation cost
```

Our RL reward uses equivalent weights:
```python
MPCRewardConfig(
    cross_track_weight=1.0,      # Maps to Q[0,0] and Q[1,1]
    progress_weight=10.0,        # Additional progress incentive
    control_smooth_weight=0.01   # Maps to R matrix
)
```

### Why This Approach?

1. **Leverages proven MPC theory** - Uses cost function structure that works in optimal control
2. **Interpretable parameters** - Weights have clear physical meanings from MPC
3. **Enables transfer learning** - Can initialize RL policies from MPC controllers
4. **Modular design** - Easy to experiment with different reward compositions
5. **Single-agent focus** - Optimizes lap times rather than competitive racing

## Next Steps

1. **Train with default config**:
   ```bash
   python train_with_perl_ppo.py --use-wandb
   ```

2. **Experiment with configurations**:
   - Try aggressive config for speed
   - Try conservative config for stability
   - Create custom configs for specific tracks

3. **Compare with MPC baselines**:
   - Run MPC controller for comparison
   - Measure lap time improvements
   - Analyze driving styles

4. **Tune for your specific task**:
   - See tuning guidelines in documentation
   - Monitor reward components during training
   - Adjust weights based on desired behavior
