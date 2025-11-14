# Single-Agent Racing Training with PPO

This guide explains how to train a single-agent racing policy using PPO with comprehensive tracking and visualization.

## Overview

The training system includes:
- **PPO Algorithm**: Proximal Policy Optimization in JAX
- **Waypoint-based Rewards**: Progress tracking along the racing line
- **WandB Integration**: Comprehensive metrics logging and visualization
- **Video Recording**: Automatic policy visualization with configurable frequency
- **Local Video Saving**: Save training videos for offline analysis

## Files

- `single_agent_env.py`: Single-agent racing environment (adapted from multi-agent jit_neppo.py)
- `ppo.py`: PPO implementation with detailed metrics tracking
- `train_single_agent.py`: Main training script with wandb and video support
- `video_recorder.py`: Video generation and recording utilities

## Quick Start

### Basic Training (No Logging)

```bash
python3 train_single_agent.py --num-iterations 500
```

### Training with WandB Logging

```bash
python3 train_single_agent.py \
  --num-iterations 500 \
  --use-wandb \
  --wandb-project "my-racing-project" \
  --wandb-run-name "baseline-run"
```

### Training with Custom Video Settings

```bash
python3 train_single_agent.py \
  --num-iterations 500 \
  --video-frequency 25 \
  --video-dir my_videos \
  --save-video-local
```

## Command Line Arguments

### Training Parameters

- `--num-iterations`: Number of training iterations (default: 500)
- `--rollout-length`: Steps per rollout, matches episode length (default: 500)
- `--batch-size`: Minibatch size for PPO (default: 64)
- `--num-epochs`: PPO epochs per iteration (default: 10)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--seed`: Random seed (default: 42)

### WandB Parameters

- `--use-wandb`: Enable wandb logging (flag)
- `--wandb-project`: WandB project name (default: "single-agent-racing")
- `--wandb-entity`: WandB entity (username or team)
- `--wandb-run-name`: Custom run name
- `--wandb-tags`: Tags for the run (space-separated)

### Video Recording Parameters

- `--video-frequency`: Record video every N iterations (default: 50)
  - **Easy to change**: Just modify this value to control recording frequency
  - Lower values = more frequent videos (e.g., 10 for every 10 iterations)
  - Higher values = less frequent videos (e.g., 100 for every 100 iterations)
- `--save-video-local`: Save videos locally (default: True)
- `--video-dir`: Directory to save videos (default: "videos")

### Save Parameters

- `--save-path`: Path to save trained policy (default: "single_agent_policy.pkl")

## Tracked Metrics

The training script tracks comprehensive metrics for debugging:

### Rollout Metrics
- `rollout/mean_reward`: Average reward per step
- `rollout/std_reward`: Standard deviation of rewards
- `rollout/min_reward`: Minimum reward
- `rollout/max_reward`: Maximum reward
- `rollout/total_reward`: Sum of all rewards in rollout

### Episode Metrics
- `episode/mean_reward`: Average episode return
- `episode/std_reward`: Standard deviation of episode returns
- `episode/min_reward`: Minimum episode return
- `episode/max_reward`: Maximum episode return
- `episode/mean_length`: Average episode length
- `episode/num_episodes`: Number of episodes completed

### Value Function Metrics
- `value/mean`: Average value estimate
- `value/std`: Standard deviation of values
- `value/min`: Minimum value
- `value/max`: Maximum value

### Advantage Metrics
- `advantage/mean`: Average advantage (should be ~0 after normalization)
- `advantage/std`: Standard deviation of advantages (should be ~1)
- `advantage/min`: Minimum advantage
- `advantage/max`: Maximum advantage

### Loss Metrics
- `loss/policy`: PPO policy loss
- `loss/value`: Value function MSE loss
- `loss/entropy`: Policy entropy (exploration measure)

### Action Metrics
- `action/mean`: Average action value
- `action/std`: Standard deviation of actions
- `action/throttle_mean`: Average throttle (0 to 1)
- `action/steering_mean`: Average steering (-1 to 1)

### Video Metrics
- `video`: Recorded episode video (uploaded to wandb)
- `video/episode_reward`: Total reward in recorded episode
- `video/episode_length`: Length of recorded episode

## Video Recording

Videos are recorded automatically during training:

1. **Frequency**: Controlled by `--video-frequency` (default: 50)
   - Videos recorded at iteration 0, 50, 100, 150, ...
   - Final iteration always includes a video

2. **Local Saving**: Enabled by default
   - Videos saved to `videos/` directory (configurable with `--video-dir`)
   - Filename format: `iteration_{iteration:04d}.mp4`
   - Example: `iteration_0000.mp4`, `iteration_0050.mp4`

3. **WandB Upload**: When `--use-wandb` is enabled
   - Videos uploaded directly to wandb dashboard
   - Viewable in the "Media" panel
   - Includes episode reward and length metadata

4. **Dependencies**:
   ```bash
   pip install imageio imageio-ffmpeg  # For local video saving
   pip install wandb  # For wandb integration
   ```

## Environment Details

### State Space
- Continuous: `(x, y, psi, vx, vy, omega)` per car
- Dynamic bicycle model with RK4 integration
- 4-step action delay buffer

### Observation Space (8-dimensional)
1. `s`: Progress along track (meters)
2. `e`: Cross-track error (meters)
3. `theta_diff`: Heading error (radians)
4. `vx`: Longitudinal velocity (m/s)
5. `vy`: Lateral velocity (m/s)
6. `omega`: Angular velocity (rad/s)
7. `curv`: Curvature at current position
8. `curv_lh`: Curvature lookahead

### Action Space (2-dimensional)
1. Throttle: [0, 1] (continuous)
2. Steering: [-1, 1] (continuous)

### Reward Function
```python
# Progress reward (scaled)
progress = wrap_diff(s, state.last_s, track_L) * 10.0

# Cross-track error penalty
cross_track_penalty = clip(abs(e), 0.0, 2.0)

# Off-track penalty
off_track = where(abs(e) > 2.0, -10.0, 0.0)

# Velocity bonus
velocity_bonus = clip(vx, 0.0, 2.0) * 0.1

# Total reward
reward = progress - cross_track_penalty + off_track + velocity_bonus
```

## Example Training Commands

### Short Test Run
```bash
python3 train_single_agent.py \
  --num-iterations 10 \
  --video-frequency 5 \
  --video-dir test_videos
```

### Full Training with WandB
```bash
python3 train_single_agent.py \
  --num-iterations 1000 \
  --rollout-length 500 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --use-wandb \
  --wandb-project "peg-racer" \
  --wandb-run-name "ppo-lr1e4" \
  --wandb-tags ppo baseline \
  --video-frequency 50 \
  --save-video-local \
  --video-dir videos/ppo-lr1e4
```

### High-Frequency Video Recording for Debugging
```bash
python3 train_single_agent.py \
  --num-iterations 100 \
  --video-frequency 10 \
  --video-dir debug_videos
```

## Debugging Tips

1. **Monitor Entropy**: Should decrease gradually but not collapse to 0
   - Too low (< 0.1): Policy may have collapsed, increase entropy coefficient
   - Too high (> 3.0): Policy not learning, may need more training

2. **Check Episode Rewards**: Should show improvement over time
   - Stagnant: Try adjusting learning rate or reward scaling
   - Decreasing: Learning rate may be too high or reward shaping issues

3. **Watch Videos**: Visual inspection is crucial
   - Car going off track: Increase cross-track penalty
   - Car too slow: Increase velocity bonus
   - Car crashing: Add collision penalties

4. **Value Function**: Should track episode returns
   - If value estimates are way off, increase `num_epochs` or reduce `batch_size`

5. **Action Statistics**:
   - Throttle should be mostly positive (0.3-0.8)
   - Steering should vary based on track curvature

## Multi-Agent Comparison

Original `jit_neppo.py` (3 agents):
- Competitive reward: Stay ahead of opponents
- 15-dimensional observations (includes opponent info)
- 3 cars spawned at different positions

Single-agent version:
- Progression reward: Complete track quickly
- 8-dimensional observations (self only)
- 1 car spawned at fixed position

## Output Files

After training:
- `single_agent_policy.pkl`: Trained policy parameters and metrics
- `videos/`: Directory containing episode videos (if enabled)
- WandB dashboard: Online metrics and videos (if `--use-wandb`)

## Loading Trained Policy

```python
import pickle
import jax.numpy as jnp
from ppo import ActorCritic

# Load policy
with open('single_agent_policy.pkl', 'rb') as f:
    data = pickle.load(f)

params = data['params']
obs_dim = data['obs_dim']
action_dim = data['action_dim']

# Create network
network = ActorCritic(action_dim=action_dim)

# Get action
def get_action(obs):
    action_mean, _, _ = network.apply(params, obs)
    return jnp.clip(action_mean, -1.0, 1.0)
```

## Requirements

```bash
pip install jax jaxlib flax optax
pip install numpy pandas pyyaml
pip install pygame  # For visualization
pip install imageio imageio-ffmpeg  # For video saving
pip install wandb  # Optional, for logging
```

## Troubleshooting

### "imageio not installed" warning
```bash
pip install imageio imageio-ffmpeg
```

### "wandb not installed" warning
```bash
pip install wandb
wandb login  # First time only
```

### Videos not playing
- Ensure you have a compatible video player
- Try VLC or convert with: `ffmpeg -i video.mp4 -vcodec h264 output.mp4`

### Out of memory errors
- Reduce `--rollout-length` (e.g., 256 or 128)
- Reduce `--batch-size` (e.g., 32 or 16)
- Reduce `--num-epochs` (e.g., 5)

### Training too slow
- Increase `--video-frequency` (e.g., 100 or 200)
- Disable local video saving (remove `--save-video-local`)
- Increase `--rollout-length` for better GPU utilization
