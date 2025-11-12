# Training with perl_jax PPO Implementation

This guide explains how to use the `perl_jax` PPO implementation for training the single-agent racing policy.

## Overview

The `train_with_perl_ppo.py` script uses the PPO implementation from the [perl_jax](https://github.com/peu-peu/perl_jax) package, which provides:
- **LSTM Actor**: Recurrent policy network for temporal dependencies
- **LSTM Critic**: Recurrent value function
- **TorchRL Integration**: Leverages TorchRL's PPO loss and GAE implementation
- **Comprehensive Logging**: Detailed metrics via `log_dict` returned by trainer

## Key Differences from Custom PPO

| Feature | Custom PPO (`ppo.py`) | perl_jax PPO (`train_with_perl_ppo.py`) |
|---------|----------------------|------------------------------------------|
| Actor | MLP (Flax/JAX) | LSTM (TorchRL/PyTorch) |
| Critic | MLP (Flax/JAX) | LSTM (TorchRL/PyTorch) |
| Framework | Pure JAX | JAX env + PyTorch RL |
| Memory | Stateless | LSTM hidden states |
| Metrics | Custom tracked | TorchRL built-in |

## Configuration

### PPO Hyperparameters

The config is based on `perl_jax/exp/ppo.py` CONFIG["trainer"]:

```python
PPO_CONFIG = {
    "rl": "ppo",
    "rl_kwargs": {
        "entropy_eps": 5e-3,          # Entropy coefficient
        "num_epochs": 10,              # PPO epochs per iteration
        "sub_batch_size": 500,         # Mini-batch size
        "gamma": 0.99,                 # Discount factor
        "gae_lambda": 0.95,            # GAE lambda
        "clip_epsilon": 0.2,           # PPO clip epsilon
        "max_grad_norm": 0.5,          # Gradient clipping
        "critic_coef": 0.5,            # Value loss coefficient
        "lr": 0.0003,                  # Learning rate
        "loss_critic_type": "smooth_l1",  # Critic loss type
    },
    "actor": "lstm",
    "critic": "lstm",
    "actor_kwargs": {
        "lstm_hidden_size": 64,        # LSTM hidden dimension
        "head_net_arch": [64, 32, 16], # MLP head architecture
        "head_activation_func": "relu",
        "distribution": "tanh",        # Action distribution
        "distribution_kwargs": {"tanh_loc": True},
    },
    "critic_kwargs": {
        "lstm_hidden_size": 64,
        "head_net_arch": [64, 32, 16],
        "head_activation_func": "relu",
    },
}
```

### Configurable Variables

All hyperparameters can be changed at the top of `train_with_perl_ppo.py` by modifying `PPO_CONFIG`, or via command-line arguments:

```bash
python3 train_with_perl_ppo.py \
  --lr 0.0001 \
  --num-epochs 15 \
  --gamma 0.995 \
  --entropy-eps 0.01 \
  --lstm-hidden-size 128
```

## Usage

### Basic Training

```bash
python3 train_with_perl_ppo.py --num-iterations 500
```

### Training with WandB

```bash
python3 train_with_perl_ppo.py \
  --num-iterations 1000 \
  --use-wandb \
  --wandb-project "racing-perl-ppo" \
  --wandb-entity "your-entity"
```

### Custom Hyperparameters

```bash
python3 train_with_perl_ppo.py \
  --num-iterations 500 \
  --rollout-length 1000 \
  --lr 0.0001 \
  --num-epochs 15 \
  --gamma 0.995 \
  --lstm-hidden-size 128 \
  --device cuda
```

### With Video Recording

```bash
python3 train_with_perl_ppo.py \
  --num-iterations 500 \
  --video-frequency 25 \
  --save-video-local \
  --video-dir perl_ppo_videos
```

## Command Line Arguments

### Training Parameters
- `--num-iterations INT`: Number of training iterations (default: 500)
- `--rollout-length INT`: Steps per rollout (default: 500)
- `--seed INT`: Random seed (default: 42)
- `--device STR`: Device to use (cpu/cuda, default: cpu)

### PPO Hyperparameters (Override Config)
- `--lr FLOAT`: Learning rate
- `--num-epochs INT`: PPO epochs per iteration
- `--gamma FLOAT`: Discount factor
- `--entropy-eps FLOAT`: Entropy coefficient
- `--lstm-hidden-size INT`: LSTM hidden dimension

### WandB Parameters
- `--use-wandb`: Enable wandb logging
- `--wandb-project STR`: WandB project name
- `--wandb-entity STR`: WandB entity
- `--wandb-run-name STR`: Custom run name

### Video Recording
- `--video-frequency INT`: Record video every N iterations (default: 50)
- `--save-video-local`: Save videos locally
- `--video-dir STR`: Video directory (default: "videos")

### Checkpointing
- `--save-path STR`: Policy save path (default: "perl_ppo_policy.pth")
- `--save-frequency INT`: Save every N iterations (default: 100)

## Logged Metrics

The perl_jax PPO trainer returns a `log_dict` every step with comprehensive metrics:

### Rollout Metrics
- `rollout/ep_len_mean`: Mean episode length
- `rollout/ep_len_min`: Minimum episode length
- `rollout/ep_len_max`: Maximum episode length
- `rollout/ep_rew_mean`: Mean episode return
- `rollout/step_rew_mean`: Mean step reward

### Training Metrics
- `train/loss`: Total PPO loss
- `train/loss_objective`: Policy loss (clipped)
- `train/loss_critic`: Value function loss
- `train/loss_entropy`: Entropy loss
- `train/entropy`: Policy entropy
- `train/kl_approx`: Approximate KL divergence
- `train/clip_fraction`: Fraction of clipped updates
- `train/explained_variance`: Value function explained variance
- `train/std`: Action standard deviation
- `train/lr`: Current learning rate

### Timing Metrics
- `time/collect_sec`: Rollout collection time
- `time/train_sec`: Training time per iteration
- `time/total_sec`: Total iteration time
- `time/fps`: Frames per second during collection

All metrics are automatically logged to WandB if `--use-wandb` is enabled.

## Architecture

### LSTM Actor

The LSTM actor maintains hidden states across timesteps:

```
Input (obs) → LSTM → MLP Head → Action Distribution (Tanh Normal)
              ↑                             ↓
              └──── hidden state (c, h) ────┘
```

- **LSTM Hidden Size**: 64 (configurable)
- **MLP Head**: [64, 32, 16] with ReLU
- **Distribution**: Tanh-transformed Gaussian

### LSTM Critic

The critic network:

```
Input (obs) → LSTM → MLP Head → State Value
```

- **LSTM Hidden Size**: 64 (configurable)
- **MLP Head**: [64, 32, 16] with ReLU
- **Output**: Scalar value estimate

### Hidden State Management

- **Initialization**: Hidden states reset at start of each episode
- **Reset on Done**: LSTM states automatically reset when episode terminates
- **Context Tracking**: Hidden states maintained in `policy_context`

## Integration Details

### Environment to TensorDict Conversion

The racing environment returns JAX arrays, which are converted to PyTorch TensorDicts:

```python
td = TensorDict({
    "observation": obs,
    "action": action,
    "step_count": step_count,
    "next": {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "step_count": step_count + 1,
        "episode_reward": episode_reward,
    }
}, batch_size=[rollout_length])
```

### Agent Wrapper

A simple agent wrapper provides the required specs:

```python
class SimpleAgent:
    @property
    def pi_obs_spec(self):
        return gym.spaces.Box(low=-inf, high=inf, shape=(8,))

    @property
    def pi_action_spec(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
```

### Video Recording with LSTM

Videos use a stateful policy wrapper to maintain LSTM hidden states across the episode:

```python
class StatefulPolicy:
    def __init__(self, actor, initial_context):
        self.actor = actor
        self.context = initial_context

    def __call__(self, obs):
        action, self.context = self.actor(obs, self.context, rng)
        return action
```

## Advantages of perl_jax PPO

1. **Memory**: LSTM actors can learn temporal patterns
2. **Mature Implementation**: TorchRL provides well-tested PPO
3. **Rich Metrics**: Automatic logging of KL divergence, clip fraction, etc.
4. **Flexibility**: Easy to swap actor/critic architectures
5. **Code Reuse**: Same infrastructure as perl_jax multi-agent training

## Troubleshooting

### Import Errors

If you get import errors for `perl_jax`:

```bash
# The script adds perl_jax to path automatically
# But ensure perl_jax is installed or path is correct
export PYTHONPATH="/Users/addison/dev/perl-jax:$PYTHONPATH"
```

### CUDA Out of Memory

Reduce batch sizes:

```bash
python3 train_with_perl_ppo.py \
  --rollout-length 250 \
  --sub-batch-size 250
```

Note: `sub_batch_size` must be modified in `PPO_CONFIG` at the top of the script.

### Poor Performance

1. **Increase LSTM size**: `--lstm-hidden-size 128`
2. **More epochs**: `--num-epochs 15`
3. **Adjust learning rate**: `--lr 0.0001`
4. **Longer rollouts**: `--rollout-length 1000`

### Video Recording Issues

If videos fail to record:

```bash
# Install imageio
pip install imageio imageio-ffmpeg

# Test without video first
python3 train_with_perl_ppo.py --video-frequency 10000
```

## Comparison with Custom PPO

Run both implementations and compare:

```bash
# Custom PPO (MLP)
python3 train_single_agent.py --num-iterations 500 --wandb-run-name "mlp-ppo"

# perl_jax PPO (LSTM)
python3 train_with_perl_ppo.py --num-iterations 500 --use-wandb --wandb-run-name "lstm-ppo"
```

Expected differences:
- LSTM may learn faster on tracks with temporal dependencies
- LSTM uses more memory and is slower per iteration
- MLP is simpler and faster for this task

## Requirements

```bash
# JAX for environment
pip install jax jaxlib

# PyTorch and TorchRL for PPO
pip install torch torchrl tensordict

# Utilities
pip install gymnasium numpy pandas pyyaml pygame

# Optional for video/logging
pip install imageio imageio-ffmpeg wandb
```

## File Structure

```
peg_racer/
├── train_with_perl_ppo.py        # Main training script (uses perl_jax)
├── train_single_agent.py         # Original training script (custom PPO)
├── single_agent_env.py           # JAX racing environment
├── ppo.py                        # Custom PPO implementation (not used here)
├── video_recorder.py             # Video generation utilities
└── PERL_PPO_README.md            # This file
```

## Notes

- The script does **not modify** perl_jax code
- perl_jax is imported from `/Users/addison/dev/perl-jax`
- Environment remains in JAX, only RL is in PyTorch
- Hidden states managed automatically by LSTM modules
- Compatible with all wandb features

## References

- [perl_jax Repository](https://github.com/peu-peu/perl_jax)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
