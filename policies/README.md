# Trained Policies

This directory contains trained policies from PPO training experiments.

## Directory Structure

Each experiment creates a separate folder named after the wandb run name:

```
policies/
├── test-experiment/           # Custom named experiment
│   ├── policy_0000.pth       # Checkpoint at iteration 0
│   ├── policy_0005.pth       # Checkpoint at iteration 5
│   └── policy_final.pth      # Final policy
├── perl-ppo-11_12-16_59_22/  # Timestamp-based experiment
│   ├── policy_0000.pth
│   ├── policy_0004.pth
│   └── policy_final.pth
└── ...
```

## Experiment Naming

Experiments are named using:
1. **Custom name**: Specified via `--wandb-run-name` argument
2. **Default name**: Auto-generated as `perl-ppo-MM_DD-HH_MM_SS`

## Checkpoint Files

- `policy_XXXX.pth`: Checkpoints saved every N iterations (set by `--save-frequency`)
- `policy_final.pth`: Final policy saved at the end of training

## Usage Example

```bash
# Train with custom experiment name
python train_with_perl_ppo.py \
    --num-iterations 500 \
    --wandb-run-name my-experiment \
    --save-frequency 100

# This will save policies to: policies/my-experiment/
```

## Loading Policies

To load a trained policy for evaluation or continued training:

```python
import torch
from perl_jax.utils.torchrl import load_actor

policy = torch.load("policies/test-experiment/policy_final.pth")
```
