#!/usr/bin/env python3
"""
Training script for single-agent racing using perl_jax PPO implementation.
Uses LSTM actor and critic with wandb integration.
"""
import sys

sys.path.insert(0, "/Users/addison/dev/perl-jax")

import jax
import jax.numpy as jnp
from jax import random
import argparse
import time
from pathlib import Path
import gymnasium as gym
import torch as th
import numpy as np

from single_agent_env import build_single_agent_env
from video_recorder import record_episode, frames_to_wandb_video
from perl_jax.rl.trainer import Trainer
from perl_jax.utils.torchrl import save_actor
from tensordict import TensorDict


# ===========================
# PPO HYPERPARAMETERS (from perl_jax/exp/ppo.py CONFIG["trainer"])
# ===========================

PPO_CONFIG = {
    "rl": "ppo",
    "rl_kwargs": {
        "entropy_eps": 5e-3,
        "num_epochs": 10,
        "sub_batch_size": 500,  # Adjusted for single-agent
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_grad_norm": 0.5,
        "critic_coef": 0.5,
        "lr": 0.0003,
        "loss_critic_type": "smooth_l1",
    },
    "actor": "lstm",
    "critic": "lstm",
    "actor_kwargs": {
        "lstm_hidden_size": 64,
        "head_net_arch": [64, 32, 16],
        "head_activation_func": "relu",
        "distribution": "tanh",
        "distribution_kwargs": {"tanh_loc": True},
    },
    "critic_kwargs": {
        "lstm_hidden_size": 64,
        "head_net_arch": [64, 32, 16],
        "head_activation_func": "relu",
    },
}


class SimpleAgent:
    """Simple agent wrapper to provide obs/action specs for perl_jax trainer."""

    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sub_idx = 0  # Required by some perl_jax code

    @property
    def pi_obs_spec(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    @property
    def pi_action_spec(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )


def collect_rollout_perl(
    env_reset,
    env_step,
    actor,
    policy_context,
    rng,
    num_steps,
    device="cpu",
):
    """
    Collect rollout and convert to TensorDict format for perl_jax PPO.

    Returns:
        TensorDict with fields required by PPO trainer
    """
    # Reset environment
    rng, reset_rng = random.split(rng)
    env_state, obs = env_reset(reset_rng)

    # Storage lists
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    truncated = []
    step_counts = []
    episode_rewards = []

    # Episode tracking
    current_step_count = 0
    current_episode_reward = 0.0

    for step in range(num_steps):
        # Get action from LSTM actor
        rng, action_rng = random.split(rng)
        action, policy_context = actor(obs, policy_context, action_rng)

        # Clip action to valid range
        action = jnp.clip(action, -1.0, 1.0)

        # Store transition (before stepping environment)
        observations.append(np.array(obs))
        actions.append(np.array(action))
        step_counts.append(current_step_count)

        # Step environment
        env_state, next_obs, reward, done, trunc = env_step(env_state, action)

        current_step_count += 1
        current_episode_reward += float(reward)

        # Store next observation
        next_observations.append(np.array(next_obs))
        rewards.append(float(reward))
        dones.append(bool(done or trunc))
        truncated.append(bool(trunc))
        episode_rewards.append(current_episode_reward)

        # Reset if done
        if done or trunc:
            rng, reset_rng = random.split(rng)
            env_state, next_obs = env_reset(reset_rng)
            # Reset LSTM hidden state
            rng, context_rng = random.split(rng)
            policy_context = actor.reset_context(policy_context, context_rng)
            current_step_count = 0
            current_episode_reward = 0.0

        obs = next_obs

    # Convert to torch tensors and create TensorDict
    # NOTE: both root and "next" need to have same keys for GAE stacking
    td = TensorDict(
        {
            "observation": th.tensor(
                np.array(observations), dtype=th.float32, device=device
            ),
            "action": th.tensor(np.array(actions), dtype=th.float32, device=device),
            "step_count": th.tensor(
                np.array(step_counts), dtype=th.int64, device=device
            ),
            "next": TensorDict(
                {
                    "observation": th.tensor(
                        np.array(next_observations), dtype=th.float32, device=device
                    ),
                    "reward": th.tensor(
                        np.array(rewards), dtype=th.float32, device=device
                    ),
                    "done": th.tensor(np.array(dones), dtype=th.bool, device=device),
                    "terminated": th.tensor(
                        np.array(dones), dtype=th.bool, device=device
                    ),
                    "truncated": th.tensor(
                        np.array(truncated), dtype=th.bool, device=device
                    ),
                    "step_count": th.tensor(
                        np.array(step_counts), dtype=th.int64, device=device
                    )
                    + 1,
                    "episode_reward": th.tensor(
                        np.array(episode_rewards), dtype=th.float32, device=device
                    ),
                },
                batch_size=[num_steps],
                device=device,
            ),
        },
        batch_size=[num_steps],
        device=device,
    )

    return td, policy_context, rng


def main():
    parser = argparse.ArgumentParser(
        description="Train single-agent racing with perl_jax PPO"
    )

    # Training parameters
    parser.add_argument(
        "--num-iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--rollout-length", type=int, default=500, help="Steps per rollout"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )

    # PPO hyperparameters (can override config)
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="PPO epochs per iteration (overrides config)",
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="Discount factor (overrides config)"
    )
    parser.add_argument(
        "--entropy-eps",
        type=float,
        default=None,
        help="Entropy coefficient (overrides config)",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="LSTM hidden size (overrides config)",
    )

    # WandB parameters
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="single-agent-racing",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="WandB run name"
    )

    # Video recording
    parser.add_argument(
        "--video-frequency",
        type=int,
        default=50,
        help="Record video every N iterations",
    )
    parser.add_argument(
        "--save-video-local",
        action="store_true",
        default=True,
        help="Save videos locally",
    )
    parser.add_argument(
        "--video-dir", type=str, default="videos", help="Directory to save videos"
    )

    # Save parameters
    parser.add_argument(
        "--save-path",
        type=str,
        default="perl_ppo_policy.pth",
        help="Path to save trained policy",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=100, help="Save policy every N iterations"
    )

    args = parser.parse_args()

    # Override config with command line args
    config = PPO_CONFIG.copy()
    if args.lr is not None:
        config["rl_kwargs"]["lr"] = args.lr
    if args.num_epochs is not None:
        config["rl_kwargs"]["num_epochs"] = args.num_epochs
    if args.gamma is not None:
        config["rl_kwargs"]["gamma"] = args.gamma
    if args.entropy_eps is not None:
        config["rl_kwargs"]["entropy_eps"] = args.entropy_eps
    if args.lstm_hidden_size is not None:
        config["actor_kwargs"]["lstm_hidden_size"] = args.lstm_hidden_size
        config["critic_kwargs"]["lstm_hidden_size"] = args.lstm_hidden_size

    print("=" * 80)
    print("Single-Agent Racing Training with perl_jax PPO")
    print("=" * 80)

    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb

            run_name = (
                args.wandb_run_name or f"perl-ppo-{time.strftime('%m_%d-%H_%M_%S')}"
            )
            wandb.init(
                project="racing",
                entity="peu",
                name=run_name,
                config={
                    **config,
                    "num_iterations": args.num_iterations,
                    "rollout_length": args.rollout_length,
                    "seed": args.seed,
                    "device": args.device,
                },
            )
            print("\nWandB initialized successfully")
            print(f"  Project: {args.wandb_project}")
            print(f"  Run: {wandb.run.name}")
        except Exception as e:
            print(f"\nWarning: Could not initialize wandb: {e}")
            args.use_wandb = False

    # Set random seeds
    jax_key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Build environment
    print("\n[1/4] Building environment...")
    reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {act_dim}")
    print("  Environment built successfully")

    # Create agent wrapper
    print("\n[2/4] Creating agent and trainer...")
    agent = SimpleAgent(obs_dim, act_dim)

    # Create perl_jax trainer with LSTM actor/critic
    jax_key, trainer_key = random.split(jax_key)
    trainer = Trainer(agent, config, trainer_key, device=args.device)

    # Initialize policy context
    jax_key, context_key = random.split(jax_key)
    policy_context = trainer.actor.init_context(context_key)

    print(f"  Actor: LSTM (hidden_size={config['actor_kwargs']['lstm_hidden_size']})")
    print(f"  Critic: LSTM (hidden_size={config['critic_kwargs']['lstm_hidden_size']})")
    print(f"  Device: {args.device}")
    print("  Trainer created successfully")

    # Create video directory
    if args.save_video_local:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n[3/4] Training...")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Rollout length: {args.rollout_length}")
    print(f"  PPO epochs: {config['rl_kwargs']['num_epochs']}")
    print(f"  Learning rate: {config['rl_kwargs']['lr']}")
    print(f"  Gamma: {config['rl_kwargs']['gamma']}")
    print(f"  GAE lambda: {config['rl_kwargs']['gae_lambda']}")
    print(f"  Entropy coef: {config['rl_kwargs']['entropy_eps']}")
    print(f"  Video frequency: every {args.video_frequency} iterations")
    print("-" * 80)

    start_time = time.time()

    for iteration in range(args.num_iterations):
        iter_start_time = time.time()

        # Collect rollout
        jax_key, collect_key = random.split(jax_key)
        td, policy_context, jax_key = collect_rollout_perl(
            reset_fn,
            step_fn,
            trainer.actor,
            policy_context,
            collect_key,
            args.rollout_length,
            device=args.device,
        )

        collect_time = time.time() - iter_start_time

        # Train step
        train_start_time = time.time()
        log_dict, total_frames = trainer.step(td)
        train_time = time.time() - train_start_time

        # Reset policy context after training
        jax_key, context_key = random.split(jax_key)
        policy_context = trainer.actor.init_context(context_key)

        # Add timing info
        log_dict.update(
            {
                "time/collect_sec": collect_time,
                "time/train_sec": train_time,
                "time/total_sec": time.time() - iter_start_time,
                "time/fps": args.rollout_length / collect_time,
            }
        )

        # Log to wandb
        if args.use_wandb:
            import wandb

            wandb.log(log_dict, step=total_frames)

        # Print progress
        if iteration % 10 == 0:
            ep_rew = log_dict.get("rollout/ep_rew_mean", 0.0)
            step_rew = log_dict.get("rollout/step_rew_mean", 0.0)
            loss = log_dict.get("train/loss", 0.0)
            entropy = log_dict.get("train/entropy", 0.0)

            print(
                f"Iter {iteration:4d} | "
                f"Frames: {total_frames:8d} | "
                f"EpRew: {ep_rew:7.2f} | "
                f"StepRew: {step_rew:6.3f} | "
                f"Loss: {loss:6.3f} | "
                f"Entropy: {entropy:5.3f}"
            )

        # Record video
        if (
            iteration % args.video_frequency == 0
            or iteration == args.num_iterations - 1
        ):
            print(f"  Recording video at iteration {iteration}...")

            # Record episode using JAX actor directly
            jax_key, video_key = random.split(jax_key)
            try:
                from video_recorder import save_video

                # Create stateful policy function that uses JAX actor
                class StatefulPolicy:
                    def __init__(self, actor, initial_context):
                        self.actor = actor
                        self.context = initial_context
                        self.rng = random.PRNGKey(0)

                    def __call__(self, obs):
                        self.rng, action_rng = random.split(self.rng)
                        # Use deterministic mode by getting mean action
                        action, self.context = self.actor(obs, self.context, action_rng)
                        return action

                # Initialize policy context for video
                jax_key, video_context_key = random.split(jax_key)
                video_context = trainer.actor.init_context(video_context_key)
                policy_fn = StatefulPolicy(trainer.actor, video_context)

                frames, episode_reward, episode_length = record_episode(
                    reset_fn, step_fn, policy_fn, video_key, max_steps=500
                )

                print(
                    f"    Episode reward: {episode_reward:.2f}, Length: {episode_length}"
                )

                # Save locally
                if args.save_video_local:
                    video_path = f"{args.video_dir}/iteration_{iteration:04d}.mp4"
                    save_video(frames, video_path, fps=20)

                # Upload to wandb
                if args.use_wandb:
                    wandb_video = frames_to_wandb_video(frames, fps=20)
                    if wandb_video is not None:
                        wandb.log(
                            {
                                "video": wandb_video,
                                "video/episode_reward": episode_reward,
                                "video/episode_length": episode_length,
                            },
                            step=total_frames,
                        )
            except Exception as e:
                print(f"    Warning: Video recording failed: {e}")
                import traceback

                traceback.print_exc()

        # Save policy checkpoint
        if iteration % args.save_frequency == 0 or iteration == args.num_iterations - 1:
            checkpoint_path = (
                f"{args.save_path.replace('.pth', '')}_{iteration:04d}.pth"
            )
            save_actor(trainer.actor.trl_actor, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    elapsed_time = time.time() - start_time

    print("-" * 80)
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Average time per iteration: {elapsed_time / args.num_iterations:.2f}s")

    # Save final policy
    print(f"\n[4/4] Saving final policy to {args.save_path}...")
    save_actor(trainer.actor.trl_actor, args.save_path)
    print("  Policy saved successfully")

    # Upload to wandb
    if args.use_wandb:
        import wandb

        wandb.save(args.save_path)
        wandb.finish()
        print("  Policy uploaded to wandb")

    print("\n" + "=" * 80)
    print("SUCCESS: Training completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
