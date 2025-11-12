#!/usr/bin/env python3
"""
Training script for single-agent racing with PPO.
Includes wandb tracking and video recording.
"""
import jax
import jax.numpy as jnp
from jax import random
import argparse
import time
from pathlib import Path
from single_agent_env import build_single_agent_env
from ppo import create_train_state, collect_rollout, compute_gae, train_step
from video_recorder import record_episode, save_video, frames_to_wandb_video
import pickle
import numpy as np


def train_ppo_with_logging(
    rng,
    env_reset,
    env_step,
    obs_dim,
    action_dim,
    num_iterations=1000,
    rollout_length=2048,
    batch_size=64,
    num_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    use_wandb=False,
    video_frequency=50,
    save_video_local=True,
    video_dir="videos",
):
    """
    Train PPO agent with wandb logging and video recording.

    Args:
        rng: JAX random key
        env_reset: Environment reset function
        env_step: Environment step function
        obs_dim: Observation dimension
        action_dim: Action dimension
        num_iterations: Number of training iterations
        rollout_length: Steps per rollout
        batch_size: Minibatch size
        num_epochs: PPO epochs per iteration
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        use_wandb: Whether to use wandb logging
        video_frequency: Record video every N iterations
        save_video_local: Whether to save videos locally
        video_dir: Directory to save videos

    Returns:
        train_state: Final training state
        metrics_history: Training metrics
    """
    # Initialize wandb if requested
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Disabling wandb logging.")
            use_wandb = False

    # Create video directory
    if save_video_local:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    rng, init_rng = random.split(rng)
    train_state_obj = create_train_state(init_rng, obs_dim, action_dim, learning_rate)

    metrics_history = []

    for iteration in range(num_iterations):
        # Collect rollout
        (
            observations,
            actions,
            rewards,
            dones,
            values,
            log_probs,
            rollout_stats,
            rng,
        ) = collect_rollout(rng, train_state_obj, env_reset, env_step, rollout_length)

        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs - track training metrics
        epoch_metrics = []
        for epoch in range(num_epochs):
            # Shuffle data
            rng, shuffle_rng = random.split(rng)
            perm = random.permutation(shuffle_rng, rollout_length)

            # Mini-batch updates
            for i in range(0, rollout_length, batch_size):
                batch_idx = perm[i : i + batch_size]
                train_state_obj, step_metrics = train_step(
                    train_state_obj,
                    observations[batch_idx],
                    actions[batch_idx],
                    log_probs[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                )
                epoch_metrics.append(step_metrics)

        # Average metrics over last epoch
        last_epoch_metrics = epoch_metrics[-len(range(0, rollout_length, batch_size)) :]
        avg_policy_loss = float(
            jnp.mean(jnp.array([m["policy_loss"] for m in last_epoch_metrics]))
        )
        avg_value_loss = float(
            jnp.mean(jnp.array([m["value_loss"] for m in last_epoch_metrics]))
        )
        avg_entropy = float(
            jnp.mean(jnp.array([m["entropy"] for m in last_epoch_metrics]))
        )

        # Compute rollout statistics
        if rollout_stats["num_episodes"] > 0:
            mean_episode_reward = float(np.mean(rollout_stats["episode_rewards"]))
            std_episode_reward = float(np.std(rollout_stats["episode_rewards"]))
            min_episode_reward = float(np.min(rollout_stats["episode_rewards"]))
            max_episode_reward = float(np.max(rollout_stats["episode_rewards"]))
            mean_episode_length = float(np.mean(rollout_stats["episode_lengths"]))
        else:
            mean_episode_reward = float(rewards.sum())
            std_episode_reward = 0.0
            min_episode_reward = float(rewards.sum())
            max_episode_reward = float(rewards.sum())
            mean_episode_length = rollout_length

        # Compute action statistics
        action_mean = float(actions.mean())
        action_std = float(actions.std())
        throttle_mean = float(actions[:, 0].mean())
        steering_mean = float(actions[:, 1].mean())

        # Log metrics
        metrics = {
            "iteration": iteration,
            # Reward metrics
            "rollout/mean_reward": float(rewards.mean()),
            "rollout/std_reward": float(rewards.std()),
            "rollout/min_reward": float(rewards.min()),
            "rollout/max_reward": float(rewards.max()),
            "rollout/total_reward": float(rewards.sum()),
            # Episode metrics
            "episode/mean_reward": mean_episode_reward,
            "episode/std_reward": std_episode_reward,
            "episode/min_reward": min_episode_reward,
            "episode/max_reward": max_episode_reward,
            "episode/mean_length": mean_episode_length,
            "episode/num_episodes": rollout_stats["num_episodes"],
            # Value function metrics
            "value/mean": float(values.mean()),
            "value/std": float(values.std()),
            "value/min": float(values.min()),
            "value/max": float(values.max()),
            # Advantage metrics
            "advantage/mean": float(advantages.mean()),
            "advantage/std": float(advantages.std()),
            "advantage/min": float(advantages.min()),
            "advantage/max": float(advantages.max()),
            # Loss metrics
            "loss/policy": avg_policy_loss,
            "loss/value": avg_value_loss,
            "loss/entropy": avg_entropy,
            # Action metrics
            "action/mean": action_mean,
            "action/std": action_std,
            "action/throttle_mean": throttle_mean,
            "action/steering_mean": steering_mean,
        }
        metrics_history.append(metrics)

        # Log to wandb
        if use_wandb:
            wandb.log(metrics, step=iteration)

        # Print progress
        if iteration % 10 == 0:
            print(
                f"Iter {iteration:4d} | "
                f"Reward: {metrics['rollout/mean_reward']:7.3f} | "
                f"Ep Reward: {mean_episode_reward:8.2f} | "
                f"Value: {metrics['value/mean']:7.3f} | "
                f"Entropy: {avg_entropy:6.3f} | "
                f"Episodes: {rollout_stats['num_episodes']:2d}"
            )

        # Record video
        if iteration % video_frequency == 0 or iteration == num_iterations - 1:
            print(f"  Recording video at iteration {iteration}...")

            # Create deterministic policy function
            def policy_fn(obs):
                action_mean, _, _ = train_state_obj.apply_fn(
                    train_state_obj.params, obs
                )
                return jnp.clip(action_mean, -1.0, 1.0)

            # Record episode
            rng, video_rng = random.split(rng)
            frames, episode_reward, episode_length = record_episode(
                env_reset, env_step, policy_fn, video_rng, max_steps=500
            )

            print(f"    Episode reward: {episode_reward:.2f}, Length: {episode_length}")

            # Save locally
            if save_video_local:
                video_path = f"{video_dir}/iteration_{iteration:04d}.mp4"
                save_video(frames, video_path, fps=20)

            # Upload to wandb
            if use_wandb:
                wandb_video = frames_to_wandb_video(frames, fps=20)
                if wandb_video is not None:
                    wandb.log(
                        {
                            "video": wandb_video,
                            "video/episode_reward": episode_reward,
                            "video/episode_length": episode_length,
                        },
                        step=iteration,
                    )

    return train_state_obj, metrics_history


def main():
    parser = argparse.ArgumentParser(description="Train single-agent racing with PPO")
    # Training parameters
    parser.add_argument(
        "--num-iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=500,
        help="Steps per rollout (matches episode length)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Minibatch size for PPO"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="PPO epochs per iteration"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda parameter"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging parameters
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="single-agent-racing",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Wandb entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name"
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=[], help="Wandb tags"
    )

    # Video recording parameters
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
        default="single_agent_policy.pkl",
        help="Path to save trained policy",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Single-Agent Racing PPO Training")
    print("=" * 80)

    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                # project=args.wandb_project,
                # entity=args.wandb_entity,
                entity="peu",
                project="racing",
                name=f"single-agent-ppo-{time.strftime('%m_%d-%H_%M_%S')}",
                tags=args.wandb_tags,
                config={
                    "num_iterations": args.num_iterations,
                    "rollout_length": args.rollout_length,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "seed": args.seed,
                },
            )
            print("\nWandB initialized successfully")
            print(f"  Project: {args.wandb_project}")
            print(f"  Run: {wandb.run.name}")
        except Exception as e:
            print(f"\nWarning: Could not initialize wandb: {e}")
            args.use_wandb = False

    # Build environment
    print("\n[1/3] Building environment...")
    try:
        reset_fn, step_fn, obs_dim, act_dim = build_single_agent_env(num_envs=1)
        print(f"  Observation dimension: {obs_dim}")
        print(f"  Action dimension: {act_dim}")
        print(f"  Episode length: {args.rollout_length}")
        print("  Environment built successfully")
    except Exception as e:
        print(f"  Failed to build environment: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test environment
    print("\n[2/3] Testing environment...")
    try:
        rng = random.PRNGKey(args.seed)
        test_rng, rng = random.split(rng)
        state, obs = reset_fn(test_rng)
        print(f"  Initial observation shape: {obs.shape}")

        # Test step
        action = jnp.array([0.5, 0.0])
        state, obs, reward, done, truncated = step_fn(state, action)
        print(f"  Step successful, reward: {float(reward):.4f}")
    except Exception as e:
        print(f"  Environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Training
    print("\n[3/3] Training PPO agent...")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Rollout length: {args.rollout_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Gamma: {args.gamma}")
    print(f"  GAE lambda: {args.gae_lambda}")
    print(f"  Video frequency: every {args.video_frequency} iterations")
    print(f"  Save videos locally: {args.save_video_local}")
    if args.save_video_local:
        print(f"  Video directory: {args.video_dir}")
    print("-" * 80)

    try:
        start_time = time.time()

        train_state, metrics_history = train_ppo_with_logging(
            rng=rng,
            env_reset=reset_fn,
            env_step=step_fn,
            obs_dim=obs_dim,
            action_dim=act_dim,
            num_iterations=args.num_iterations,
            rollout_length=args.rollout_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            use_wandb=args.use_wandb,
            video_frequency=args.video_frequency,
            save_video_local=args.save_video_local,
            video_dir=args.video_dir,
        )

        elapsed_time = time.time() - start_time

        print("-" * 80)
        print(f"Training completed in {elapsed_time:.2f} seconds")
        print(f"Average time per iteration: {elapsed_time / args.num_iterations:.2f}s")

        # Print final statistics
        final_metrics = metrics_history[-10:]
        mean_reward = sum(m["rollout/mean_reward"] for m in final_metrics) / len(
            final_metrics
        )
        mean_episode_reward = sum(
            m["episode/mean_reward"] for m in final_metrics
        ) / len(final_metrics)

        print("\nFinal Performance (last 10 iterations):")
        print(f"  Mean reward per step: {mean_reward:.4f}")
        print(f"  Mean episode reward: {mean_episode_reward:.2f}")

        # Save policy
        print(f"\nSaving policy to {args.save_path}...")
        with open(args.save_path, "wb") as f:
            pickle.dump(
                {
                    "params": train_state.params,
                    "obs_dim": obs_dim,
                    "action_dim": act_dim,
                    "metrics_history": metrics_history,
                },
                f,
            )
        print("  Policy saved successfully")

        # Upload final model to wandb
        if args.use_wandb:
            import wandb

            wandb.save(args.save_path)
            print("  Policy uploaded to wandb")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        if args.use_wandb:
            import wandb

            wandb.finish(exit_code=1)
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS: Training completed!")
    print("=" * 80)

    # Print training summary
    print("\nTraining Summary:")
    print(f"  Total iterations: {len(metrics_history)}")
    print(f"  Initial mean reward: {metrics_history[0]['rollout/mean_reward']:.4f}")
    print(f"  Final mean reward: {metrics_history[-1]['rollout/mean_reward']:.4f}")

    # Check improvement
    initial_perf = sum(m["rollout/mean_reward"] for m in metrics_history[:10]) / min(
        10, len(metrics_history)
    )
    final_perf = sum(m["rollout/mean_reward"] for m in metrics_history[-10:]) / min(
        10, len(metrics_history)
    )
    improvement = final_perf - initial_perf

    print(f"  Improvement: {improvement:+.4f}")

    if improvement > 0:
        print("  Status: Policy is improving!")
    else:
        print("  Status: Policy may need tuning")

    # Finish wandb
    if args.use_wandb:
        import wandb

        wandb.finish()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
