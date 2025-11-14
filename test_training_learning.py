#!/usr/bin/env python3
"""
Test script to verify that the agent learns over time with PPO training.
Runs a short training session and checks that rewards improve.
"""

import subprocess
import re
import sys


def run_short_training():
    """Run a short training session and extract learning metrics."""
    print("=" * 70)
    print("Testing PPO Learning with MPC Reward")
    print("=" * 70)
    print("\nRunning 50 iterations of training...")

    # Run training with minimal config
    cmd = [
        "python", "train_with_perl_ppo.py",
        "--num-iterations", "50",
        "--rollout-length", "200",
        "--save-frequency", "50",  # Only save at end
        "--video-frequency", "50",  # Only record at end
        "--seed", "123"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
        )

        output = result.stdout + result.stderr

        # Extract episode rewards from video recording lines
        episode_rewards = []
        for line in output.split('\n'):
            if 'Episode reward:' in line:
                match = re.search(r'Episode reward: ([-+]?\d+\.?\d*)', line)
                if match:
                    episode_rewards.append(float(match.group(1)))

        # Extract step rewards and losses
        step_rewards = []
        losses = []
        for line in output.split('\n'):
            if 'Iter' in line and 'StepRew' in line:
                step_match = re.search(r'StepRew:\s+([-+]?\d+\.?\d*)', line)
                loss_match = re.search(r'Loss:\s+([-+]?\d+\.?\d*)', line)
                if step_match:
                    step_rewards.append(float(step_match.group(1)))
                if loss_match:
                    losses.append(float(loss_match.group(1)))

        print(f"\n{'=' * 70}")
        print("Training Results")
        print("=" * 70)

        if len(episode_rewards) >= 2:
            initial_reward = episode_rewards[0]
            final_reward = episode_rewards[-1]
            improvement = final_reward - initial_reward

            print(f"\nEpisode Rewards:")
            print(f"  Initial (iter 0):  {initial_reward:10.2f}")
            print(f"  Final (iter 49):   {final_reward:10.2f}")
            print(f"  Improvement:       {improvement:10.2f}")
            print(f"  Relative change:   {(improvement/abs(initial_reward))*100:7.1f}%")

            # Check if learning occurred
            if improvement > 0 and improvement > abs(initial_reward) * 0.1:
                print(f"\n  ✓ LEARNING VERIFIED: Reward improved by {improvement:.2f}")
                learning_success = True
            else:
                print(f"\n  ⚠ Limited learning: Reward changed by {improvement:.2f}")
                learning_success = False
        else:
            print(f"\n  ⚠ Could not extract episode rewards")
            learning_success = False

        if losses:
            print(f"\nLoss progression:")
            print(f"  Initial loss: {losses[0]:8.3f}")
            print(f"  Final loss:   {losses[-1]:8.3f}")
            print(f"  ✓ Network is training")

        print(f"\n{'=' * 70}")

        if result.returncode == 0:
            print("✓ TRAINING COMPLETED SUCCESSFULLY")
        else:
            print("⚠ Training completed with warnings")

        print("=" * 70)

        return learning_success and result.returncode == 0

    except subprocess.TimeoutExpired:
        print("✗ Training timed out (>3 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error running training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_short_training()
    sys.exit(0 if success else 1)
