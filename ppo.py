"""
Proximal Policy Optimization (PPO) implementation in JAX.
"""
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
from typing import NamedTuple, Tuple


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    action_dim: int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x):
        # Shared feature extraction
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        # Actor head (policy)
        actor = nn.Dense(self.hidden_size // 2)(x)
        actor = nn.relu(actor)
        action_mean = nn.Dense(self.action_dim)(actor)
        action_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        # Critic head (value function)
        critic = nn.Dense(self.hidden_size // 2)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)

        return action_mean, action_logstd, value.squeeze(-1)


class Transition(NamedTuple):
    """Single transition for PPO training."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


class PPOTrainState(train_state.TrainState):
    """Training state for PPO with additional fields."""
    pass


def create_train_state(rng, obs_dim, action_dim, learning_rate=3e-4):
    """Create initial training state for PPO."""
    network = ActorCritic(action_dim=action_dim)
    params = network.init(rng, jnp.ones((1, obs_dim)))
    tx = optax.adam(learning_rate)
    return PPOTrainState.create(apply_fn=network.apply, params=params, tx=tx)


def sample_action(rng, params, network_apply, obs):
    """Sample action from policy."""
    action_mean, action_logstd, value = network_apply(params, obs)
    action_std = jnp.exp(action_logstd)

    # Sample action
    action = action_mean + action_std * random.normal(rng, shape=action_mean.shape)

    # Compute log probability
    log_prob = -0.5 * (
        jnp.sum(((action - action_mean) / action_std) ** 2)
        + jnp.sum(2 * action_logstd)
        + action_mean.shape[-1] * jnp.log(2 * jnp.pi)
    )

    return action, log_prob, value


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0

    # Iterate backwards through time
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = jnp.array(advantages)
    returns = advantages + values
    return advantages, returns


def compute_log_prob(params, network_apply, obs, action):
    """Compute log probability of an action under current policy."""
    action_mean, action_logstd, _ = network_apply(params, obs)
    action_std = jnp.exp(action_logstd)

    log_prob = -0.5 * (
        jnp.sum(((action - action_mean) / action_std) ** 2)
        + jnp.sum(2 * action_logstd)
        + action_mean.shape[-1] * jnp.log(2 * jnp.pi)
    )
    return log_prob


def ppo_loss(
    params,
    network_apply,
    obs_batch,
    action_batch,
    old_log_prob_batch,
    advantage_batch,
    return_batch,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """Compute PPO loss."""

    # Get current policy outputs
    def forward(obs):
        action_mean, action_logstd, value = network_apply(params, obs)
        action_std = jnp.exp(action_logstd)
        return action_mean, action_std, action_logstd, value

    action_means, action_stds, action_logstds, values = jax.vmap(forward)(obs_batch)

    # Compute log probabilities
    def log_prob_fn(action, action_mean, action_std, action_logstd):
        return -0.5 * (
            jnp.sum(((action - action_mean) / action_std) ** 2)
            + jnp.sum(2 * action_logstd)
            + action_mean.shape[-1] * jnp.log(2 * jnp.pi)
        )

    log_probs = jax.vmap(log_prob_fn)(
        action_batch, action_means, action_stds, action_logstds
    )

    # PPO clipped objective
    ratio = jnp.exp(log_probs - old_log_prob_batch)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -jnp.mean(
        jnp.minimum(ratio * advantage_batch, clipped_ratio * advantage_batch)
    )

    # Value loss
    value_loss = jnp.mean((values - return_batch) ** 2)

    # Entropy bonus
    entropy = jnp.mean(jnp.sum(action_logstds + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))

    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    return total_loss, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'total_loss': total_loss,
    }


def train_step(state, obs_batch, action_batch, old_log_prob_batch, advantage_batch, return_batch):
    """Single PPO training step."""

    def loss_fn(params):
        return ppo_loss(
            params,
            state.apply_fn,
            obs_batch,
            action_batch,
            old_log_prob_batch,
            advantage_batch,
            return_batch,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def collect_rollout(
    rng,
    train_state,
    env_reset,
    env_step,
    num_steps,
):
    """Collect a rollout of experience."""
    rng, reset_rng, action_rng = random.split(rng, 3)

    # Reset environment
    env_state, obs = env_reset(reset_rng)

    # Storage
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    episode_rewards = []
    episode_lengths = []

    # Episode tracking
    current_episode_reward = 0.0
    current_episode_length = 0

    # Collect experience
    for step in range(num_steps):
        # Sample action
        action_rng, rng = random.split(action_rng)
        action, log_prob, value = sample_action(
            action_rng, train_state.params, train_state.apply_fn, obs
        )

        # Clip action to valid range
        action = jnp.clip(action, -1.0, 1.0)

        # Store transition
        observations.append(obs)
        actions.append(action)
        values.append(value)
        log_probs.append(log_prob)

        # Step environment
        env_state, obs, reward, done, truncated = env_step(env_state, action)

        rewards.append(reward)
        dones.append(done)

        current_episode_reward += float(reward)
        current_episode_length += 1

        # Reset if done (but keep collecting)
        if done or truncated:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0.0
            current_episode_length = 0

            rng, reset_rng = random.split(rng)
            env_state, obs = env_reset(reset_rng)

    # Rollout statistics
    rollout_stats = {
        'num_episodes': len(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return (
        jnp.array(observations),
        jnp.array(actions),
        jnp.array(rewards),
        jnp.array(dones),
        jnp.array(values),
        jnp.array(log_probs),
        rollout_stats,
        rng,
    )


def train_ppo(
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
):
    """
    Train PPO agent.

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

    Returns:
        train_state: Final training state
        metrics_history: Training metrics
    """
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
        ) = collect_rollout(
            rng, train_state_obj, env_reset, env_step, rollout_length
        )

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
        last_epoch_metrics = epoch_metrics[-len(range(0, rollout_length, batch_size)):]
        avg_policy_loss = float(jnp.mean(jnp.array([m['policy_loss'] for m in last_epoch_metrics])))
        avg_value_loss = float(jnp.mean(jnp.array([m['value_loss'] for m in last_epoch_metrics])))
        avg_entropy = float(jnp.mean(jnp.array([m['entropy'] for m in last_epoch_metrics])))

        # Compute rollout statistics
        if rollout_stats['num_episodes'] > 0:
            mean_episode_reward = float(np.mean(rollout_stats['episode_rewards']))
            std_episode_reward = float(np.std(rollout_stats['episode_rewards']))
            mean_episode_length = float(np.mean(rollout_stats['episode_lengths']))
        else:
            mean_episode_reward = float(rewards.sum())
            std_episode_reward = 0.0
            mean_episode_length = rollout_length

        # Log metrics
        metrics = {
            'iteration': iteration,
            'mean_reward': float(rewards.mean()),
            'std_reward': float(rewards.std()),
            'total_reward': float(rewards.sum()),
            'mean_value': float(values.mean()),
            'std_value': float(values.std()),
            'mean_advantage': float(advantages.mean()),
            'std_advantage': float(advantages.std()),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'num_episodes': rollout_stats['num_episodes'],
            'mean_episode_reward': mean_episode_reward,
            'std_episode_reward': std_episode_reward,
            'mean_episode_length': mean_episode_length,
        }
        metrics_history.append(metrics)

        if iteration % 10 == 0:
            print(
                f"Iter {iteration:4d} | "
                f"Reward: {metrics['mean_reward']:7.3f} | "
                f"Total: {metrics['total_reward']:8.2f} | "
                f"Value: {metrics['mean_value']:7.3f} | "
                f"Episodes: {rollout_stats['num_episodes']:2d}"
            )

    return train_state_obj, metrics_history
