"""
Test the JAX-based simulation framework with a random policy
"""

import jax
import jax.numpy as jnp
import numpy as np

from car_jax.env import CarEnv
from car_jax.sim import Agent, Reward, Sim, collect_factory, batch_collect_factory
from car_jax.policy import Policy, PolicyAggregator, RandomPolicy, ForwardPolicy
from car_jax.sim.agent import IdentityAgent, RelativePositionAgent
from car_jax.sim.reward import VelocityReward, ProgressReward, SurvivalReward


def test_single_agent():
    """Test single agent simulation"""
    print("\n" + "="*80)
    print("Testing Single Agent Simulation")
    print("="*80)

    # Create environment
    env = CarEnv(num_cars=1, dt=0.05, max_steps=50)

    # Create agent and reward
    agent = IdentityAgent(agent_idx=0)
    reward_fn = ProgressReward(agent_idx=0, scale=1.0)

    # Create sim
    sim = Sim(env=env, agents=[agent], reward=[reward_fn], max_steps=50)

    # Create policy
    policy = RandomPolicy(env.action_space)

    # Test reset
    key = jax.random.key(42)
    obs, obs_for_pi, agent_context, sim_context = sim.reset(key)

    print(f"Initial obs_for_pi: {obs_for_pi[0]}")
    print(f"Initial obs timestep: {obs.timestep}")

    # Test step
    key = jax.random.key(43)
    action, policy_context = policy(obs_for_pi[0], {}, key)
    print(f"Sampled action: {action}")

    next_obs, reward, terminated, truncated, next_obs_for_pi, next_agent_context, next_sim_context = sim.step(
        obs, [action], agent_context, sim_context
    )

    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Next obs timestep: {next_obs.timestep}")

    print("\n✓ Single agent test passed!")
    return True


def test_multi_agent():
    """Test multi-agent simulation"""
    print("\n" + "="*80)
    print("Testing Multi-Agent Simulation")
    print("="*80)

    # Create environment with 3 cars
    env = CarEnv(num_cars=3, dt=0.05, max_steps=50)

    # Create agents and rewards for each car
    agents = [
        IdentityAgent(agent_idx=0),
        IdentityAgent(agent_idx=1),
        RelativePositionAgent(agent_idx=2, other_agent_indices=[0, 1]),
    ]

    rewards = [
        ProgressReward(agent_idx=0, scale=1.0),
        VelocityReward(agent_idx=1, scale=0.5),
        SurvivalReward(reward_value=1.0),
    ]

    # Create sim
    sim = Sim(env=env, agents=agents, reward=rewards, max_steps=50)

    # Create policies
    policies = [
        RandomPolicy(env.action_space),
        ForwardPolicy(target_vel=0.8, target_steer=0.0),
        RandomPolicy(env.action_space),
    ]

    aggregated_policy = PolicyAggregator(policies)

    # Test reset
    key = jax.random.key(42)
    obs, obs_for_pi, agent_context, sim_context = sim.reset(key)

    print(f"Number of agents: {len(obs_for_pi)}")
    print(f"Agent 0 obs shape: {obs_for_pi[0].shape}")
    print(f"Agent 1 obs shape: {obs_for_pi[1].shape}")
    print(f"Agent 2 obs shape (relative): {obs_for_pi[2].shape}")

    # Test step
    key = jax.random.key(43)
    policy_context = aggregated_policy.init_context(jax.random.key(44))
    actions, policy_context = aggregated_policy(obs_for_pi, policy_context, key)

    print(f"Number of actions: {len(actions)}")
    print(f"Action 0: {actions[0]}")
    print(f"Action 1: {actions[1]}")
    print(f"Action 2: {actions[2]}")

    next_obs, rewards, terminated, truncated, next_obs_for_pi, next_agent_context, next_sim_context = sim.step(
        obs, actions, agent_context, sim_context
    )

    print(f"Rewards: {rewards}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    print("\n✓ Multi-agent test passed!")
    return True


def test_collection():
    """Test data collection with random policy"""
    print("\n" + "="*80)
    print("Testing Data Collection")
    print("="*80)

    # Create environment
    env = CarEnv(num_cars=2, dt=0.05, max_steps=20)

    # Create agents and rewards
    agents = [
        IdentityAgent(agent_idx=0),
        IdentityAgent(agent_idx=1),
    ]

    rewards = [
        ProgressReward(agent_idx=0, scale=1.0),
        ProgressReward(agent_idx=1, scale=1.0),
    ]

    # Create sim
    sim = Sim(env=env, agents=agents, reward=rewards, max_steps=20)

    # Create policy
    policies = [
        ForwardPolicy(target_vel=0.8, target_steer=0.1),
        ForwardPolicy(target_vel=0.7, target_steer=-0.1),
    ]

    aggregated_policy = PolicyAggregator(policies)

    # Create collection function
    record_indices = [0, 1]  # Record both agents
    timesteps = 30

    collect = collect_factory(
        sim_reset=sim.reset,
        sim_step=sim.step,
        pi=aggregated_policy,
        record_indices=record_indices,
        timesteps=timesteps,
        return_obs=False,
        deterministic=False,
    )

    # Collect data
    key = jax.random.key(42)
    policy_context = aggregated_policy.init_context(jax.random.key(43))

    print("Starting collection...")
    transitions = collect(key, policy_context)

    print(f"Collected {timesteps} transitions")
    print(f"Transition obs shape: {len(transitions.obs)}")
    print(f"Transition action shape: {len(transitions.action)}")
    print(f"Transition reward shape: {transitions.reward.shape}")
    print(f"Episode reward at end: {transitions.episode_reward[-1]}")

    print("\n✓ Collection test passed!")
    return transitions


def test_batch_collection():
    """Test batched data collection"""
    print("\n" + "="*80)
    print("Testing Batch Collection")
    print("="*80)

    # Create environment
    env = CarEnv(num_cars=1, dt=0.05, max_steps=20)

    # Create agent and reward
    agent = IdentityAgent(agent_idx=0)
    reward_fn = ProgressReward(agent_idx=0, scale=1.0)

    # Create sim
    sim = Sim(env=env, agents=[agent], reward=[reward_fn], max_steps=20)

    # Create policy
    policy = RandomPolicy(env.action_space)

    # Create batch collection function
    record_indices = [0]
    total_timesteps = 100
    num_envs = 5

    batch_collect = batch_collect_factory(
        sim_reset=sim.reset,
        sim_step=sim.step,
        pi=policy,
        record_indices=record_indices,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        deterministic=False,
    )

    # Collect data
    key = jax.random.key(42)
    policy_context = policy.init_context(jax.random.key(43))

    print("Starting batch collection...")
    transitions = batch_collect(key, policy_context)

    print(f"Collected {total_timesteps} transitions across {num_envs} environments")
    print(f"Transition reward shape: {transitions.reward.shape}")
    print(f"Transition truncated shape: {transitions.truncated.shape}")

    print("\n✓ Batch collection test passed!")
    return transitions


def visualize_trajectory(transitions):
    """Create visualization data from transitions"""
    print("\n" + "="*80)
    print("Creating Visualization Data")
    print("="*80)

    # Extract position data
    # transitions.obs is a list with one element per agent
    agent_0_obs = transitions.obs[0]  # Shape: (timesteps, 6) [x, y, psi, vx, vy, omega]

    # Convert to numpy for visualization
    agent_0_obs_np = np.array(agent_0_obs)

    print(f"Agent 0 trajectory shape: {agent_0_obs_np.shape}")
    print(f"Agent 0 x range: [{agent_0_obs_np[:, 0].min():.3f}, {agent_0_obs_np[:, 0].max():.3f}]")
    print(f"Agent 0 y range: [{agent_0_obs_np[:, 1].min():.3f}, {agent_0_obs_np[:, 1].max():.3f}]")

    # Create visualization-compatible data structure
    # Format: (timesteps, features) where features = [x, y, psi, ...]
    viz_data = agent_0_obs_np

    print(f"\nVisualization data shape: {viz_data.shape}")
    print("Data format: [x, y, psi, vx, vy, omega]")

    # Save for visualization
    np.save('car_jax_trajectory.npy', viz_data)
    print("Saved trajectory to 'car_jax_trajectory.npy'")

    print("\n✓ Visualization data created!")
    return viz_data


if __name__ == "__main__":
    print("\n" + "="*80)
    print("JAX Car Simulation Framework Test Suite")
    print("="*80)

    # Run tests
    test_single_agent()
    test_multi_agent()
    transitions = test_collection()
    batch_transitions = test_batch_collection()

    # Create visualization data
    viz_data = visualize_trajectory(transitions)

    print("\n" + "="*80)
    print("All Tests Passed! ✓")
    print("="*80)
    print("\nFramework is ready to use!")
    print("- Single agent: ✓")
    print("- Multi-agent: ✓")
    print("- Data collection: ✓")
    print("- Batch collection: ✓")
    print("- Visualization format: ✓")
