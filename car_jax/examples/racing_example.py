"""
Comprehensive example: Multi-agent car racing with data collection and visualization
"""

import jax
import jax.numpy as jnp
import numpy as np

from car_jax.env import CarEnv
from car_jax.sim import Sim, collect_factory
from car_jax.sim.agent import IdentityAgent, RelativePositionAgent
from car_jax.sim.reward import ProgressReward, CompositeReward, VelocityReward, SurvivalReward
from car_jax.policy import PolicyAggregator, ForwardPolicy, RandomPolicy


def main():
    print("\n" + "="*80)
    print("Multi-Agent Car Racing Example")
    print("="*80)

    # ========================================================================
    # 1. Setup Environment
    # ========================================================================
    print("\n[1/6] Setting up environment...")

    num_cars = 3
    env = CarEnv(
        num_cars=num_cars,
        dt=0.05,  # 50ms timestep
        max_steps=200,
    )

    print(f"  - Created environment with {num_cars} cars")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")

    # ========================================================================
    # 2. Create Agents
    # ========================================================================
    print("\n[2/6] Creating agents...")

    agents = [
        # Agent 0: Gets ego observations only
        IdentityAgent(agent_idx=0),

        # Agent 1: Gets ego observations only
        IdentityAgent(agent_idx=1),

        # Agent 2: Gets ego + relative observations to others
        RelativePositionAgent(agent_idx=2, other_agent_indices=[0, 1]),
    ]

    print(f"  - Agent 0: IdentityAgent (ego obs only)")
    print(f"  - Agent 1: IdentityAgent (ego obs only)")
    print(f"  - Agent 2: RelativePositionAgent (ego + relative obs)")

    # ========================================================================
    # 3. Create Reward Functions
    # ========================================================================
    print("\n[3/6] Creating reward functions...")

    rewards = [
        # Agent 0: Maximize progress
        ProgressReward(agent_idx=0, scale=1.0),

        # Agent 1: Composite reward (progress + velocity)
        CompositeReward(
            rewards=[
                ProgressReward(agent_idx=1, scale=0.8),
                VelocityReward(agent_idx=1, scale=0.2),
            ],
            weights=[1.0, 1.0]
        ),

        # Agent 2: Maximize survival time
        SurvivalReward(reward_value=1.0),
    ]

    print(f"  - Agent 0: ProgressReward")
    print(f"  - Agent 1: CompositeReward (progress + velocity)")
    print(f"  - Agent 2: SurvivalReward")

    # ========================================================================
    # 4. Create Simulation
    # ========================================================================
    print("\n[4/6] Creating simulation...")

    sim = Sim(
        env=env,
        agents=agents,
        reward=rewards,
        max_steps=200,
    )

    print(f"  - Simulation created with {len(agents)} agents")

    # ========================================================================
    # 5. Create Policies
    # ========================================================================
    print("\n[5/6] Creating policies...")

    policies = [
        # Agent 0: Drives fast and turns left
        ForwardPolicy(target_vel=0.95, target_steer=0.08),

        # Agent 1: Drives moderate speed, turns right
        ForwardPolicy(target_vel=0.75, target_steer=-0.05),

        # Agent 2: Random exploration
        RandomPolicy(env.action_space),
    ]

    aggregated_policy = PolicyAggregator(policies)

    print(f"  - Agent 0: ForwardPolicy (fast left turn)")
    print(f"  - Agent 1: ForwardPolicy (moderate right turn)")
    print(f"  - Agent 2: RandomPolicy (exploration)")

    # ========================================================================
    # 6. Run Simulation and Collect Data
    # ========================================================================
    print("\n[6/6] Running simulation...")

    # Create collection function
    collect = collect_factory(
        sim_reset=sim.reset,
        sim_step=sim.step,
        pi=aggregated_policy,
        record_indices=[0, 1, 2],  # Record all agents
        timesteps=200,
        deterministic=True,  # Use deterministic policies for reproducibility
    )

    # Run collection
    key = jax.random.key(42)
    policy_context = aggregated_policy.init_context(jax.random.key(43))

    print("  - Collecting data...")
    transitions = collect(key, policy_context)

    # ========================================================================
    # Results Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("Results")
    print("="*80)

    # Extract trajectories for each agent
    for agent_idx in range(num_cars):
        agent_obs = np.array(transitions.obs[agent_idx])  # (timesteps, 6)
        agent_actions = np.array(transitions.action[agent_idx])  # (timesteps, 2)
        agent_rewards = np.array(transitions.reward[:, agent_idx])  # (timesteps,)

        # Compute statistics
        total_reward = agent_rewards.sum()
        distance = np.sum(np.sqrt(np.diff(agent_obs[:, 0])**2 + np.diff(agent_obs[:, 1])**2))
        avg_speed = np.mean(np.sqrt(agent_obs[:, 3]**2 + agent_obs[:, 4]**2))
        final_pos = agent_obs[-1, :2]

        print(f"\nAgent {agent_idx}:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Distance traveled: {distance:.2f} m")
        print(f"  Average speed: {avg_speed:.3f} m/s")
        print(f"  Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        print(f"  Avg target velocity: {np.mean(agent_actions[:, 0]):.3f}")
        print(f"  Avg target steer: {np.mean(agent_actions[:, 1]):.3f}")

    # Episode statistics
    print(f"\nEpisode statistics:")
    print(f"  Episode length: {transitions.step_count} steps")
    print(f"  Terminated: {transitions.terminated}")
    print(f"  Truncated: {transitions.truncated}")
    print(f"  Total episode reward (all agents): {np.array(transitions.episode_reward[-1]).sum():.2f}")

    # ========================================================================
    # Save Data
    # ========================================================================
    print("\n" + "="*80)
    print("Saving Data")
    print("="*80)

    # Save trajectories in visualization format
    all_trajectories = []
    for i in range(num_cars):
        traj = np.array(transitions.obs[i])
        all_trajectories.append(traj)

    # Stack: (timesteps, num_agents * 6)
    viz_data = np.concatenate(all_trajectories, axis=1)
    np.save('racing_example_data.npy', viz_data)

    print(f"  Saved trajectory data to 'racing_example_data.npy'")
    print(f"  Shape: {viz_data.shape}")
    print(f"  Format: (timesteps, num_agents * [x, y, psi, vx, vy, omega])")

    # Save transitions for RL training
    np.save('racing_example_transitions.npy', {
        'obs': [np.array(o) for o in transitions.obs],
        'actions': [np.array(a) for a in transitions.action],
        'rewards': np.array(transitions.reward),
        'episode_rewards': np.array(transitions.episode_reward),
    })

    print(f"  Saved transitions to 'racing_example_transitions.npy'")

    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)

    return transitions


if __name__ == "__main__":
    transitions = main()

    print("\nTo visualize the race:")
    print("  from car_jax.visualize import plot_multi_agent_trajectory")
    print("  plot_multi_agent_trajectory(transitions, num_agents=3, save_path='race.png')")
