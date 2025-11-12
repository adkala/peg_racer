"""
Visualization utilities for car_jax trajectories
Compatible with existing visualization code in car_ros2
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def euler_matrix(roll, pitch, yaw):
    """Create rotation matrix from Euler angles"""
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, 0],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0],
        [-sp, cp*sr, cp*cr, 0],
        [0, 0, 0, 1]
    ])

from car_jax.env import CarEnv
from car_jax.sim import Sim, collect_factory
from car_jax.sim.agent import IdentityAgent
from car_jax.sim.reward import ProgressReward
from car_jax.policy import PolicyAggregator, ForwardPolicy, RandomPolicy


# Car dimensions (from existing visualization code)
LF = 0.06
LR = 0.12
L = LF + LR


def draw_pose(pose, color='r', linewidth=2):
    """Draw a car at given pose (x, y, psi)"""
    px, py, psi = pose
    pts = np.array([
        [LF, L/3],
        [LF, -L/3],
        [-LR, -L/3],
        [-LR, L/3],
        [LF, L/3],
    ])
    # transform to world frame
    R = euler_matrix(0, 0, psi)[:2, :2]
    pts = np.dot(R, pts.T).T
    pts += np.array([px, py])
    plt.plot(pts[:, 0], pts[:, 1], color, linewidth=linewidth)


def transitions_to_trajectory(transitions, agent_idx=0):
    """
    Convert transitions to trajectory format compatible with visualization

    Args:
        transitions: Transition object from collect_factory
        agent_idx: Index of agent to extract

    Returns:
        trajectory: numpy array of shape (timesteps, 6) [x, y, psi, vx, vy, omega]
    """
    # Extract observations for the agent
    agent_obs = transitions.obs[agent_idx]  # Shape: (timesteps, 6)

    # Convert to numpy
    trajectory = np.array(agent_obs)

    return trajectory


def plot_multi_agent_trajectory(transitions, num_agents, save_path='multi_agent_trajectory.png'):
    """
    Plot trajectories for multiple agents

    Args:
        transitions: Transition object from collect_factory
        num_agents: Number of agents
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 10))

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(num_agents):
        trajectory = transitions_to_trajectory(transitions, agent_idx=i)

        # Plot trajectory
        color = colors[i % len(colors)]
        plt.plot(trajectory[:, 0], trajectory[:, 1], color, label=f'Agent {i}', alpha=0.7)

        # Draw initial and final poses
        draw_pose(trajectory[0, :3], color=color, linewidth=1)
        draw_pose(trajectory[-1, :3], color=color, linewidth=2)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Multi-Agent Car Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

    return plt.gcf()


def save_trajectory_for_ros2_viz(transitions, num_agents, save_path='trajectory_data.npy'):
    """
    Save trajectory data in format compatible with ROS2 visualization

    Format: (timesteps, features_per_agent * num_agents)
    where features = [x, y, psi, vx, vy, omega] for each agent

    Args:
        transitions: Transition object from collect_factory
        num_agents: Number of agents
        save_path: Path to save the numpy file
    """
    # Collect trajectories for all agents
    all_trajectories = []
    for i in range(num_agents):
        traj = transitions_to_trajectory(transitions, agent_idx=i)
        all_trajectories.append(traj)

    # Stack along features dimension
    # Result: (timesteps, num_agents * 6)
    stacked = np.concatenate(all_trajectories, axis=1)

    # Save
    np.save(save_path, stacked)
    print(f"Saved trajectory data to {save_path}")
    print(f"Shape: {stacked.shape}")
    print(f"Format: Each row has {num_agents} agents x [x, y, psi, vx, vy, omega]")

    return stacked


def create_racing_visualization_demo():
    """
    Create a demo visualization showing 2 cars racing
    """
    print("\n" + "="*80)
    print("Creating Racing Visualization Demo")
    print("="*80)

    # Create environment with 2 cars
    env = CarEnv(num_cars=2, dt=0.05, max_steps=100)

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
    sim = Sim(env=env, agents=agents, reward=rewards, max_steps=100)

    # Create policies - one goes faster and turns left, one slower and turns right
    policies = [
        ForwardPolicy(target_vel=0.9, target_steer=0.05),
        ForwardPolicy(target_vel=0.7, target_steer=-0.05),
    ]

    aggregated_policy = PolicyAggregator(policies)

    # Collect data
    collect = collect_factory(
        sim_reset=sim.reset,
        sim_step=sim.step,
        pi=aggregated_policy,
        record_indices=[0, 1],
        timesteps=100,
        return_obs=False,
        deterministic=True,
    )

    key = jax.random.key(42)
    policy_context = aggregated_policy.init_context(jax.random.key(43))

    print("Collecting race data...")
    transitions = collect(key, policy_context)

    print("Creating visualization...")
    fig = plot_multi_agent_trajectory(transitions, num_agents=2, save_path='racing_demo.png')

    print("Saving data for ROS2 visualization...")
    data = save_trajectory_for_ros2_viz(transitions, num_agents=2, save_path='racing_data.npy')

    # Print statistics
    for i in range(2):
        traj = transitions_to_trajectory(transitions, agent_idx=i)
        distance = np.sum(np.sqrt(np.diff(traj[:, 0])**2 + np.diff(traj[:, 1])**2))
        final_speed = np.sqrt(traj[-1, 3]**2 + traj[-1, 4]**2)
        print(f"\nAgent {i} stats:")
        print(f"  Total distance: {distance:.3f} m")
        print(f"  Final speed: {final_speed:.3f} m/s")
        print(f"  Final position: ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")

    print("\n✓ Demo complete!")
    return transitions, data


if __name__ == "__main__":
    # Run demo
    transitions, data = create_racing_visualization_demo()

    print("\n" + "="*80)
    print("Visualization files created:")
    print("  - racing_demo.png: Plot of trajectories")
    print("  - racing_data.npy: Data for ROS2/external visualization")
    print("="*80)
