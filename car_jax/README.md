# Car JAX - JAX-based Multi-Agent Car Racing Framework

A high-performance, JAX-based framework for simulating single and multi-agent car racing scenarios. This framework follows the design patterns from `perl_jax` and `pegrid_jax` to provide a clean, functional approach to multi-agent reinforcement learning research.

## Features

- **Pure JAX Implementation**: Fully JIT-compatible for maximum performance
- **Multi-Agent Support**: Dynamic number of agents with flexible observation/action spaces
- **Modular Design**: Separate Agent, Reward, and Simulation components
- **Data Collection**: Built-in collection factories for RL training data
- **Visualization**: Compatible with existing ROS2/Unity visualizers

## Architecture

The framework consists of several key components:

### 1. Environment (`car_jax.env`)

- **CarEnv**: JAX-based car environment using Dynamic Bicycle Model
- **Observation**: Structured observation with car state and metadata
- **CarEnvState**: Internal state including positions, velocities, and action buffer

### 2. Simulation (`car_jax.sim`)

- **Agent**: Transforms observations between environment and policy spaces
  - `IdentityAgent`: Pass-through agent for single observations
  - `RelativePositionAgent`: Provides relative position observations
- **Reward**: Computes rewards based on state transitions
  - `VelocityReward`: Rewards based on forward velocity
  - `ProgressReward`: Rewards based on distance traveled
  - `SurvivalReward`: Constant reward per timestep
  - `CompositeReward`: Weighted combination of multiple rewards
- **Sim**: Combines environment, agents, and rewards into a unified simulation

### 3. Policy (`car_jax.policy`)

- **Policy**: Base class for all policies
- **PolicyAggregator**: Manages multiple policies for multi-agent scenarios
- **RandomPolicy**: Samples random actions from action space
- **ForwardPolicy**: Simple policy that drives forward
- **ConstantPolicy**: Returns fixed action
- **TrajCategoricalPolicy**: Per-trajectory categorical sampling
- **DeterministicWrapper**: Wraps policy for deterministic execution

### 4. Data Collection (`car_jax.sim.collect`)

- **collect_factory**: Creates single-environment collection function
- **batch_collect_factory**: Creates parallel multi-environment collection
- **Transition**: Stores environment transitions with observations, actions, rewards

## Installation

```bash
# Install dependencies
pip install jax jaxlib flax jaxtyping gymnasium numpy matplotlib

# Add to PYTHONPATH
export PYTHONPATH=/path/to/peg_racer:/path/to/peg_racer/car_dynamics:$PYTHONPATH
```

## Quick Start

### Single Agent Simulation

```python
import jax
from car_jax.env import CarEnv
from car_jax.sim import Sim
from car_jax.sim.agent import IdentityAgent
from car_jax.sim.reward import ProgressReward
from car_jax.policy import ForwardPolicy

# Create environment
env = CarEnv(num_cars=1, dt=0.05, max_steps=100)

# Create agent and reward
agent = IdentityAgent(agent_idx=0)
reward_fn = ProgressReward(agent_idx=0, scale=1.0)

# Create simulation
sim = Sim(env=env, agents=[agent], reward=[reward_fn], max_steps=100)

# Create policy
policy = ForwardPolicy(target_vel=0.8, target_steer=0.0)

# Run simulation
key = jax.random.key(42)
obs, obs_for_pi, agent_context, sim_context = sim.reset(key)

for _ in range(100):
    action, policy_context = policy(obs_for_pi[0], {}, key)
    obs, reward, terminated, truncated, obs_for_pi, agent_context, sim_context = sim.step(
        obs, [action], agent_context, sim_context
    )
    if terminated or truncated:
        break
```

### Multi-Agent Simulation

```python
from car_jax.policy import PolicyAggregator

# Create environment with multiple cars
env = CarEnv(num_cars=3, dt=0.05, max_steps=100)

# Create agents for each car
agents = [
    IdentityAgent(agent_idx=0),
    IdentityAgent(agent_idx=1),
    RelativePositionAgent(agent_idx=2, other_agent_indices=[0, 1]),
]

# Create rewards
rewards = [
    ProgressReward(agent_idx=0),
    ProgressReward(agent_idx=1),
    ProgressReward(agent_idx=2),
]

# Create simulation
sim = Sim(env=env, agents=agents, reward=rewards, max_steps=100)

# Create policies
policies = [
    ForwardPolicy(target_vel=0.9, target_steer=0.05),
    ForwardPolicy(target_vel=0.7, target_steer=-0.05),
    RandomPolicy(env.action_space),
]

aggregated_policy = PolicyAggregator(policies)

# Run multi-agent simulation
key = jax.random.key(42)
obs, obs_for_pi, agent_context, sim_context = sim.reset(key)
policy_context = aggregated_policy.init_context(key)

for _ in range(100):
    actions, policy_context = aggregated_policy(obs_for_pi, policy_context, key)
    obs, rewards, terminated, truncated, obs_for_pi, agent_context, sim_context = sim.step(
        obs, actions, agent_context, sim_context
    )
    if terminated or truncated:
        break
```

### Data Collection for RL

```python
from car_jax.sim import collect_factory

# Create collection function
collect = collect_factory(
    sim_reset=sim.reset,
    sim_step=sim.step,
    pi=aggregated_policy,
    record_indices=[0, 1, 2],  # Record all agents
    timesteps=1000,
    deterministic=False,
)

# Collect data
key = jax.random.key(42)
policy_context = aggregated_policy.init_context(key)
transitions = collect(key, policy_context)

# Access collected data
print(f"Observations: {len(transitions.obs)}")  # One per agent
print(f"Actions: {len(transitions.action)}")    # One per agent
print(f"Rewards: {transitions.reward.shape}")   # (timesteps, num_agents)
```

### Batch Collection (Parallel Environments)

```python
from car_jax.sim import batch_collect_factory

# Create batch collection function
batch_collect = batch_collect_factory(
    sim_reset=sim.reset,
    sim_step=sim.step,
    pi=policy,
    record_indices=[0],
    total_timesteps=10000,
    num_envs=10,  # Run 10 environments in parallel
    deterministic=False,
)

# Collect data in parallel
transitions = batch_collect(key, policy_context)
print(f"Total timesteps collected: {transitions.reward.shape[0]}")
```

### Visualization

```python
from car_jax.visualize import (
    plot_multi_agent_trajectory,
    save_trajectory_for_ros2_viz,
    create_racing_visualization_demo,
)

# Create visualization
plot_multi_agent_trajectory(transitions, num_agents=3, save_path='race.png')

# Save for ROS2/Unity visualization
save_trajectory_for_ros2_viz(transitions, num_agents=3, save_path='race_data.npy')

# Or run the demo
transitions, data = create_racing_visualization_demo()
```

## Examples

Run the test suite:

```bash
PYTHONPATH=/path/to/peg_racer:/path/to/peg_racer/car_dynamics:$PYTHONPATH python car_jax/test_sim.py
```

Run the visualization demo:

```bash
PYTHONPATH=/path/to/peg_racer:/path/to/peg_racer/car_dynamics:$PYTHONPATH python car_jax/visualize.py
```

## Output Format

The framework outputs data in formats compatible with existing visualizers:

### Transition Format

```python
@struct.dataclass
class Transition:
    obs: PyTree[Array]           # Observations (one per agent)
    action: PyTree[Array]         # Actions (one per agent)
    next_obs: PyTree[Array]       # Next observations
    reward: Array                 # Rewards (timesteps, num_agents)
    episode_reward: Array         # Cumulative rewards
    terminated: bool              # Episode termination flag
    truncated: bool               # Episode truncation flag
    step_count: int               # Current timestep
    agent_win: Array              # Which agents have won
```

### Visualization Format

Trajectories are saved as numpy arrays with shape `(timesteps, features_per_agent * num_agents)`
where features = `[x, y, psi, vx, vy, omega]` for each agent.

## Design Principles

1. **Functional Programming**: All components are stateless functions operating on immutable data
2. **JAX-Compatible**: Fully JIT-compilable for maximum performance
3. **Composable**: Mix and match agents, rewards, and policies
4. **Type-Safe**: Uses `jaxtyping` for array shape annotations
5. **Extensible**: Easy to add new agents, rewards, and policies

## Extending the Framework

### Custom Agent

```python
from car_jax.sim import Agent

class MyAgent(Agent):
    def obs_env2pi(self, obs, context):
        # Transform observation for your policy
        obs_for_pi = jnp.array([...])
        return obs_for_pi, context

    def action_pi2env(self, action, context):
        # Transform policy action for environment
        env_action = jnp.array([...])
        return env_action, context
```

### Custom Reward

```python
from car_jax.sim import Reward

class MyReward(Reward):
    def __call__(self, obs, action, next_obs, context):
        # Compute custom reward
        reward = jnp.array(0.0)
        return reward, context
```

### Custom Policy

```python
from car_jax.policy import Policy

class MyPolicy(Policy):
    def __call__(self, obs_for_pi, context, key):
        # Sample action from your policy
        action = jnp.array([...])
        return action, context

    def deterministic_sample(self, obs_for_pi, context):
        # Deterministic version (optional)
        action = jnp.array([...])
        return action, context
```

## Testing

The framework includes comprehensive tests in `test_sim.py`:

- ✓ Single agent simulation
- ✓ Multi-agent simulation
- ✓ Data collection
- ✓ Batch collection
- ✓ Visualization output

All tests pass successfully!

## Performance

The framework leverages JAX's JIT compilation for high performance:
- Pure JAX operations enable GPU/TPU acceleration
- Batch collection runs multiple environments in parallel
- Functional design eliminates Python overhead

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{car_jax2024,
  title={Car JAX: Multi-Agent Car Racing Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/peg_racer}
}
```

## License

[Your License Here]

## Acknowledgments

This framework follows design patterns from:
- `perl_jax`: Policy Evaluation and Reinforcement Learning in JAX
- `pegrid_jax`: Multi-agent gridworld environments in JAX

The car dynamics are based on the Dynamic Bicycle Model implementation in `car_dynamics`.
