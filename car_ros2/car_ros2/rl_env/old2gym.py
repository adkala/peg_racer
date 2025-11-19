import gymnasium as gym
from gymnasium import spaces

class OldToGymnasium(gym.Env):
 
    def __init__(self, old_env):
        super().__init__()

        self._env = old_env
        old_obs = old_env.observation_space
        old_act = old_env.action_space

        self.observation_space = spaces.Box(low=old_obs.low, 
                                            high=old_obs.high,
                                            shape=old_obs.shape, 
                                            dtype=old_obs.dtype)
        
        self.action_space = spaces.Box(low=old_act.low, 
                                       high=old_act.high,
                                       shape=old_act.shape, 
                                       dtype=old_act.dtype)
        
        self.oracle_observation_space = spaces.Box(low=old_env.oracle_observation_space.low,
                                                   high=old_env.oracle_observation_space.high,
                                                  shape=old_env.oracle_observation_space.shape,
                                                  dtype=old_env.oracle_observation_space.dtype)
        
        self.metadata = getattr(old_env, "metadata", {})

    def reset(self, *, seed=None, options=None):

        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)

        obs, info = self._env.reset()
        return obs, info

    def step(self, action, collect_all=False):
        out = self._env.step(action, collect_all)

        if len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False

        else:
            obs, reward, terminated, truncated, info = out

        return obs, reward, terminated, truncated, info
    
    def set_opponents(self, opponents_dict): 
        
        return self._env.set_opponents(opponents_dict)

    def render(self, *a, **k): 
        return self._env.render(*a, **k)

    def close(self):
        return self._env.close()