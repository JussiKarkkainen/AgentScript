import gymnasium as gym
from typing import List, Any, Tuple


class Environment:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.done = False

    def reset(self):
        state, info = self.env.reset()
        return state

    def step(self, action: Any):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        return next_state, reward, self.done

    def is_terminal(self):
        return self.done

    def action_space(self):
        return self.env.action_space

    def obs_space(self):
        return self.env.observation_space
