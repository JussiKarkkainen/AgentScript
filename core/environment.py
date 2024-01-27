from typing import Dict, Any, Tuple
import gymnasium as gym
import numpy as np
from tinygrad.tensor import Tensor

implemented_envs = {"gym": ["CarRacing-v2", "CartPole-v1", "Pendulum-v1", "LunarLander-v2"]}

def preprocess(obs):
    raise NotImplementedError

class Environment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = self.make_env() 

    def step(self, action: Any) -> Tuple[Any]:
        if isinstance(action, Tensor) or isinstance(action, np.ndarray):
            action = action.item()
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info

    def make_env(self):
        if self.config["name"] not in implemented_envs["gym"]:
            raise NotImplementedError("The Environment you are using is not yet supported")
        return gym.make(self.config["name"], render_mode="rgb_array")

    def init(self) -> np.ndarray:
        obs, info = self.env.reset()
        if self.config["preprocess"]:
            obs = preprocess(obs)
        return obs
