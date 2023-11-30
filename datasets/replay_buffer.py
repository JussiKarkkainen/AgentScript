import gymnasium as gym
from collections import namedtuple

# TODO: This is a hack that needs to be rewritten
implemented_envs = ["CarRacing-V2"]

Episode = namedtuple()


class ReplayBuffer:
    def __init__(self, data_config):
        self.num_episodes = data_config["num_episodes"]
        self.env_config = data_config["env"]
        self.policy = data_config["policy"]

        self.buffer = []
    
    def make_env(self):
        if self.env_config not in implemented_envs:
            raise NotImplementedError("The Environment you are using is not yet supported")
        self.env = gym.make(self.env_config, render_mode="rgb_array")

    def collect_epochs(self):
        for i in range(self.num_episodes):
            observation, info = self.env.reset() 
            for frame in range(MAX_FRAMES):
                action = self.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
            
                if terminated or truncated:
                    observation, info = env.reset()
            env = 
