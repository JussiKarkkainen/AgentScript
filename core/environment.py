from typing import Dict
import gymnasium as gym
import numpy as np
import os

implemented_envs = {"gym": ["CarRacing-v2"]}

class Environment:
    def __init__(self, config):
        self.config = config
        self.make_env() 

    def create_perception_dataset(self) -> str:
        """
        Collects self.config["data_config"]["max_frames"] frames from self.config["data_config"]["num_episodes"]
        episodes on the specified environment. Returns the directory path of the dataset files. Dataset is saved 
        as a numpy array and stored using np.savez_compressed
        """
        frames = 0
        for episode in range(self.config["data_config"]["num_episodes"]):
            obs, info = self.env.reset()
            observations = []
            actions = []
            for frame in range(self.config["data_config"]["max_frames"]):
                observations.append(obs)
                action = self.env.action_space.sample()
                actions.append(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break

            frames += frame + 1
            observations_np = np.array(observations, dtype=np.uint8)
            actions_np = np.array(actions, dtype=np.float16)
            file_path = self.config["meta"]["dataset_path"]+"vision/"
            np.savez_compressed(file_path+f"episode{episode}", obs=observations_np, action=actions_np)
        return file_path 


    def create_datasets(self) -> Dict[str, str]:
        if os.getenv("DATASET"):
            perception_dataset_path = self.create_perception_dataset() 
        dataset_paths = {"perception": f"{self.config['meta']['dataset_path']}"}
        return dataset_paths
    
    def make_env(self):
        if self.config["data_config"]["env"] not in implemented_envs["gym"]:
            raise NotImplementedError("The Environment you are using is not yet supported")
        self.env = gym.make(self.config["data_config"]["env"], render_mode="rgb_array")
