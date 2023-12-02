import gymnasium as gym
import numpy as np

implemented_envs = {"gym": ["CarRacing-V2"]}

class Environment:
    def __init__(self, config):
        self.config = config
        

    def create_perception_dataset(self):
        raise Exception("create_perception_dataset()")
        frames = 0
        for episode in range(self.config["num_episodes"]):
            obs, info = self.env.reset()
            observations = []
            actions = []
            for frame in range(self.config["max_frames"]):
                observations.append(obs)
                action = self.env.action_space.sample()
                actions.append(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break

            frames += frame + 1
            observations_np = np.array(observations, dtype=np.uint8)
            actions_np = np.array(actions, dtype=np.float16)
            np.savez_compressed(config["perception"]["data_path"], obs=observations_np, action=actions_np)


    def create_datasets(self):
        # perception_dataset_path = self.create_perception_dataset() 
        dataset_paths = {"perception": "self.config['meta']['dataset_path']+'/vision.npz'"}
        return dataset_paths
    
    def make_env(self):
        if self.config["env"] not in implemented_envs:
            raise NotImplementedError("The Environment you are using is not yet supported")
        self.env = gym.make(self.config["env"], render_mode="rgb_array")
