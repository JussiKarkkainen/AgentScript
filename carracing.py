import gymnasium as gym
from core.agent import Agent
import cv2

CarRacingConfig = {
            "perception": {
                "vision_resolution": (1, 3, 64, 64),
                "hidden_dim": 32,
                "enc_kernel_size": 4,
                "dec_kernel_size": 5,
                "text": None,
                "audio": None
            },
            "world_model": {
                "input_shape": 32,
                "hidden_size": 256
            },
            "actor": {
                "input_size": 64,
                "action_space": "discrete",
                "actions": 5
            }
}

def preprocess(observation):
    observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_AREA)
    return observation

if __name__ == "__main__":
    env = gym.make("CarRacing-v2")
    observation, info = env.reset()
    agent = Agent(CarRacingConfig)
    for _ in range(1000):
        action = agent.act(preprocess(observation))
        raise Exception
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
          observation, info = env.reset()
    
    env.close()
