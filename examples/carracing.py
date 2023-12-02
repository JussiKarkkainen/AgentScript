import gymnasium as gym
from core.agent import Agent
from configs.config import CarRacingConfig
import cv2

if __name__ == "__main__":
    agent = Agent(CarRacingConfig)
    agent.execute()

