import gymnasium as gym
from core.agent import Agent
from configs.config import CarRacingConfig, TestCarRacingConfig
import os

if __name__ == "__main__":
    agent = Agent(CarRacingConfig if not os.getenv("DEBUG") else TestCarRacingConfig)
    agent.execute()

