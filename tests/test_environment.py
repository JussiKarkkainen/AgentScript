from core.environment import Environment
from configs.config import TestCarRacingConfig
import os

def test_create_datasets():
    env = Environment(TestCarRacingConfig)
    assert type(env.create_datasets()) == dict

def test_num_dataset_files():
    assert len(os.listdir(TestCarRacingConfig["meta"]["dataset_path"]+"vision/")) == TestCarRacingConfig["data_config"]["num_episodes"]

