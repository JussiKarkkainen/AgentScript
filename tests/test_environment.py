from core.environment import Environment
from configs.config import TestCarRacingConfig

def test_environment():
    env = Environment(CarRacingConfig)
    assert type(env.create_datasets()) == dict
