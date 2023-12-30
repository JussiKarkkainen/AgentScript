from typing import Dict, Any, Tuple, List
import utils
from core.environment import Environment
from core.replay_buffer import ReplayBuffer
from core.perception import Perception
from core.world_model import WorldModel
from core.actor import Actor
from core.agent import Agent
from tinygrad.tensor import Tensor
import numpy as np

def builder(config: List[Dict[str, dict[str, Any]]], python: List[str]):
    # TODO: Validate the configuration, Use the validate_config() function in config_parser.py

    modules = [(eval(list(conf.keys())[0]), list(conf.keys())[0], list(conf.values())[0]) for conf in config]
    # module[0] = module class, module[1] = module name, module[2] = module init params
    for module in modules:
        if module[1] == "Environment":
            environment = module[0](module[2])
        elif module[1] == "ReplayBuffer":
            replay_buffer = module[0](module[2])
        elif module[1] == "Agent":
            agent = module[0](module[2])
    
    return agent, environment, replay_buffer

