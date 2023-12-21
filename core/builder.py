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


class Builder:
    def __init__(self, config: List[Dict[str, Dict[str, Any]]], python: List[str]):
        self.config = config
        self.python = python
        self.modules = [(eval(list(conf.keys())[0]), list(conf.keys())[0], list(conf.values())[0]) for conf in self.config]
        
        for module in self.modules:
            if module[1] == "Environment":
                self.environment = module[0](module[2])
            elif module[1] == "ReplayBuffer":
                self.replay_buffer = module[0](module[2])
            elif module[1] == "Agent":
                self.agent = module[0](module[2])


        print(self.modules)
        raise Exception
