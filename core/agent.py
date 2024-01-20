from typing import Dict, Any, Tuple, List
import utils
from core.environment import Environment
from core.replay_buffer import ReplayBuffer
from core.perception import Perception
from core.world_model import WorldModel
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.actor import Actor
import numpy as np


class Agent:
    def __init__(self, config: Dict[str, Dict[str, Any]], python_update):
        self.config = config
        local_scope = {}
        exec(python_update, globals(), local_scope)
        self.update = local_scope['update'] 
        

