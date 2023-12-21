from typing import Dict, Any, Tuple, List
import utils
from core.environment import Environment
from core.replay_buffer import ReplayBuffer
from core.perception import Perception
from core.world_model import WorldModel
from tinygrad.tensor import Tensor
from core.actor import Actor
import numpy as np

class Agent:
    def __init__(self, config: Dict[str, Dict[str, Any]], python: List[str]):
        self.config = config
        self.python = python
        print(config)
        raise Exception
        self.environment = Environment(config)
        self.worldmodel = WorldModel(config)
        self.perception = Perception(config)
        #self.cost = Cost()
        self.actor = Actor(config["actor"])
        #self.short_term_memory = ShortTermMemory()

    def train(self):
        if self.config["meta"]["make_dataset"]:
            datasets = self.environment.create_datasets()
        self.perception.train(datasets["vision"])
        self.worldmodel.train(datasets["world_model"])
        self.actor.train()
        #self.cost.train()
    
    def load_weights(self):
        pass

    def act(self, obs: np.ndarray) -> Tensor:
        latent = self.perception.inference(obs)
        pred = self.worldmodel.predict(latent)
        action = self.actor.action(latent, pred)
        raise Exception("single actor step")
        return action

    def execute(self):
        '''
        ----------- For testing inference -----------
        if self.config["meta"]["train"]:
            self.train()
        self.load_weights()
        '''
        obs = self.environment.init()
        # wm_state = self.worldmodel.initial_state()
        terminate = None
        while no_terminate() and not terminate:
            #action, wm_state = self.act(obs)
            action = self.act(obs)
            raise Exception("single inference step")
            obs, reward, terminate = self.environment.step(action)
        
        self.environment.shutdown()
        exit()

def no_terminate() -> bool:
    """
    Checks if the current agent & eenvironment should terminate.
    Checks for: User input, terminal failure
    """
    return True
