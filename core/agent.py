from typing import Dict, Any
from core.environment import Environment
from core.perception import Perception
from core.world_model import WorldModel
from tinygrad.tensor import Tensor
from core.actor import Actor


def no_terminate() -> bool:
    """
    Checks if the current agent & eenvironment should terminate.
    Checks for: User input, terminal failure
    """
    return True

class Agent:
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.config = config
        self.environment = Environment(config)
        self.worldmodel = WorldModel(config)
        self.perception = Perception(config)
        #self.cost = Cost()
        self.actor = Actor(config["actor"])
        #self.short_term_memory = ShortTermMemory()

    def act(self, observation):
        observation = Tensor(observation)
        observation = observation.reshape(1, 3, 64, 64)
        z = self.perceive(observation)
        z_t1 = self.worldmodel.predict(z, self.initial_state)
        a = self.actor.act(z, z_t1)
        x_hat = self.perception.decode(z)
        print(x_hat.shape)
        raise Exception  
    
    def train(self):
        if self.config["meta"]["make_dataset"]:
            datasets = self.environment.create_datasets()
        self.perception.train(datasets["vision"])
        self.worldmodel.train(datasets["world_model"])
        self.actor.train()
        #self.cost.train()
    
    def load_weights(self):
        pass

    def act(self, obs):
        latent = self.perception.inference(obs)
        pred, wm_state = self.worldmodel.predict(latent)
        raise Exception("single world model step")
        action = self.actor(latent, pred)
        return action, wm_state

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

