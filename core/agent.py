from core.environment import Environment
from core.perception import Perception
from core.world_model import WorldModel
from tinygrad.tensor import Tensor
from core.actor import Actor


def no_terminate():
    """
    Checks if the current agent & eenvironment should terminate.
    Checks for: User input, terminal failure
    """
    return True

class Agent:
    def __init__(self, config):
        self.config = config
        self.environment = Environment(config["data_config"])
        self.worldmodel = WorldModel(config["world_model"])
        self.initial_state = self.worldmodel.initial_state()
        self.perception = Perception(config["perception"])
        #self.cost = Cost()
        self.actor = Actor(config["actor"])
        #self.short_term_memory = ShortTermMemory()
        

    def perceive(self, observation):
        z = self.perception.encode(observation)
        return z
    
    def world_model(self, z):
        pass

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
        self.world_model.train(datasets["world_model"])
        self.actor.train()
        self.cost.train()
    
    def load_weights(self):
        pass

    def execute(self):
        if self.config["meta"]["train"]:
            self.train()
        else:
            self.load_weights()

