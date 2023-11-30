from core.perception import Perception
from core.world_model import WorldModel
from tinygrad.tensor import Tensor
from core.actor import Actor

class Agent:
    def __init__(self, config):
        self.config = config
        self.worldmodel = WorldModel(config["world_model"])
        self.initial_state = self.worldmodel.initial_state()
        self.perception = Perception(config["perception"])
        #self.cost = Cost()
        self.actor = Actor(config["actor"])
        #self.short_term_memory = ShortTermMemory()
        
        #self.world_model.train()
        #self.perception.train()
        #self.cost.train()
        #self.actor.train()
        #self.short_term_memory.init()

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
