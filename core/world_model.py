from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import MDNLSTM

class WorldModel:
    def __init__(self, config):
        self.config = config
        self.model = MDNLSTM(config["world_model"]) 
    
    """
    def initial_state(self):
        return Tensor.zeros(self.config["world_model"]["input_size"], self.config["world_model"]["hidden_size"])
    """

    def predict(self, x):
        # Add batch dim
        x = x.unsqueeze(0)
        out, hc = self.model(x)
        return out 



