from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import MDNLSTM

class WorldModel:
    def __init__(self, config):
        self.config = config
        self.model = MDNLSTM(config) 
        #self.lstm = LSTMCell(config["input_shape"], config["hidden_size"], 0.1)
        self.lstm = None 

    def initial_state(self):
        return Tensor.zeros(self.config["input_shape"], self.config["hidden_size"])

    def predict(self, x, h):
        # Add batch dim
        x = x.unsqueeze(0)
        for obs in x:
            out, h = self.model(obs, h)
        return out, h



