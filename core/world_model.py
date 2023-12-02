from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from models.rnnt import LSTM


class WorldModel:
    def __init__(self, config):
        self.config = config
        #self.lstm = LSTMCell(config["input_shape"], config["hidden_size"], 0.1)
        self.lstm = None 

    def initial_state(self):
        return Tensor.zeros(self.config["input_shape"], self.config["hidden_size"])

    def predict(self, x, h):
        print(x.shape, h.shape)
        for obs in x:
            out = self.lstm(x, h)
        return out



