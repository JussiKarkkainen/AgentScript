from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import SimpleVarAutoEnc


class Perception:
    def __init__(self, config):
        self.model = SimpleVarAutoEnc(config) 

    def preprocess(self):
        pass

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)
    

