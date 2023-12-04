from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import MLP

class Actor:
    def __init__(self, config):
        self.actor = MLP(config)

    def act(self, z, z_t1):
        print(z.shape, z_t1.shape)
        combined = z.cat(z_t1)
        return self.actor(combined)

