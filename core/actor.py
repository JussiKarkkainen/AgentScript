from tinygrad.tensor import Tensor
import tinygrad.nn as nn



class Actor:
    def __init__(self, config):
        self.lin = nn.Linear(config["input_size"], config["actions"])

    def act(self, z, z_t1):
        print(z.shape, z_t1.shape)
        combined = z.cat(z_t1)
        return self.lin(combined)

