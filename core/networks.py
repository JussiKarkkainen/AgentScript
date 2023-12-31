from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class NeuralNetwork:
    def __init__(self, python_def, config):
        local_scope = {}
        exec(python_def, globals(), local_scope)
        self.network_class = local_scope['Network'](config)
    
    def __call__(self, x):
        return self.network_class(x)
