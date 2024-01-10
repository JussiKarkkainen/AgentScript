from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class NeuralNetwork:
    def __init__(self, python_def, config):
        local_scope = {}
        exec(python_def, globals(), local_scope)
        self.network_class = local_scope['Network'](config)
    
    def parameters(self):
        return nn.state.get_parameters(self.network_class)

    def __call__(self, x):
        if type(x) != Tensor:
            x = Tensor(x, requires_grad=False)
        return self.network_class(x)
