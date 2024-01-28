from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class NeuralNetwork:
    def __init__(self, python_defs, config):
        self.networks = {}
        
        for i, python_def in enumerate(python_defs):
            name = list(config["networks"].keys())[i]
            local_scope = {}
            exec(python_def, globals(), local_scope)
            self.networks[name] = (local_scope[name](config["networks"][name]))

    def parameters(self, network_name):
        return nn.state.get_parameters(self.networks[network_name])

    def __call__(self, name, x):
        if type(x) != Tensor:
            x = Tensor(x, requires_grad=False)
        return self.networks[name](x)
