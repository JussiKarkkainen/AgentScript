from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class NeuralNetwork:
    def __init__(self, python_defs, config):
        self.networks = {}
        for i, python_def in enumerate(python_defs):
            name = list(config["network"].keys())[i]
            local_scope = {}
            exec(python_def, globals(), local_scope)
            print(config["network"][name])
            self.networks[name] = (local_scope[name](config["network"][name]))

    def parameters(self):
        return nn.state.get_parameters(self.network_class)

    def __call__(self, x):
        if type(x) != Tensor:
            x = Tensor(x, requires_grad=False)
        return self.network_class(x)
