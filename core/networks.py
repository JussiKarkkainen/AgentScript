from tinygrad.tensor import Tensor

class NeuralNetwork:
    def __init__(self, python_def):
        self.network_class = eval(python_def)
    
    def __call__(self, x):
        return self.network_class(x)
