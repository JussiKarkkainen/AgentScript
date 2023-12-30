from tinygrad.tensor import Tensor

class NeuralNetwork:
    def __init__(self, python_def):
        local_scope = {}
        exec(python_def, globals(), local_scope)
        self.network_class = local_scope['Network'] 
    
    def __call__(self, x):
        return self.network_class(x)
