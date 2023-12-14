from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import MDNLSTM

class WorldModel:
    def __init__(self, config):
        self.config = config
        self.model = MDNLSTM(config["world_model"]) 
    
    def train(self):
        pass

    def loss(self, logmix, mean, logstd, y):
        y = y.unsqueeze(1)  # Adjust the shape of y for broadcasting
        v = logmix + log_normal_pdf(y, mean, logstd)
        v = torch.logsumexp(v, dim=1, keepdim=True)
        return -torch.mean(v)

    def predict(self, x):
        # Add batch dim
        x = x.unsqueeze(0)
        out = self.model(x)
        return out 



