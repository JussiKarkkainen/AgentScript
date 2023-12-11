from typing import Dict
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import MLP

class Actor:
    def __init__(self, config: Dict[str, Dict[str, str]]):
        self.actor = MLP(config)

    def action(self, z: Tensor, z_t1: Tensor) -> Tensor:
        print(z.shape, z_t1.shape)
        combined = z.cat(z_t1, dim=-1)
        return self.actor(combined)

