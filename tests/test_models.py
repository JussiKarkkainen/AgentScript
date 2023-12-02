import core.nn_models as models
from configs.config import TestCarRacingConfig
import numpy as np
from tinygrad.tensor import Tensor

def test_vae_shapes():
    vae = models.SimpleVarAutoEnc(TestCarRacingConfig["perception"])
    sample_input = Tensor(np.random.randn(1, 3, 64, 64))
    z = vae.encode(sample_input)
    assert z.shape == (1, 32)
