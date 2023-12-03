import core.nn_models as models
from configs.config import TestCarRacingConfig
import numpy as np
from tinygrad.tensor import Tensor

def test_vae_input_shape():
    vae = models.SimpleVarAutoEnc(TestCarRacingConfig["perception"])
    sample_input = Tensor(np.random.randn(1, 3, 64, 64))
    z, mu, sigma = vae.encode(sample_input)
    assert z.shape == (1, 32)


def test_vae_output_shape():
    vae = models.SimpleVarAutoEnc(TestCarRacingConfig["perception"])
    z = Tensor.randn(1, 32)
    out = vae.decode(z)
    assert out.shape == (1, 3, 64, 64)
