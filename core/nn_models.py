from tinygrad.tensor import Tensor
import tinygrad.nn as nn

# A Convolutional net used for Gymnasium environments like Car-Racing
# Taken from https://arxiv.org/pdf/1803.10122.pdf
class SimpleConvEnc:
    def __init__(self, input_shape, hidden_dim=32, kernel_size=4):
        # Encoder config
        assert input_shape == (1, 3, 64, 64)
        self.hidden_dim = hidden_dim
        B, C, H, W = input_shape
        out_shape = 32
        self.layers = []
        
        for i in range(4):
            self.layers.append(nn.Conv2d(C, out_shape, kernel_size=kernel_size, padding=0, stride=2))
            C = out_shape
            out_shape *= 2
        
        self.mu = nn.Linear(2*2*256, self.hidden_dim)
        self.sigma = nn.Linear(2*2*256, self.hidden_dim) 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        x = x.reshape(-1, 2*2*256).flatten(start_dim=1)
        mu = self.mu(x)
        sigma = (self.sigma(x) * 0.5).exp()
        z = mu + sigma * Tensor.randn(1, 32)
        return z

class SimpleConvDec:
    def __init__(self, hidden_dim, kernel_size=5):
        self.lin = nn.Linear(hidden_dim, 1024)
        in_channels = [1024, 128, 64, 32]
        out_channels = in_channels[1:]
        out_channels.append(3)
        self.layers = []
        kernel = lambda x: 5 if x <= 1 else 6
        for i in range(4):
            self.layers.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], kernel_size=kernel(i), padding=0, stride=2))

    def __call__(self, x):
        x_hat = self.lin(x)
        x_hat = x_hat.reshape(-1, 1024, 1, 1)
        for layer in self.layers[:-1]:
            x_hat = layer(x_hat).relu()
        x_hat = self.layers[-1](x_hat).sigmoid()
        return x_hat

class SimpleVarAutoEnc:
    def __init__(self, config):
        self.encoder = SimpleConvEnc(config["vision_resolution"], config["hidden_dim"], config["enc_kernel_size"])
        self.decoder = SimpleConvDec(config["hidden_dim"], config["dec_kernel_size"])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

