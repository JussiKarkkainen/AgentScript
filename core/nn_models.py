from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from tinygrad.nn import optim, state

#################### VAE ####################
# A Convolutional net used for Gymnasium environments like Car-Racing
# Taken from https://arxiv.org/pdf/1803.10122.pdf
class SimpleConvEnc:
    def __init__(self, input_shape, hidden_dim=32, kernel_size=4):
        # Encoder config
        assert input_shape == [1, 3, 64, 64]
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
        return z, mu, sigma

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

    def __call__(self, x):
        return self.decode(self.encode(x)[0])

#################### MLP ####################
class MLP:
    def __init__(self, config):
        self.layers = []
        for i in range(config["num_mlp_layers"]):
            self.layers.append(nn.Linear(config["mlp_input_dim"], config["mlp_hidden_dim"]))
        self.layers.append(nn.Linear(config["mlp_hidden_dim"], config["mlp_output_dim"]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        x = layers[-1].softmax()
        return x 

#################### MDNLSTM ####################
class MDNLSTM:
    def __init__(self, config):
        self.lstm = LSTM(config)
        self.mdn = MDN(config)

    def __call__(self, x):
        pass

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # TODO: These weights can be unified, which requires fewer matmuls, find out how it's equivelant to this
        weights = lambda: (Tensor.uniform(input_size, hidden_size),
                           Tensor.uniform(hidden_size, hidden_size),
                           Tensor.zeros(hidden_size))
        self.W_xi, self.W_hi, self.b_i = weights() 
        self.W_xf, self.W_hf, self.b_f = weights() 
        self.W_xo, self.W_ho, self.b_o = weights() 
        self.W_xc, self.W_hc, self.b_c = weights() 

    def __call__(self, x, hc):
        H, C = hc
        i = (x.matmul(self.W_xi) + H.matmul(self.W_hi) + self.b_i).sigmoid()
        f = (x.matmul(self.W_xf) + H.matmul(self.W_hf) + self.b_f).sigmoid()
        o = (x.matmul(self.W_xo) + H.matmul(self.W_ho) + self.b_o).sigmoid()
        c = (x.matmul(self.W_xc) + H.matmul(self.W_hc) + self.b_c).tanh()

        C = f * C + i * c
        H = o * C.tanh()
        return o, (H, C)

class LSTM:
    def __init__(self, config):
        self.config = config
        self.cells = [LSTMCell(config["input_size"], config["hidden_size"] for cell in range(config["num_cells"]]

    def __call__(self, x, h):
        pass

class MDN:
    def __init__(self):
        self.config = 
