from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import SimpleVarAutoEnc
from logs.wandb_logs import WandbLogger
import numpy as np
import os

def load_dataset(path: str, num_episodes: int, num_frames: int) -> np.array:
    # TODO: The input resolution is hardcoded for CarRacing example, fix this
    data = np.zeros((num_episodes, num_frames, 64, 64, 3), dtype=np.uint8)
    files = [os.path.join(os.getcwd()+"/tests/datasets/vision", file) for file in os.listdir(path)]
    idx = 0
    for episode in range(num_episodes):
        filename = files[episode]
        raw_data = np.load(filename)["obs"]
        l = len(raw_data)
        if (idx + l) > (num_episodes * num_frames):
            data = data[0:idx]
            break
        data[idx:idx+l] = raw_data
        idx += l
        if ((episode+1) % 100 == 0):
            print("loading file", episode+1)
    return data

class Perception:
    def __init__(self, config):
        self.config = config
        self.model = SimpleVarAutoEnc(config["perception"]) 
        self.optimizer = nn.optim.Adam(nn.state.get_parameters(self.model), lr=3e-4)

    def loss_fn(self, x, y_hat, mean, log_var):
        l2_loss = ((y_hat - x) ** 2).mean()
        kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum()
        return l2_loss, kl_loss, l2_loss + kl_loss

    def load_pretrained(self):
        pass

    def train(self, train_dataset_path: str):
        data = load_dataset(train_dataset_path, self.config["data_config"]["num_episodes"], self.config["data_config"]["max_frames"])
        num_batches = data[0]
        self.model.train()
        with WandbLogger(project_name="Perception Module", lr=self.config.lr, epochs=config.num_epochs) as wblogs:
            for epoch in range(self.config["train"]["num_epochs"]):
                np.random.shuffle(data)
                for batch in data:
                    self.optimizer.zero_grad()
                    obs = batch.astype(np.float) / 255.0
                    z, mu, sigma = self.model.encode(obs)
                    y_hat = self.model.decode(z)
                    l2_loss, kl_loss, loss = self.loss_fn(obs, y_hat, mu, sigma)
                    wandb.log({"L2 Loss": l2_loss, "KL Divergence": kl_loss, "VAE Loss": loss})
                    loss.backward()
                    self.optimizer.step()

    def inference(self, obs):
        # Numpy to tinygrad -> include batch dim -> swap channel dim
        obs = Tensor(obs).unsqueeze(dim=0).transpose(1, 3)
        latent, _, _ = self.model.encode(obs)
        return latent

                
    

