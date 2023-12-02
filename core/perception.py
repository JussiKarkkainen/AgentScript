from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from core.nn_models import SimpleVarAutoEnc
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

    def preprocess(self):
        pass

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def train(self, train_dataset_path: str):
        data = load_dataset(train_dataset_path, self.config["data_config"]["num_episodes"], self.config["data_config"]["max_frames"])
        print(data.shape)
    

