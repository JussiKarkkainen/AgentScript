import os
import torch

class WeightManager:
    def __init__(self, directory: str = "weights"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_weights(self, model, filename: str):
        filepath = os.path.join(self.directory, filename)
        torch.save(model.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, model, filename: str):
        filepath = os.path.join(self.directory, filename)
        if os.path.exists(filepath):
            model.load_state_dict(torch.load(filepath))
            print(f"Model weights loaded from {filepath}")
        else:
            raise Exception(f"No weights found at {filepath}")

