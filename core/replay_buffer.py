from typing import Dict
from collections import deque

class ReplayBuffer:
    def __init__(self, config: Dict[str, int]):
        self.config = config
        self.buffer = deque(maxlen=self.config["capacity"])
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

