import random
from collections import namedtuple
from typing import List, Tuple

# Define a simple structure for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'done'])

class BaseReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []  
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Experience(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity 

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

