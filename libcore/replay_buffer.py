import random
from collections import namedtuple
from typing import List, Tuple

Experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []  
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if len(args) == 1:
            # User passes a custom Experience namedtuple
            experience = args[0]
        elif len(args) == 5:
            experience = Experience(*args)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity 

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

