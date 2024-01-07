from typing import Dict
from collections import deque
import random
from tinygrad.tensor import Tensor

class ReplayBuffer:
    def __init__(self, config: Dict[str, int]):
        self.config = config
        if not self.config["episodic"]:
            self.buffer = deque(maxlen=self.config["capacity"])
        else:
            self.buffer = []

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        sampled_transitions = random.sample(self.buffer, batch_size)
        batch = {
            'states': Tensor([t.state for t in sampled_transitions]),
            'actions': Tensor([t.action for t in sampled_transitions]),
            'rewards': Tensor([t.reward for t in sampled_transitions]),
            'next_states': Tensor([t.next_state for t in sampled_transitions]),
            'dones': Tensor([t.done for t in sampled_transitions])
        }
        return batch
    
    def __len__(self):
        return len(self.buffer)

