from typing import Dict
from collections import deque
import random
from tinygrad.tensor import Tensor

class ReplayBuffer:
    def __init__(self, config: Dict[str, int]):
        self.config = config
        if self.config["update_freq"] == "Batch":
            self.buffer = deque(maxlen=self.config["capacity"])
        else:
            self.episode = None

    def push(self, transition):
        if self.config["update_freq"] == "Batch":
            self.buffer.append(transition)
        else:
            self.episode = transition

    def sample(self, batch_size):
        
        if self.config["update_freq"] == "Timestep":
            batch = {
                'state': self.episode
            }
            return batch
        elif self.config["update_freq"] == "Episodic":
            batch = {
                'states': self.episode.states,
                'actions': self.episode.actions,
                'rewards': self.episode.rewards,
                'log_probs': self.episode.log_probs
            }
            return batch
        elif self.config["update_freq"] == "Batch":
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
        return len(self.buffer) if self.config["update_freq"] == "Batch" else 1

