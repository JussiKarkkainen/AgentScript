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
            self.episode = None

    def push(self, transition):
        if not self.config["episodic"]:
            self.buffer.append(transition)
        else:
            self.episode = transition

    def sample(self, batch_size):
        if not self.config["episodic"]:
            sampled_transitions = random.sample(self.buffer, batch_size)
        if not self.config["episodic"]:
            batch = {
                'states': Tensor([t.state for t in sampled_transitions]),
                'actions': Tensor([t.action for t in sampled_transitions]),
                'rewards': Tensor([t.reward for t in sampled_transitions]),
                'next_states': Tensor([t.next_state for t in sampled_transitions]),
                'dones': Tensor([t.done for t in sampled_transitions])
            }
            return batch
        if self.config["type"] == "ActorCritic":
            batch = {
                'state': self.episode
            }
            return batch
        if self.config["type"] != "ActorCritic":
            batch = {
                'rewards': self.episode.rewards,
                'log_probs': self.episode.log_probs
            }
            return batch
    
    def __len__(self):
        return len(self.buffer) if not self.config["episodic"] else 1

