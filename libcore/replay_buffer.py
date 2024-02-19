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

    def sample(self, batch_size: int, num_unroll_steps: int, td_steps: int) -> List[Experience]:
        episodes = random.sample(self.buffer, batch_size)
        positions = [(episode, random.randint(0, max(0, (len(episode.actions) - num_unroll_steps - 1)))) for episode in episodes]
        batch = [(episode.obs[i], episode.actions[i:i+num_unroll_steps], self.make_target(i, num_unroll_steps, td_steps, episode)) for episode, i in positions]
        return batch
    
    def make_target(self, i, num_unroll_steps, td_steps, episode):
        targets = []
        for idx in range(i, i+num_unroll_steps+1):
            bootstrap_index = idx + td_steps
            if bootstrap_index < len(episode.root_values):
                value = episode.root_values[bootstrap_index] * config["discount"]**td_steps
            else:
                value = 0
            for i, reward in enumerate(episode.rewards[idx:bootstrap_index]):
                value += reward * config["discount"]**i
            if idx > 0 and idx <= len(episode.rewards):
                last_reward = episode.rewards[idx - 1]
            else:
                last_reward = 0
            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                targets.append((0, last_reward, []))
        return targets

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

