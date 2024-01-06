import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.001
gamma = 0.99

# Environment
env = gym.make('CartPole-v1')

# Policy Network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Function to calculate Q-values
def calc_qvals(rewards):
    R = 0.0
    qvals = []
    for r in reversed(rewards):
        R = r + gamma * R
        qvals.insert(0, R)
    return qvals

# REINFORCE algorithm
def reinforce():
    state, info = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        log_probs.append(m.log_prob(action))
        rewards.append(reward)

        state = next_state

    # Calculate Q-values
    qvals = calc_qvals(rewards)
    qvals = torch.tensor(qvals)
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-5)

    # Policy gradient update
    optimizer.zero_grad()
    policy_loss = sum([-log_prob * q for log_prob, q in zip(log_probs, qvals)])
    policy_loss.backward()
    optimizer.step()
    return policy_loss, sum(rewards)

# Training loop
num_episodes = 500
for episode in range(num_episodes):
    loss, reward = reinforce()
    if episode % 50 == 0:
        print(f"Episode {episode}, Loss {loss}, Rewards {reward}")

env.close()

testenv = gym.make("CartPole-v1", render_mode="human")
state, info = testenv.reset()
done = False
while not done:
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    next_state, reward, terminated, truncated, info = testenv.step(action.item())
    done = terminated or truncated
    state = next_state
testenv.close()
