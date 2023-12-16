import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1")

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


# Network parameters
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Create the policy network
policy = PolicyNetwork(input_dim, output_dim)

# Optimizer
import torch.optim as optim
learning_rate = 0.01
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


from collections import deque

def train(policy, optimizer, episodes):
    for episode in range(episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False

        # Generate a trajectory
        while not done:
            state = torch.FloatTensor(state)
            probs = policy(state)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            state = next_state

        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Policy gradient update
        returns = torch.tensor(returns)
        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        optimizer.zero_grad()
        policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
        policy_loss.backward()
        optimizer.step()

        # Print episode info
        total_reward = sum(rewards)
        print(f"Episode: {episode}, Total Reward: {total_reward}")

gamma = 0.99
episodes = 10000
train(policy, optimizer, episodes)





