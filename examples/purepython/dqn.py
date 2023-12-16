import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

# Create the CartPole environment
env = gym.make("CartPole-v1")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Network parameters
input_dim = env.observation_space.shape[0]  # Input dimension
output_dim = env.action_space.n  # Number of actions

# Create the DQN network instance
dqn = DQN(input_dim, output_dim)

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99  # Discount factor
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 200

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Replay buffer parameters
replay_buffer = ReplayBuffer(10000)
batch_size = 32

optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

def update(dqn, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    # Sample a batch of experiences from the replay buffer
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(np.array(dones))

    # Q values for the current states
    curr_Q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q values for the next states
    next_Q = dqn(next_states).max(1)[0]
    expected_Q = rewards + gamma * next_Q * (1 - dones)

    # Loss
    loss = F.mse_loss(curr_Q, expected_Q.detach())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 1000
scores = deque(maxlen=100)

for episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0

    for t in range(1, 10000):  # Limit the number of time steps per episode
        # Select action according to epsilon-greedy strategy
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-1. * episode / epsilon_decay)
        if random.random() > epsilon:
            action = dqn(torch.FloatTensor(state))
            action = action.max(0)
            action = action[1].item()
        else:
            action = env.action_space.sample()

        # Execute action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store the transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Update the network
        update(dqn, optimizer, replay_buffer, batch_size, gamma)

        if done:
            break

    scores.append(episode_reward)
    mean_score = np.mean(scores)

    print(f"Episode: {episode}, Total Reward: {episode_reward}, Mean reward {mean_score}")

    if mean_score >= 300:
        print("Environment solved in {} episodes!".format(episode))
        break

testenv = gym.make("CartPole-v1", render_mode="human")
state, info = testenv.reset()
done = False
total_reward = 0

while not done:
    action = dqn(torch.FloatTensor(state)).max(0)[1].item()  # Choose the best action
    state, reward, terminated, truncated, _ = testenv.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()  # Close the environment when done
print("Test episode: Total Reward = {}".format(total_reward))
