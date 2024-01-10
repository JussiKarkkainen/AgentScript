from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import gymnasium as gym
import numpy as np


# Hyperparameters
learning_rate = 0.001
gamma = 0.99

# Environment
env = gym.make('CartPole-v1')

# Policy Network
class Policy:
    def __init__(self):
        self.fc = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def __call__(self, x):
        x = self.fc(x).relu()
        x = self.fc2(x)
        return x.softmax().realize()

policy = Policy()
optimizer = nn.optim.Adam(nn.state.get_parameters(policy), lr=learning_rate)

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
        state = Tensor(state, requires_grad=False)
        probs = policy(state)
        
        action = probs.multinomial().item()
        action_one_hot = np.zeros_like(probs.numpy())
        action_one_hot[action] = 1
        action_one_hot = Tensor(action_one_hot)
        selected_prob = (probs * action_one_hot).sum()
        log_prob = selected_prob.log()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

    # Calculate Q-values
    qvals = calc_qvals(rewards)
    qvals = Tensor(qvals, requires_grad=False)
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-5)
    
    # Policy gradient update
    optimizer.zero_grad()
    policy_loss = [-log_prob * q for log_prob, q in zip(log_probs, qvals)]
    policy_loss = sum(policy_loss)
    policy_loss.backward()
    optimizer.step()
    return policy_loss, sum(rewards)

# Training loop
num_episodes = 500
for episode in range(num_episodes):
    loss, reward = reinforce()
    if episode % 50 == 0:
        print(f"Episode {episode}, Loss {loss.numpy()}, Rewards {reward}")

env.close()

testenv = gym.make("CartPole-v1", render_mode="human")
state, info = testenv.reset()
done = False
while not done:
    state = Tensor(state).unsqueeze(0)
    probs = policy(state)
    action = np.random.choice(len(probs.data), p=probs.data)
    next_state, reward, terminated, truncated, info = testenv.step(action.item())
    done = terminated or truncated
    state = next_state
testenv.close()
