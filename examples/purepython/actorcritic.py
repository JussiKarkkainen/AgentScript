import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Environment setup
env = gym.make('CartPole-v1')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99

# Actor Model
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # Define your actor network here
        self.layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
    
    def forward(self, state):
        x = self.layer(state)
        return Categorical(torch.softmax(x, dim=-1))

# Critic Model
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Define your critic network here
        self.layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.layer(state)

# Initialize Actor-Critic models
actor = Actor()
critic = Critic()

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

def train():
    for episode in range(1000):  # Number of episodes
        rewards = []
        state, info = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        done = False
        I = 1

        while not done:
            action_prob = actor(state)
            action = action_prob.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)       
            reward = torch.tensor([reward]).float()
            rewards.append(reward)
            
            state_val = critic(state)
            next_state_val = critic(next_state) if not done else torch.tensor([0]).float()
            advantage = reward + gamma * next_state_val - state_val

            actor_loss = -action_prob.log_prob(action) * advantage
            critic_loss = F.mse_loss(reward + gamma * next_state_val, state_val)

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()

            # Update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Move to the next state
            state = next_state

            if done:
                break
            
            I *= gamma

        print(f"Rewards on episode: {episode} were {sum(rewards).item()}")
        if sum(rewards).item() > 300:
            test()


def test():
    testenv = gym.make('CartPole-v1', render_mode="human")
    state, info = testenv.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = actor(state)
        action = action_prob.sample().item()

        state, reward, terminated, truncated, _ = testenv.step(action)
        done = terminated or truncated

    testenv.close()

if __name__ == "__main__":
    train()
