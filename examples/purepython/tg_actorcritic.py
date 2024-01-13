import gymnasium as gym
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import numpy as np

# Environment setup
env = gym.make('CartPole-v1')

# Hyperparameters
learning_rate = 0.001
gamma = 0.99

# Actor Model
class Actor:
    def __init__(self):
        self.al1 = nn.Linear(4, 128)
        self.al2 = nn.Linear(128, 2)
    
    def __call__(self, state):
        x = self.al2(self.al1(state).relu())
        return x.softmax()

class Critic:
    def __init__(self):
        self.cl1 = nn.Linear(4, 128)
        self.cl2 = nn.Linear(128, 1)
    
    def __call__(self, state):
        return self.cl2(self.cl1(state).relu())

# Initialize Actor-Critic models
actor = Actor()
critic = Critic()

# Optimizers
actor_optimizer = nn.optim.Adam(nn.state.get_parameters(actor), lr=learning_rate)
critic_optimizer = nn.optim.Adam(nn.state.get_parameters(critic), lr=learning_rate)

def train():
    for episode in range(1000):  # Number of episodes
        rewards = []
        state, info = env.reset()
        state = Tensor(state, requires_grad=False).unsqueeze(0)
        done = False
        I = 1

        while not done:
            action_prob = actor(state)
            action = action_prob.multinomial()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            next_state = Tensor(next_state).unsqueeze(0)       
            reward = Tensor([reward])
            rewards.append(reward)
            
            state_val = critic(state)
            next_state_val = critic(next_state) if not done else Tensor([0.])
            advantage = reward + gamma * next_state_val - state_val
        
            action_one_hot = np.zeros_like(action_prob.squeeze().numpy())
            action_one_hot[action.item()] = 1
            action_one_hot = Tensor(action_one_hot)
            selected_prob = (action_prob.squeeze() * action_one_hot).sum()
            log_prob = selected_prob.log()
           
            actor_loss = (-log_prob * advantage.detach()).sum()
            critic_loss = (((reward + gamma * next_state_val) - state_val) ** 2).squeeze()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
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
        state = Tensor(state).unsqueeze(0)
        action_prob = actor(state)
        action = action_prob.multinomial().item()

        state, reward, terminated, truncated, _ = testenv.step(action)
        done = terminated or truncated

    testenv.close()

if __name__ == "__main__":
    train()
