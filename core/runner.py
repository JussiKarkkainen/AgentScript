import random
import numpy as np
import math
from collections import deque
from tinygrad.tensor import Tensor

class Runner:
    def __init__(self, agent, env, replay_buffer, network):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.network = network

    def load_weights(self):
        pass
    
    def update(self):
        if len(replay_buffer) < self.agent.config["training"]["batch_size"]:
            return
        batch = self.replay_buffer.sample()
        loss = self.agent.update(batch)
        return loss

    def train(self):
        # TODO: This is going to be DQN specific at first, to be fixed later
        scores = deque(maxlen=100)
        for episode in range(self.agent.config["training"]["episodes"]):
            state = self.env.init()
            episode_reward = 0
            for t in range(self.agent.config["training"]["max_time_steps"]):
                epsilon = self.agent.config["exploration"]["epsilon_end"] + (self.agent.config["exploration"]["epsilon_start"] \
                        - self.agent.config["exploration"]["epsilon_end"]) * math.exp(-1. * episode / self.agent.config["exploration"]["epsilon_decay"])
                if random.random() > epsilon:
                    action = network(state)
                    action = action.max(0)
                    action = action[1].item()
                else:
                    action = self.env.env.action_space.sample()

                # Execute action in the environment
                next_state, reward, terminated, truncated, _ = self.env.env.step(action)
                done = terminated or truncated

                # Store the transition in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                # Update the network
                self.update()

                if done:
                    break

            scores.append(episode_reward)
            mean_score = np.mean(scores)

            print(f"Episode: {episode}, Total Reward: {episode_reward}, Mean reward {mean_score}")

            if mean_score >= 300:
                print("Environment solved in {} episodes!".format(episode))
                break


    def execute(self):
        if self.agent.config["meta"]["train"] == True:
            self.train()
        self.load_weights()
        raise Exception("execute")
        terminate = None
        while no_terminate() and not terminate:
            action = self.act(obs)
            obs, reward, terminate = self.environment.step(action)
        
        self.environment.shutdown()
        exit()

