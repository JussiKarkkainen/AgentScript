import random
import numpy as np
import math
from collections import deque, namedtuple
from tinygrad.tensor import Tensor
import tinygrad.nn as nn

Transition = namedtuple("Transition", "state action reward next_state done")
Episode = namedtuple("Episode", "rewards, log_probs")

optimizers = {
    "Adam": nn.optim.Adam,
    "AdamW": nn.optim.AdamW,
    "SGD": nn.optim.SGD
}

class Runner:
    def __init__(self, agent, env, replay_buffer, network):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.network = network
        
        # TODO: Need to pass other params like: momentum, weight_decay, etc...
        try:
            OptimizerClass = optimizers[self.agent.config["optimizer"]["type"]]
            self.optimizer = OptimizerClass(self.network.parameters(), self.agent.config["optimizer"]["learning_rate"])
        except KeyError:
            raise NotImplementedError(f"The specified optimizer '{config['optimizer']}' is not available. Available optimizers: {', '.join(optimizers.keys())}")


    def load_weights(self):
        pass
    
    def save_weights(self):
        pass

    def update_fun(self):
        if self.replay_buffer and len(self.replay_buffer) < self.agent.config["training"]["batch_size"]:
            return
        if self.replay_buffer:
            batch = self.replay_buffer.sample(self.agent.config["training"]["batch_size"])
        self.optimizer.zero_grad()
        loss = self.agent.update(self.network, batch, self.agent.config)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        # TODO: This is going to be DQN specific at first, to be fixed later
        scores = []
        for episode in range(self.agent.config["training"]["episodes"]):
            state = self.env.init()
            episode_reward = 0
            if self.agent.config["on_policy"]:
                log_probs = []
                rewards = []
                done = False

                while not done:
                    probs = self.network(state)
                    action = probs.multinomial().item()
                    next_state, reward, terminated, truncated, info = self.env.env.step(action)
                    done = terminated or truncated
                    
                    action_one_hot = np.zeros_like(probs.numpy())
                    action_one_hot[action] = 1
                    action_one_hot = Tensor(action_one_hot)
                    selected_prob = (probs * action_one_hot).sum()
                    log_prob = selected_prob.log()
                    
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    state = next_state
                
                episode_batch = Episode(rewards=rewards, log_probs=log_probs)
                self.replay_buffer.push(episode_batch)
                loss = self.update_fun()

                episode_reward = sum(rewards)

            elif not self.agent.config["on_policy"]:
                for t in range(self.agent.config["training"]["max_time_steps"]):
                    epsilon = self.agent.config["exploration"]["epsilon_end"] + (self.agent.config["exploration"]["epsilon_start"] \
                            - self.agent.config["exploration"]["epsilon_end"]) * math.exp(-1. * episode / self.agent.config["exploration"]["epsilon_decay"])
                    if random.random() > epsilon:
                        action = self.network(state)
                        action = int(action.argmax(0).numpy())
                    else:
                        action = self.env.env.action_space.sample()

                    # Execute action in the environment
                    next_state, reward, terminated, truncated, _ = self.env.env.step(action)
                    done = terminated or truncated

                    # Store the transition in replay buffer
                    transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
                    self.replay_buffer.push(transition)

                    state = next_state
                    episode_reward += reward

                    # Update the network
                    self.update_fun()

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
            self.save_weights()
            exit()
        self.load_weights()
        raise Exception("execute")
        terminate = None
        while no_terminate() and not terminate:
            action = self.act(obs)
            obs, reward, terminate = self.environment.step(action)
        
        self.environment.shutdown()
        exit()

