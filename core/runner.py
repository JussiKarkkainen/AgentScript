import random
import os
import numpy as np
import math
from collections import deque, namedtuple
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from logs.wandb_logs import WandbLogger

Transition = namedtuple("Transition", "state action reward next_state done")
Episode = namedtuple("Episode", "states actions rewards log_probs_tens log_probs_list")

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
        self.optimizers = self.create_optimizers()
        self.logger = WandbLogger(self.agent.config) if self.agent.config["logs"]["logging"] else None

    def create_optimizers(self):
        # TODO: Need to pass other params like: momentum, weight_decay, etc...
        optimizer_dict = {}
        for network_name, network in self.network.networks.items():
            try:
                optimizer_config = self.agent.config["optimizer"][network_name]
                OptimizerClass = optimizers[optimizer_config["type"]]
                optimizer_dict[network_name] = OptimizerClass(self.network.parameters(network_name), optimizer_config["learning_rate"])
            except KeyError as e:
                raise NotImplementedError(f"The specified optimizer for {network_name} is not available. Error: {e}")
        return optimizer_dict

    def update_fun(self):
        if self.replay_buffer and len(self.replay_buffer) < self.agent.config["training"]["batch_size"]:
            return
        for optim in self.optimizers.values():
            optim.zero_grad()
        losses, meta = self.agent.update_funcs["update"](self.network, self.replay_buffer, self.agent.config, self.env)
        losses = [losses] if type(losses) != tuple else losses
        for loss, optim in zip(losses, self.optimizers.values()):
            # TODO: This assumes the losses are returned in the same order as the optimizers are defined, this shoudn't matter
            loss.backward()
            optim.step()
        return losses, meta

    def timestep_update(self, state):
        done = False
        rewards = []
        self.replay_buffer.push(state)
        while not done:
            (actor_loss, critic_loss), (reward, done) = self.update_fun()
            rewards.append(reward.numpy())

            if done:
                break
            

        episode_reward = sum(rewards)
        return episode_reward

    def episodic_update(self, state):
        rewards = []
        done = False
        #TODO: reinforce need list, ppo needs tens, to be fixed
        states, actions, rewards, log_probs_tens, log_probs_list = [], [], [], [], []
        while not done:
            probs = self.network("Policy", state)
            action = probs.multinomial().item()
            next_state, reward, terminated, truncated, info = self.env.env.step(action)
            done = terminated or truncated
            
            action_one_hot = np.zeros_like(probs.numpy())
            action_one_hot[action] = 1
            action_one_hot = Tensor(action_one_hot)
            selected_prob = (probs * action_one_hot).sum()
            log_prob = selected_prob.log()
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs_tens.append(log_prob.item())
            log_probs_list.append(log_prob)

            state = next_state
        
        episode_batch = Episode(states=states, actions=actions, rewards=rewards, log_probs_tens=log_probs_tens, log_probs_list=log_probs_list)
        self.replay_buffer.push(episode_batch)
        loss = self.update_fun()

        episode_reward = sum(rewards)
        return episode_reward

    def batch_update(self, state, episode):
        rewards = []
        for t in range(self.agent.config["training"]["max_time_steps"]):
            epsilon = self.agent.config["exploration"]["epsilon_end"] + (self.agent.config["exploration"]["epsilon_start"] \
                    - self.agent.config["exploration"]["epsilon_end"]) * math.exp(-1. * episode / self.agent.config["exploration"]["epsilon_decay"])
            if random.random() > epsilon:
                action = self.network("DQN", state)
                action = action.argmax(0).item()
            else:
                action = self.env.env.action_space.sample()

            next_state, reward, terminated, truncated, _ = self.env.env.step(action)
            done = terminated or truncated

            transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
            self.replay_buffer.push(transition)

            state = next_state
            rewards.append(reward)

            self.update_fun()

            if done:
                break
        
        episode_reward = sum(rewards)
        return episode_reward

    def train(self):
        scores = []
        for episode in range(1, self.agent.config["training"]["episodes"]):
            state = self.env.init()
            episode_reward = 0

            if self.agent.config["update_freq"] == "Timestep":
                episode_reward = self.timestep_update(state)
            elif self.agent.config["update_freq"] == "Episodic":
                episode_reward = self.episodic_update(state)
            elif self.agent.config["update_freq"] == "Batch":
                episode_reward = self.batch_update(state, episode)

            scores.append(episode_reward)
            if self.logger:
                self.logger.log({"Episode_reward": episode_reward})
            mean_score = np.mean(scores)

            print(f"Episode: {episode}, Total Reward: {episode_reward}, Mean reward {mean_score}")

            if mean_score >= 300:
                print("Environment solved in {} episodes!".format(episode))
                break


    def execute(self):
        if self.agent.config["meta"]["train"] == True:
            if self.logger:
                with self.logger:
                    self.train()
            else:
                self.train()
        
        if self.agent.config["meta"]["weight_path"]:
            for name, network in self.network.networks.items():
                path = os.path.join(os.getcwd(), self.agent.config["meta"]["weight_path"], name) 
                if not os.path.exists(path):
                    os.mkdir(path)
                nn.state.safe_save(nn.state.get_state_dict(network), path+".safetensors")
        
        return True
