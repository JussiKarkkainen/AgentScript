import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
import collections
import random
import numpy as np

from libcore.environment import Environment
from libcore.logging import Logger
from libcore.replay_buffer import ReplayBuffer
from libcore.weights import WeightManager

class SharedStorage:
    def __init__(self):
        self.networks = {}

    def push(self, idx, network):
        self.networks[idx] = network

    def get(self):
        if len(self.networks.keys()) == 0:
            return MuZeroNet()
        return self.networks[max(self.networks.keys())]

class Node:
    def __init__(self, policy_prior):
        self.hidden_state = None
        self.reward = 0
        self.children = {}
        self.policy_prior = policy_prior
        self.visit_count = 0

    def is_expanded(self):
        return len(list(self.children.keys())) > 0

class MinMaxStats:
    def __init__(self, maximum, minimum):
        self.max = maximum
        self.min = minimum

    def update(self, value):
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value):
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value

class MLP(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.core = nn.Linear(128, 128)
        self.net = net
        if net == "representation":
            # Observation space -> hidden state
            self.l1 = nn.Linear(4, 128)
            self.o = nn.Linear(128, 16)
        elif net == "dynamics":
            # Hidden state + action -> next hidden state + reward
            self.l1 = nn.Linear(17, 128)
            self.o = nn.Linear(128, 17)
        elif net == "prediction":
            # Hidden state -> policy + value
            self.l1 = nn.Linear(16, 128)
            self.value_head = nn.Linear(128, 1)
            self.policy_head = nn.Linear(128, 2)
        else:
            raise Exception("Invalid neural network type")

    def forward(self, x):
        x = f.relu(self.l1(x))
        x = f.relu(self.core(x))
        if self.net == "prediction":
            policy = f.softmax(self.policy_head(x), dim=-1)
            value = self.value_head(x)
            out = (policy, value)
        else:
            out = self.o(x)

        return out

class MuZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation_net = MLP("representation")
        self.dynamics_net = MLP("dynamics")
        self.prediction_net = MLP("prediction")

    def representation(self, x):
        x = torch.from_numpy(x)
        hidden_state = self.representation_net(x)
        return hidden_state

    def dynamics(self, s, a):
        combined_input = torch.stack(s, a)
        return self.dynamics_net(combined_input)

    def prediction(self, s):
        policy, value = self.prediction_net(s)
        return policy, value

    def initial_inference(self, x):
        # Used to start MCTS: Obs -> hidden state -> policy, value
        hidden_state = self.representation(x)
        policy, value = self.prediction(hidden_state)
        return policy, value, hidden_state

    def recurrent_inference(self, s, a):
        # Moving between states in MCTS: hidden state, action -> next hidden state, reward -> policy, value
        hidden_state, reward = self.dynamics(s, s)
        policy, value = self.prediction(hidden_state)
        return (hidden_state, reward), (policy, value)

def expand_node(node, network_output, actions):
    hidden_state, reward, policy, value = network_output
    node.hidden_state = hidden_state
    node.reward = reward
    for action, policy_prob in zip(range(actions.n), policy):
        node.children[action] = Node(policy_prob)

def add_exploration_noise(node):
    actions = len(list(node.children.keys()))
    noise = np.random.dirichlet([muzero_config["root_dirichlet_alpha"]] * actions)
    frac = muzero_config["root_exploration_fraction"]
    for action, n in zip(range(actions), noise):
        node.children[action].policy_prior = node.children[action].policy_prior * (1 - frac) + n * frac

def ucb_score_function(parent, child, min_max_stats):
    # TODO: Explain the UCB function
    raise Exception
    pb_c = math.log((parent.visit_count + muzero_config["pb_c_base"] + 1) / muzero_config["pb_c_base"]) + muzero_config["pb_c_init"]
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score

def select_child(node, min_max_stats):
    _, action, child = max((ucb_score_function(node, child, min_max_stats), action, child) for action, child in node.children.items())
    return action, child     

def backpropagate(search_path, value, min_max_stats):
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value
        
def mcts(root, network):
    min_max_stats = MinMaxStats(muzero_config["reward_max"], muzero_config["reward_min"])
    for simulation in range(muzero_config["num_simulations"]):
        node = root
        search_path = [node]
        history = []
        while node.is_expanded():
            action, node = select_child(node, min_max_stats)
            history.append(action)
            search_path.append(node)
        
        parent = search_path[-2]
        hidden_state, reward, policy, value = network.recurrent_inference(parent.hidden_state, history.last_action)
        expand_node(node, (hidden_state, reward, policy, value), history)
        backpropagate(search_path, value, min_max_stats)

def select_action(history_len, root, network):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    temp = muzero_config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=network.training_steps())
    _, action = softmax_sample(visit_counts, t)
    return action

def self_play(replay_buffer, storage):
    environment = Environment("CartPole-v1")
    root_state = environment.reset()         
    muzero = storage.get()
    actions, rewards, child_visits, root_values = [], [], [], [], []
    done = False
    while not done:
        policy, value, hidden_state = muzero.initial_inference(root_state)
        root = Node(0)
        expand_node(root, (hidden_state, 0, policy, value), environment.action_space())
        add_exploration_noise(root)
        mcts(root, muzero)
        action = select_action(game_history, root, muzero)
        next_state, reward, terminated, truncated, info = environment.step(action.item())
        done = terminated or truncated
        actions.append(actions)
        rewards.append(reward)
        root_values.append(root.value())
        self.child_visits.append([root.children[a].visit_count / sum(child.visit_count for child in root.children.values()) if a in root.children else 0 for a in action_space])

    episode = Episode(actions=actions, rewards=rewards, child_visits=child_visits, root_values=root_values)
    replay_buffer.push(episode)

def train(muzero, replay_buffer, storage):
    network = MuZero()
    learning_rate = training_config["lr"]
    optimizer = torch.optim.Adam(network.parameters(), learning_rate)
    for step in range(training_config["num_train_steps"]):
        if step % training_config["checkpoint_checks"] == 0:
            storage.save_network(network)
        batch = replay_buffer.sample(training_config["num_unroll_steps"], training_config["td_steps"])
        print(batch)
        raise Exception
        update_weights(network, optimizer, batch)
    storage.save_network(network)

muzero_config = {"root_dirichlet_alpha": 0.3, # This is used to add exploration noise to the root node.
                 "root_exploration_fraction": 0.25, # TODO: Explain
                 "num_simulations": 10, # Number of MCTS simulations to do per move 
                 "reward_max": 500,     # Used in MinMaxStats to normalize rewards
                 "reward_min": 0}

training_config = {"batch_size": 64,
                   "num_train_steps": 1000}


Episode = collections.namedtuple("Episode", "states actions rewards child_visits root_values")

if __name__ == "__main__":
    weight_manager = WeightManager("weights/MuZero")
    logger = Logger("MuZero", muzero_config | training_config)
    replay_buffer = ReplayBuffer(capacity=10000)
    storage = SharedStorage()
    # NOTE: This sort of aligns with the pseudocode: two jobs, selfplay and train
    # In the official code, there are multiple selfplay jobs running at the same time; selfplay and train might do that too.
    
    self_play(replay_buffer, storage)
    train(replay_buffer, storage)

