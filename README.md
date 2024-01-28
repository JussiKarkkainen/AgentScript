# AgentScript
![Tests](https://github.com/JussiKarkkainen/AgentLib/actions/workflows/python-app.yml/badge.svg)

### Note
WIP - Currently only supporting: DQN, REINFORCE, PPO and Actor-Critic. Other algorithms require 
modifications. The plan is to make the framework algorithm-agnostic in the future.

## Introduction
AgentScript is a Reinforcement Learning framework that simplifies the implementation of many algorithms.
It allows the user to define hyperparameters in YAML-syntax and the neural network definitions and 
update functions in Python. [Tinygrad](https://github.com/tinygrad/tinygrad) is used as the DL framework.

## Installation
```
git clone git@github.com:JussiKarkkainen/AgentScript.git
cd AgentScript
pip3 install .
```

## Example
Here is an example file that implements a DQN agent. Logging is done with [Weights & Biases](https://wandb.ai/site)
More examples are in the ```examples/``` directory.

To train the agent, run ```python3 agentlib.py examples/dqn.as```

```python
#DEFINE CONFIG
Environment:
  name: CartPole-v1
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
  capacity: 10000
  batch_size: 32
  update_freq: Batch
  type: DQN

#DEFINE CONFIG
Agent:
  type: DQN
  update_freq: Batch
  networks:
    DQN:
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 200
  discount_factor: 0.99
  training:
    episodes: 200
    max_time_steps: 10000
    batch_size: 32
  optimizer:
    DQN:
      type: Adam
      learning_rate: 0.001
  meta:
    train: True
    weight_path: "weights/dqn"
  logs: 
    logging: True
    config: [exploration, discount_factor, training, optimizer]

#DEFINE NN
class DQN:
    def __init__(self, config):
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 2)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)

#DEFINE PYTHON
def update(networks, replay_buffer, config, env=None):
    batch = replay_buffer.sample(config["training"]["batch_size"])
    states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
    curr_Q = networks("DQN", states)
    curr_Q = curr_Q.gather(actions.unsqueeze(-1), 1).squeeze(-1)
    next_Q = networks("DQN", next_states).max(1)
    expected_Q = rewards + config["discount_factor"] * next_Q * (1 - dones)
    loss = Tensor.mean((curr_Q - expected_Q.detach()) ** 2)
    return loss, None
```
## Syntax
An AgentScript file consists of YAML, NN and PYTHON sections. They need to be specified with a
```#DEFINE CONFIG```, ```#DEFINE NN``` or a ```#DEFINE PYTHON``` line at the beginning of the definition. YAML is
used to configure three objects: Environment, Replay Buffer and Agent. These are then implemented by the framework.

The ```NN``` section is used to define the neural networks used by the algorithm. They should be defined using [tinygrad](https://github.com/tinygrad/tinygrad) 
syntax. a ```config``` dictionary onject is passed to the ````___init___()``` method of the neural network definition. This contains parameters defined in the
```networks``` section of the ```Agent``` config.

The ```PYTHON``` section is used to define the update step of the algorithm. For DQN for example, it takes a batch of inputs and outputs the loss. The rest is 
handled by the framework. 

