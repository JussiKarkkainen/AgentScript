# AgentScript
![Tests](https://github.com/JussiKarkkainen/AgentLib/actions/workflows/python-app.yml/badge.svg)


## Introduction
AgentScript is a config language used to define Deep Reinforcement Learning (DRL) algorithms and agents.
It allows the user to define agents that work in a variety of environments that follow the usual 
RL loop of ```state -> action -> reward -> new state```. It uses [tinygrad](https://github.com/tinygrad/tinygrad)
for the Neural Network Implementations.

## Installation
Clone the repo and install
```
git clone git@github.com:JussiKarkkainen/AgentScript.git
cd AgentScript
pip3 install .
```

## Details
More specifically, this library implementes the following loop:

```
state = environment.reset()
while not converged:
    action = agent.forward(state)
    new_state, reward, done = environment.step(action)
    replay_buffer.update(state, action, reward, new_state, done)
    agent.update(replay_buffer)
```
Most RL agents can be explained using the above loop. Therefore writing it everytime
you want to implement an RL agent can get tedious. AgentScript simplifies this in two ways:
1. It provides implementations of Replay Buffers and Environments that the user can take advantage
   of and configure to their needs using a YAML-configuration.
2. It allows the user to define the Neural Network part of the RL agent (```agent.forward()```)
   as well as the ```update()``` function that defines how the agent learns. The rest is handled by 
   the library itself.

### Syntax
An AgentScript file consists of YAML and python sections. They need to be specified with a
```#DEFINE CONFIG``` or a ```#DEFINE PYTHON``` at the beginning of the definition. YAML is
used to configure three objects: Environment, Replay Buffer and Agent. Python is used to
define the Neural Networks and the Agent's update functions. The Neural Networks are defined
using [tinygrad](https://github.com/tinygrad/tinygrad).

## Example
Here is an example file that implements a DQN agent. More examples are in the ```examples/```
directory.

To train the agent, run ```python3 agentlib.py examples/dqn.as```

```
#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
  capacity: 10000
  batch_size: 64

#DEFINE CONFIG
Agent:
  type: DQN
  network:
    input_shape: 4  
    hidden_layers: [32, 64] 
    output_shape: 2  
    activation: relu
  exploration:
    type: EpsGreedy
    epsilon: 1.0
    decay_rate: 0.99
    min_epsilon: 0.1
  memory: replay_buffer 
  tnameget_update_frequency: 100
  discount_factor: 0.99
  loss_function: huber_loss
  optimizer:
    type: Adam
    learning_rate: 0.001
  meta:
    train: false
    weight_path: None

#DEFINE PYTHON
class DQNNetwork(nn.Module):
    def __init__(self, config):
        super(DQNNetwork, self).__init__()
        self.layers = nn.ModuleList()
        input_size = config['network']['input_shape']
        for hidden_size in config['network']['hidden_layers']:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output = nn.Linear(input_size, config['network']['output_shape'])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

#DEFINE PYTHON
def dqn_update(agent, batch):
    states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
    curr_Q = agent.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_Q = agent.target_network(next_states).max(1)[0]
    expected_Q = rewards + (agent.gamma * next_Q * (1 - dones))
    loss = F.mse_loss(curr_Q, expected_Q.detach())
    return loss
```

## TODO
- [ ] Support for online learning agents
- [ ] Support for offline learning agents
