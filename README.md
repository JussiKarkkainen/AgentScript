### AgentScript
![Tests](https://github.com/JussiKarkkainen/AgentLib/.github/workflows/python-app.yml/badge.svg)


## Introduction
AgentScript is a config language used to define Deep Reinforcement Learning (DRL) algorithms and agents.
It allows the user to define agents that work in a variety of environments that follow the usual 
RL loop of ```state -> action -> reward -> new state```. It uses [tinygrad](https://github.com/tinygrad/tinygrad)
for the Neural Network Implementations.


## Example
Here is an example config file that implements [Q-learning](https://en.wikipedia.org/wiki/Q-learning)

```
Environment {
    Gymnasium: Cartpole-v2
}


Agent {
    
    Perception {
        observation_space: Vector[9, float32],
    }    
    
    NeuralNetwork {
        MLP: Layers[3]
    }
    
    Actor {
        action_space: Discrete[2] 
    }
}
```



## Installation
Clone the repo and install
```
git clone git@github.com:JussiKarkkainen/AgentScript.git
cd AgentScript
pip3 install .
```


