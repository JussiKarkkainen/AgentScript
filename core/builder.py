from typing import Dict, Any, Tuple, List
import utils
import parser.config_parser as parser
from core.environment import Environment
from core.replay_buffer import ReplayBuffer
from core.agent import Agent
from core.networks import NeuralNetwork

def builder(config: List[Dict[str, dict[str, Any]]], nn: List[str], python: List[str]):
    # TODO: Validate the configuration, Use the validate_config() function in config_parser.py
    parser.validate_config(config)

    # Turns the string config values into the actual classes
    modules = [(eval(list(conf.keys())[0]), list(conf.keys())[0], list(conf.values())[0]) for conf in config]
    # module[0] = module class, module[1] = module name, module[2] = module init params
    environment, replay_buffer, agent = None, None, None

    for module in modules:
        if module[1] == "Environment":
            environment = module[0](module[2])
        elif module[1] == "ReplayBuffer":
            replay_buffer = module[0](module[2])
        elif module[1] == "Agent":
            agent = module[0](module[2], python)

    # Neural Network definition
    network = NeuralNetwork(nn, module[2])
    
    return agent, environment, replay_buffer, network

