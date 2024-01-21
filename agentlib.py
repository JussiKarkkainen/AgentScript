import sys
from parser.config_parser import * 
from core.agent import Agent
from core.builder import builder
from core.runner import Runner
import os

if __name__ == "__main__": 
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguments. Usage: python3 agentlib.py <file>")
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        contents = f.read() 
        config, nn, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, nn, python)
    runner = Runner(agent, env, replay_buffer, network)
    result = runner.execute()


