import sys
from parser.config_parser import * 
from core.agent import Agent
from core.builder import builder
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguments. Usage: python3 agentlib.py <config.al>")
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        contents = f.read() 
        if os.getenv("SCRIPT"):
            config, python = config_parser(contents)
        else:
            config = json_config_parser(contents)
    
    agent, env, replay_buffer = builder(config, python)
    agent.execute(env, replay_buffer)
