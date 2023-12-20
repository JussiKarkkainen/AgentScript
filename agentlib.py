import sys
from parser.config_parser import json_config_parser, config_parser
from core.agent import Agent
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguments. Usage: python3 agentlib.py <config.al>")
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        contents = f.read() 
        if os.getenv("SCRIPT"):
            config = config_parser(contents)
        else:
            config = json_config_parser(contents)

    agent = Agent(config)
    agent.execute()
