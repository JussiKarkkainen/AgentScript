from core.builder import builder
from core.runner import Runner
from parser.config_parser import *
import os

def run_test(test_file: str):
    os.environ["CPU"] = "1" # reinforce doesn't work on metal for some reason
    with open(test_file, "r") as f:
        contents = f.read() 
        config, nn, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, nn, python)
    runner = Runner(agent, env, replay_buffer, network)
    try:
        result = runner.execute()
        assert result == True
    except:
        raise Exception

def test_dqn():
    run_test("tests/test_files/test_dqn.agsc")

def test_reinforce():
    run_test("tests/test_files/test_reinforce.agsc")

def test_actorcritic():
    run_test("tests/test_files/test_actor-critic.agsc")
    
def test_ppo():
    run_test("tests/test_files/test_ppo.agsc")

