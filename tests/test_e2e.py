from core.builder import builder
from core.runner import Runner
from parser.config_parser import *

def test_dqn():
    with open("tests/test_files/test_dqn.as", "r") as f:
        contents = f.read() 
        config, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, python)
    runner = Runner(agent, env, replay_buffer, network)
    try:
        result = runner.execute()
        assert result == True
    except:
        raise Exception

def test_reinforce():
    with open("tests/test_files/test_reinforce.as", "r") as f:
        contents = f.read() 
        config, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, python)
    runner = Runner(agent, env, replay_buffer, network)
    try:
        result = runner.execute()
        assert result == True
    except:
        raise Exception

def test_actorcritic():
    with open("tests/test_files/test_actor-critic.as", "r") as f:
        contents = f.read() 
        config, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, python)
    runner = Runner(agent, env, replay_buffer, network)
    try:
        result = runner.execute()
        assert result == True
    except:
        raise Exception
    

def test_ppo():
    with open("tests/test_files/ppo.as", "r") as f:
        contents = f.read() 
        config, python = config_parser(contents)
    
    agent, env, replay_buffer, network = builder(config, python)
    runner = Runner(agent, env, replay_buffer, network)
    try:
        result = runner.execute()
        assert result == True
    except:
        raise Exception

