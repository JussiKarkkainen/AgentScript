import json
import yaml
from typing import List, Tuple, Dict
import utils
from core.environment import implemented_envs

def json_config_parser(config: str) -> Dict:
    config_dict = json.loads(config)
    return config_dict

def config_parser(config_str: str) -> Tuple[List[Dict[str, Dict[str, str]]], List[str]]:
    sections = config_str.split("#DEFINE") 
    sections = [section.strip() for section in sections if section.strip()]
    python_sections = [section[len("PYTHON"):].strip() for section in sections if section.startswith("PYTHON")]
    config_sections = [section[len("CONFIG"):].strip() for section in sections if section.startswith("CONFIG")]
    parsed_config_sections = [yaml.safe_load(section) for section in config_sections]
    return parsed_config_sections, python_sections


def validate_env_config(env_config):
    for key, value in env_config.items():
        assert key == "name" or key == "horizon" or key == "preprocess"
        if key == "name":
            # TODO: Will have more than gym envs in the future
            assert value in implemented_envs["gym"]
        elif key == "horizon":
            assert type(value) == int
        elif key == "preprocess":
            assert type(value) == bool

def validate_replay_buffer_config(env_config):
    for key, value in env_config.items():
        assert key == "update_freq" or key == "capacity" or key == "batch_size" or key == "type"
        if key == "updated_freq":
            assert value == "Timestep" or value == "Episodic" or value == "Batch"
        elif key == "capacity":
            assert type(value) == int
        elif key == "batch_size":
            assert type(value) == int
        elif key == "type":
            assert type(value) == str

def validate_agent_config(env_config):
    # TODO: This will change so much that its not worth implementing now
    pass

def validate_config(config_dict: Dict[str, List[str]]):
    #TODO: This function basically defines the syntax, if this doesn't raise an exception, the config is valid
    assert len(config_dict) == 3, "You must have configurations for 'Environment', 'ReplayBuffer' and 'Agent', other configurations as invalid"
    try:
        env_config = config_dict[0]["Environment"]
        replay_buffer_config = config_dict[1]["ReplayBuffer"]
        agent_config = config_dict[2]["Agent"]
    except KeyError as e:
        print(f"KeyError in configuration definition: {e}")
    
    validate_env_config(env_config)
    validate_replay_buffer_config(replay_buffer_config)
    validate_agent_config(agent_config)

