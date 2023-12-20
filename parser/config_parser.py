import json
import yaml
from typing import List, Tuple, Dict

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


accepted_configs = ["environment", "replay_buffer", "agent"]


def validate_config(config_dict: Dict[str, List[str]]) -> bool:
    for key in config_dict.keys():
        if key == "environment":
            pass
        elif key == "replay_buffer":
            pass
        elif key == "agent":
            pass
        else:
            return

def config_builder(config_sections: List[Dict[str, Dict[str, str]]]):
    configs = []
    for config in config_sections:
        for key in config.keys():
            assert key in accepted_configs, f"Illegal configuration object: {config}"
        configs.append(config)

    
    
    raise Exception
