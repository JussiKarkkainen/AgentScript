import json
import yaml

def json_config_parser(config):
    config_dict = json.loads(config)
    return config_dict

def parse_update_function(formula: str):
    #print(formula)
    #print(type(formula))
    #raise Exception
    pass

def config_parser(config_str):
    sections = config_str.split("#DEFINE") 
    sections = [section.strip() for section in sections if section.strip()]
    python_sections = [section[len("PYTHON"):].strip() for section in sections if section.startswith("PYTHON")]
    config_sections = [section[len("CONFIG"):].strip() for section in sections if section.startswith("CONFIG")]
    parsed_config_sections = [yaml.safe_load(section) for section in config_sections]
    raise Exception
    return parsed_config


