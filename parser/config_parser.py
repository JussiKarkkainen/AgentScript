import json
import yaml

def json_config_parser(config):
    config_dict = json.loads(config)
    return config_dict

def parse_update_function(formula: str):
    print(formula)
    print(type(formula))
    raise Exception

def config_parser(config_str):
    # Parse the YAML configuration string
    config = yaml.safe_load(config_str)

    # Initialize a dictionary to hold the parsed configuration
    parsed_config = {
        'environment': None,
        'agent': {
            'network': None,
            'optimizer': None,
            'replay_buffer': None,
            'update_function': None
        }
    }

    # Parse the environment
    parsed_config['environment'] = config.get('environment', None)

    # Parse the agent configuration
    agent_config = config.get('agent', {})

    # Parse the network configuration
    if 'network' in agent_config:
        # Assuming the format is "Type[parameter1, parameter2, ...]"
        network_str = agent_config['network']
        network_type, network_params = network_str.split('[')
        network_params = network_params.rstrip(']').split(', ')
        parsed_config['agent']['network'] = {
            'type': network_type,
            'params': network_params
        }

    # Parse the optimizer configuration
    if 'optimizer' in agent_config:
        optimizer_str = agent_config['optimizer']
        optimizer_type, optimizer_params = optimizer_str.split('[')
        optimizer_params = optimizer_params.rstrip(']').split(', ')
        parsed_config['agent']['optimizer'] = {
            'type': optimizer_type,
            'params': {p.split('=')[0]: p.split('=')[1] for p in optimizer_params}
        }

    # Parse the replay buffer configuration
    if 'replay_buffer' in agent_config:
        parsed_config['agent']['replay_buffer'] = agent_config['replay_buffer']

    # The update_function will be parsed later as mentioned
    if 'update_function' in agent_config:
        parsed_config['agent']['update_function'] = parse_update_function(agent_config['update_function'])
        
    return parsed_config


def custom_config_parser(config):
    lines  = config.split("\n")
    lines = [line for line in lines if line]
    
    
    main_definitions = ["environment:", "agent:"]
    for line in lines:
        if line[0] != " ":
            assert line == "environment:" or line == "agent:"

    agentscript_config = {}

    for line in lines:
        if line in main_definitions:
            line = line.strip(":")
            agentscript_config[line] = dict()
    main_definitions = []
    for line in lines:
        if line == "environment:" or line == "agent:":
            main_definitions.append(line)
    
    pprint(main_definitions)

    agentscript_config = {"environment:": None, "agent": None}

    
    raise Exception
