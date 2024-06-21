# main.py
# runs the simulation

def debug():
    import os
    import sys
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    package_root_directory = os.path.dirname(os.path.dirname(current_directory))
    sys.path.append(package_root_directory)

try:
    import agent_torch
except:
    debug()
    import agent_torch

import argparse
from tqdm import trange
from agent_torch import Registry, Runner
from agent_torch.helpers import read_config, read_from_file, grid_network
from substeps import *
from helpers import *

print(':: execution started')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path to yaml config file')
config_file = parser.parse_args().config

config = read_config(config_file)
metadata = config.get('simulation_metadata')
num_episodes = metadata.get('num_episodes')
num_steps_per_episode = metadata.get('num_steps_per_episode')

registry = Registry()
registry.register(read_from_file, 'read_from_file', 'initialization')
registry.register(grid_network, 'grid', key='network')
registry.register(map_network, 'map', key='network')

runner = Runner(config, registry)
runner.init()

print(':: preparing simulation...')

for episode in trange(num_episodes, desc=':: running simulation'):
  runner.step(num_steps_per_episode)

print(':: execution completed')