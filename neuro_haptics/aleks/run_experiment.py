# Run the script with
# `python run_experiment.py -t 1`
# This will run a simulation trial, with a total length of 1 seconds.
# Participant answers will be given every 0.001 to 0.002 seconds.
# The participant "true" level of feedback is 6 (correct_action).

# The script will output:
# - A list of actions taken, in the format: 
# `elapsed_time > action_taken -> reward_received`
# - The Q-table at the end of the trial,
# with rows representing states and columns representing acions
# - The total count of given actions taken in a given state, 
# with rows representing states and columns representing acions
# - Total timesteps (actions) taken
# - Total reward obtained

# The agent is rewarded `-n` for guessing the wrong feedback level,
# where n = abs(guessed_level - correct_level),
# with `0` for guessing correctly

import argparse
import time
import numpy as np

from ucbq_agent_stateless import UCBQAgent
from thompson_sampling_agent import ThompsonSamplingAgentTemporaryWrapper
from ucbq_environment_stateless import ModifiedRandomEnvironment
from modified_pendulum_processor_noiseless import ModifiedPendulumProcessorNoiseless
import utils
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--TimeOut", help = "Stop script after n seconds")
args = parser.parse_args()
timeOut = float(args.TimeOut) if bool(args.TimeOut) else 1.69

# def default_params():
#     """ These are the default parameters used int eh framework. """
#     return {
#             # # Runner parameters
#             # 'max_episodes': int(1E6),         # experiment stops after this many episodes
#             # 'max_steps': int(1E9),            # experiment stops after this many steps
#             # 'multi_runner': False,            # uses multiple runners if True
#             # # Exploration parameters
#             # 'epsilon_anneal_time': int(5E3),  # exploration anneals epsilon over these many steps
#             # 'epsilon_finish': 0.1,            # annealing stops at (and keeps) this epsilon
#             # 'epsilon_start': 1,               # annealing starts at this epsilon
#             'epsilon': 1,               # annealing starts at this epsilon
#             'epsilon_decay': 0.5,
#             # Optimization parameters
#             'alpha': 0.5,                       # learning rate of optimizer
#             # 'gamma': 0.99,                    # discount factor gamma
#            }

params = default_params()
t = 0

num_actions = 7
# agent = UCBQAgent()
agent = ThompsonSamplingAgentTemporaryWrapper()
env = ModifiedRandomEnvironment()
state = 0

# # Surrogate rewards setup
# from modified_pendulum_processor import ModifiedPendulumProcessor
# post_processor = ModifiedPendulumProcessor(surrogate=True)
# def adjust_rewards(reward, state, action):    
#     observation, reward, done, info = post_processor.process_step(state, reward, None, None, action)
#     return reward

start_time = time.time()

episode_rewards = 0

while True:
    elapsed_time = time.time() - start_time

    # Auto shut down scipt 
    if elapsed_time > timeOut:
        break

    action = agent.choose_action(state) 
    reward, next_state, done = env.step(action)
    print(f"{round(elapsed_time, 2)} > {action} -> {reward}")
    
    # reward = adjust_rewards(reward, state, action)
    
    agent.learn(state, action, reward, next_state)
    episode_rewards += reward

    t += 1
    

    # if done:
    #     break

utils.print_agent_stats(agent)
print(f'Episode rewards: {episode_rewards}')