# This script runs the learning loop. It is responsible for:
# 1. instantiatiating the agent, which sends the first action (feedback)
# 2. instantiating the environment, which sends the action (feedback) to the participant, and receies the rewards (answers) from the participant
# 3. and then sending these rewards to the agent.

# This script currently simulates the participant answer
# Run the script with
# `python run_experiment.py -t 1`
# This will run a simulation trial, with a total length of 1 seconds.

# The agent is rewarded `-n` for guessing the wrong feedback level,
# where n = abs(guessed_level - correct_level),
# with `0` for guessing correctly

# The script will output:
# - A list of actions taken, in the format: 
# `step: curernt_step, time: elapsed_time > action_taken -> reward_received`
# - The Q-table at the end of the trial
# - The total count of given actions taken
# - Total reward obtained (lower is better)


import argparse
import time

from rl.ucbq_agent_stateless import UCBQAgent

import rl.utils as utils
from rl.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--TimeOut", help = "Stop script after n seconds")
args = parser.parse_args()

timeOut = None
max_steps = 120

params = default_params()
agent = UCBQAgent()

from rl.ucbq_environment_lsl import UCBQEnvironmentLSL
env = UCBQEnvironmentLSL()
names = ['t', 'action', 'reward', 'reward_adjusted', 'new_Q_value', 'alpha', 'epsilon']

# State is fixed to 0
state = 0
t = 0
start_time = time.time()

while True:
    if t == max_steps:
        break

    elapsed_time = time.time() - start_time

    # Auto shut down script 
    if timeOut and elapsed_time > timeOut:
        break

    action = agent.choose_action(state) 
    reward, next_state, done = env.step(action)
       
    learn_response = agent.learn(state, action, reward, next_state)

    zipped = dict(zip(names, learn_response))
    formatted_zipped = ', '.join([f'{key}: {value}' for key, value in zipped.items()])
    env.environment_logger.info(f't: {learn_response[0]} - l - {formatted_zipped}')

    t += 1
    
    # if done:
    #     break

utils.print_agent_stats(agent)