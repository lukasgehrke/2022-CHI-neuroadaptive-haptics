# Run the script with
# `python run_experiment.py -t 1`
# This will run a simulation trial, with a total length of 1 seconds.
# Participant answere will be given every 0.001 to 0.002 seconds.
# The participant "ture" level of feedback is 2.

# The script will output:
# - A list of actions taken, in the format: 
# `elapsed_time > action_taken -> reward_received`
# - The Q-table at the end of the trial,
# with rows representing states and columns representing acions
# - The total count of given actions taken in a given state, 
# with rows representing states and columns representing acions
# - Total timesteps (actions) taken
# - Total reward obtained

# The agent is rewrded `-1` for guessing the wrong feedback level (not 2),
# and `0` for guessing correctly

import argparse
from ucbq_agent import UCBQAgent
from ucbq_environment import ModifiedRandomEnvironment
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--TimeOut", help = "Stop script after n seconds")
args = parser.parse_args()

num_states = 10
agent = UCBQAgent(num_states=num_states, num_actions=num_states)
state = 1
env = ModifiedRandomEnvironment(current_state = state, num_states=num_states)

start_time = time.time()

episode_rewards = 0

while True:
    elapsed_time = time.time() - start_time

    # Auto shut down scipt 
    if bool(args.TimeOut) and (elapsed_time > float(args.TimeOut)):
        break

    action = agent.choose_action(state) 
    # TODO: 
    # send_action_to_stream
    reward, next_state = env.step(action)
    
    print(f"{round(elapsed_time, 2)} > {action} -> {reward}")
    
    agent.learn(state, action, reward, next_state)
    state = next_state

    episode_rewards += reward   

print(f'Q-table:')
print(f'{np.around(agent.Q, decimals=4)}')
print(f'Number of times action was taken:')
print(f'{agent.N}')
print(f'Total timesteps: {sum(sum(agent.N)) - 100}')
print(f'Episode rewards: {episode_rewards}')