import numpy as np

def print_agent_stats(agent):
    print(f'Q-table:')
    print(f'{np.around(agent.Q, decimals=4)}')
    print(f'Number of times action was taken:')
    print(f'{agent.N}')
    print(f'Total timesteps:')
    print(agent.t)

def get_num_unique_rewards(num_actions, correct_action):
    return max(abs(num_actions - correct_action), abs(correct_action + 1))