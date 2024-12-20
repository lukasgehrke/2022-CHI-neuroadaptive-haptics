def default_params():
    """ These are the default parameters used in the framework. """
    return {
            # Runner parameters
            'max_steps': 1000,
            'num_episodes': 100,
            'num_actions': 4, 
            'convergence_count_start': 0,
            'convergence_consecutive_limit': 10,
            # Optimization parameters
            'alpha': 0.5,
            'alpha_decay': 40,
            'alpha_min': 0.001,
            # Exploration parameters
            'epsilon': 1,
            'epsilon_decay': 20,
            'epsilon_min': 0.01,    
            'gamma': 0.95,
            'plots': False,
            'noise': False,
            'surrogate': False,
            'surrogate_c_interval': 10,
            'surrogate_c_interval_min': 30,
            # Simulation
            'correct_action': 1,    # Zero indexed
           }

optimized_params = {'alpha': 0.19008773103975352, 'alpha_decay': 348.142389538547, 'alpha_min': 0.9826884932389596, 'epsilon': 0.28921143036114305, 'epsilon_decay': 445.3413704431987, 'epsilon_min': 0.4278782749167904, 'gamma': 0.9096496016555845, 'surrogate_c_interval': 50.65927148168811, 'surrogate_c_interval_min': 67.48408958504444}

# Experiment print methods

import numpy as np

def print_agent_stats(agent):
    print('Total rewards:')
    print(sum(sum(agent.rewards.values(), [])))
    print(f'Q-table:')
    print(f'{np.around(agent.Q, decimals=4)}')
    print(f'Number of times action was taken:')
    print(f'{agent.N}')
    print(f'Total timesteps:')
    print(agent.t)

def get_mean_across_episodes(arr):
    min_cols = np.amin([len(row) for row in arr])
    truncated_arr = [ x[:min_cols] for x in arr ]
    res = np.array(truncated_arr)
    mean = res.mean(axis=0)

    return mean    

def get_cumsum_rewards(rewards):
    cumulative_sum_rewards = np.cumsum(rewards)
    time_steps = np.arange(1, len(rewards) + 1)
    mean_rewards = cumulative_sum_rewards / time_steps
    return mean_rewards


from .ucbq_agent_stateless import UCBQAgent
from .ucbq_environment_stateless import ModifiedRandomEnvironment
# from .modified_pendulum_processor import ModifiedPendulumProcessor

def runner(adjust_rewards=None, 
           agent=None,
           env=None,
           params={}):

    # TODO: cleaner
    agent = UCBQAgent(params=params) if agent is None else agent(params=params) if callable(agent) else agent
    env = env if env else ModifiedRandomEnvironment(params=params)

    episode_rewards = 0
    rewards = []
    alphas = []
    epsilons = []
    q_values_for_chart = []
    
    t = 0
    # start_action = params.get('start_action', 0)
    # action = start_action
    state = 0
    max_steps = params.get('max_steps', 120)
    plots = params.get('plots', True)
    noise = params.get('noise', False)
    surrogate = params.get('surrogate', False)
    surrogate_c_interval = params.get('surrogate_c_interval', 10)
    diag = params.get('diag', 0.5)
    
    reward_processor = None

    # Disabling noise and surrogate in this place for now
    
    # # surrogate can only be with noise=True for now
    # if surrogate:
    #     noise = True
    
    # if noise: 
    #     #TODO: should we keep/carry over the estimated confusion matrix across all episodes?
    #     num_unique_rewards = env.get_num_unique_rewards()
    #     reward_processor = ModifiedPendulumProcessor(num_unique_rewards=num_unique_rewards,
    #                                                  diag=diag,
    #                                                  params=params)

    if plots:
        sum_q_values_across_states = np.around(np.ravel(agent.Q), decimals=4)
        q_values_for_chart.append(sum_q_values_across_states)

    while True:
        if t == max_steps:
            break

        action = agent.choose_action(state)
        reward, next_state, done = env.step(action)        
        
        if done:
            sum_q_values_across_states = np.around(np.ravel(agent.Q), decimals=4)
            q_values_for_chart.append(sum_q_values_across_states)            
            break     

        rewards.append(reward)
        alphas.append(agent.alpha)
        epsilons.append(agent.epsilon)

        if noise or surrogate:
            observation, reward, done, info = reward_processor.process_step(state, reward, None, None, action)
        
        agent.learn(state, action, reward, next_state)
        episode_rewards += reward
        t += 1

        if plots:
            if t % 10 == 0:
                sum_q_values_across_states = np.around(np.ravel(agent.Q), decimals=4)
                q_values_for_chart.append(sum_q_values_across_states)
      
                
    episode_length = t + 1
    selected_action = action 
    
    if t == max_steps - 1:
        # If we reached the end of the episode
        # select the action with the highest Q-values as the correct one
        selected_action = np.argmax(np.ravel(agent.Q))

    return q_values_for_chart, rewards, episode_length, selected_action, reward_processor, alphas, epsilons

from tqdm import tqdm 

def qLearningExperiment(params={}):
    agent = params.get('agent', None)
    plots = params.get('plots', True)
    num_episodes = params.get('num_episodes', 100)
    correct_action = params.get('correct_action', 3)

    q_values_all_experiments = []
    rewards_all_experiments = []
    episode_lengths = []
    selected_actions = []

    for i in tqdm(range(num_episodes)):
        # TODO: .reset() instead of re-creating?
        env = ModifiedRandomEnvironment(params=params)
        q_values_for_chart, rewards, episode_length, selected_action, reward_processor, _, _ = runner(env=env, agent=agent, params=params)
        selected_actions.append(selected_action)
        episode_lengths.append(episode_length)
                
        rewards_all_experiments.append(rewards)
        q_values_all_experiments.append(q_values_for_chart)

    correct_count = selected_actions.count(correct_action)
    accuracy = (correct_count / len(selected_actions)) * 100
    
    return q_values_all_experiments, rewards_all_experiments, episode_lengths, selected_actions, accuracy, reward_processor

import matplotlib.pyplot as plt
import pandas as pd

def get_mean_rewards_across_episodes(rewards_all_experiments):
    all_mean_rewards = [ get_cumsum_rewards(rewards) for rewards in rewards_all_experiments ]

    all_mean_rewards = pd.DataFrame(all_mean_rewards) # rewards have different lengths
    # because they terminate earlier sometimes
    mean_matrix = np.mean(all_mean_rewards, axis=0)

    return mean_matrix    

def plot_mean_q_values(params={}):
    params_new = {
        'plots': True,
        }
    
    params = params | params_new    

    q_values_all_experiments, rewards_all_experiments, episode_lengths, selected_actions, accuracy, last_reward_processor = qLearningExperiment(params=params)
    print(f'Accuracy: {accuracy}')    
    print(f'Mean episode length: {np.mean(episode_lengths)}')

    mean_matrix = get_mean_rewards_across_episodes(rewards_all_experiments)
    mean_rewards_across_episodes = pd.DataFrame(mean_matrix)
    mean_matrix = get_mean_across_episodes(q_values_all_experiments)
    mean_q_values_across_episodes = pd.DataFrame(mean_matrix) 

    if params.get('noise', False): 
        print('Last reward processor:')
        last_reward_processor.print()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    mean_rewards_across_episodes.plot(ax=axes[0, 0], title='Reward (mean across all episodes)')
    
    ax = plt.subplot(2, 2, 2)
    ax.set_ylabel('Q-value (mean across all episodes)')
    lines = ax.plot(mean_q_values_across_episodes)
    ax.legend(lines, mean_q_values_across_episodes.columns)
    
    ax = plt.subplot(2, 2, 3)
    ax.hist(episode_lengths, bins=10, edgecolor='black')
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Frequency')    

    pd.Series(selected_actions).value_counts().sort_index().plot.bar(ax=axes[1, 1], title='Guessed correct action')
    
    plt.tight_layout()
    plt.show()

    return mean_matrix[-1]