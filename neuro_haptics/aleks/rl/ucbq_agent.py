import numpy as np
import logging
import json
import logging
import datetime
from collections import Counter
import os



# np.random.seed(69)

class UCBQAgent:
    def __init__(self, params={}):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the logs directory
        logs_dir = os.path.join(current_script_dir, 'logs')

        # Ensure the logs directory exists
        os.makedirs(logs_dir, exist_ok=True)

        # Construct the full path for the log file
        log_filename = os.path.join(logs_dir, f'log-{current_datetime}.csv')              
        
        # Create a logger for the agent
        self.logger = logging.getLogger('agent_logger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_filename)

        headers = "timestamp,t,action,reward,reward_adjusted,new_Q_value,alpha,epsilon"
        with open(log_filename, 'w') as f:
            f.write(headers + '\n')  

        fh.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)

        # In our case actions == states
        self.num_states = params.get('num_states', 1)
        self.num_actions = params.get('num_actions', 4)
        self.alpha = params.get('alpha', 0.5)  # learning rate
        self.alpha_decay_denumerator = params.get('alpha_decay', 40)
        self.alpha_min = params.get('alpha_min', 0.001)
        self.gamma = params.get('gamma', 0.95)  # discount factor
        # TODO: implement decay. Is it compatible with ucb?
        # TODO: Do we need epsilon greedy?
        # Is there any psychological reason why we can't just switch to the
        # next highest level incrementally?
        self.n_preemptive_exploration_steps = params.get('preemptive_exploration_steps', 
                                                       0)
        self.epsilon = params.get('epsilon', 1)  # epsilon for epsilon-greedy action selection
        self.epsilon_decay_denumerator = params.get('epsilon_decay', 20)
        self.epsilon_min = params.get('epsilon_min', 0.01)        
        # self.epsilon_decay = lambda t: np.log10(t+1)/params.get('epsilon_decay', 20)
        self.ucb_c = params.get('ucb_c', 2)

        # start_q_value = -(self.num_actions - 1)
        start_q_value = 0
        # Need to set this expilcitly to float, otherwise when we assign the
        # new value to the Q-table, it will be casted to int
        # TODO: However, the performance with the casting (rounding) looked better
        # self.Q = np.full((self.num_states, self.num_actions), start_q_value)
        self.Q = np.full((self.num_states, self.num_actions), float(start_q_value))

        # Initialize N-table for action counts
        # Needs to be `one` to avoid div by zero
        # self.N = np.ones((self.num_states, self.num_actions), dtype=int)
        self.N = np.zeros((self.num_states, self.num_actions), dtype=int)

        self.t = 0

        possible_actions = range(0, self.num_actions, 1)
        self.rewards = {action: [] for action in possible_actions}

    def choose_action(self, state):
        self.t += 1

        # if self.t <= self.n_preemptive_exploration_steps:
        #     N_state = self.N[state]
        #     # Find least taken actions
        #     idxs_min_values = np.flatnonzero(N_state == N_state.min())
        #     action = np.random.choice(idxs_min_values)

        # # Epsilon-greedy action selection        
        # else:

        if np.random.uniform(0, 1) < self.epsilon:
            # Take a random action
            # np.random.seed(69)
            action = np.random.choice(self.num_actions)
        else:
            # Calculate the UCB value for each action
            # TODO: in the original paper there was no `2` but they had a `c`
            # Assign a high value to encourage exploration of this unvisited state
            # ucb_values = np.where(self.N[state] == 0, float('inf'), self.Q[state] + self.ucb_c * np.sqrt(np.log(self.t) / self.N[state]))

            ucb_values = np.where(self.N[state] == 0, 
                                np.inf, 
                                self.Q[state] + self.ucb_c * np.sqrt(np.divide(np.log(self.t), self.N[state], where=self.N[state]!=0)))

            # Select action with maximum UCB value
            # Break ties randomly
            idxs_max_values = np.flatnonzero(ucb_values == ucb_values.max())
            # np.random.seed(69)
            action = np.random.choice(idxs_max_values)
        
        alpha_decay = lambda t: np.log10(t+1-self.n_preemptive_exploration_steps)/self.alpha_decay_denumerator
        if self.alpha - alpha_decay(self.t) > self.alpha_min:
            self.alpha -= alpha_decay(self.t)

        epsilon_decay = lambda t: np.log10(t+1-self.n_preemptive_exploration_steps)/self.epsilon_decay_denumerator
        if self.epsilon  - epsilon_decay(self.t) > self.epsilon_min:
            self.epsilon -= epsilon_decay(self.t)

        return action

    def learn(self, state, action, reward, next_state):
        # Update N-table for action counts
        self.N[state][action] += 1

        # Update reward list
        self.rewards[action].append(reward)

        # # Adjust reward
        # reward_adjusted, _ = Counter(self.rewards[action]).most_common(1)[0]

        # TODO:
        reward_adjusted = reward

        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward_adjusted + self.gamma * np.max(self.Q[next_state]))
        
        self.logger.info(f'{self.t},{action},{reward},{reward_adjusted},{self.Q[state][action]},{self.alpha},{self.epsilon}')

        return self.t, action, reward, reward_adjusted, round(self.Q[state][action], 4), round(self.alpha, 4), round(self.epsilon, 4)