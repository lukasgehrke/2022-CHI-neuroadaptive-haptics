import numpy as np
import logging
import json
import logging
import datetime
from collections import Counter


np.random.seed(69)

class UCBQAgent:
    def __init__(self, params={}):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f'logs/log-{current_datetime}.csv'

        logging.basicConfig(filename=log_filename,
                            level=logging.INFO, 
                            format='%(asctime)s, %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        
        headers = "timestamp, t, action, reward, new_Q_value, alpha, epsilon"
        with open(log_filename, 'w') as f:
            f.write(headers + '\n')   


        # In our case actions == states
        self.num_states = params.get('num_states', 7)
        self.num_actions = params.get('num_actions', 7)
        self.alpha = params.get('alpha', 0.5)  # learning rate
        self.alpha_decay_denumerator = params.get('alpha_decay', 40)
        self.alpha_min = params.get('alpha_min', 0.001)
        self.gamma = params.get('gamma', 0.95)  # discount factor
        # TODO: implement decay. Is it compatible with ucb?
        # TODO: Do we need epsilon greedy?
        # Is there any psychological reason why we can't just switch to the
        # next highest level incrementally?
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

        # Epsilon-greedy action selection
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

        alpha_decay = lambda t: np.log10(t+1)/self.alpha_decay_denumerator
        if self.alpha - alpha_decay(self.t) > self.alpha_min:
            self.alpha -= alpha_decay(self.t)

        epsilon_decay = lambda t: np.log10(t+1)/self.epsilon_decay_denumerator
        if self.epsilon  - epsilon_decay(self.t) > self.epsilon_min:
            self.epsilon -= epsilon_decay(self.t)

        return action

    def learn(self, state, action, reward, next_state):
        # Update N-table for action counts
        self.N[state][action] += 1

        # Update reward list
        self.rewards[action].append(reward)

        # Adjust reward
        # reward, _ = Counter(self.rewards[action]).most_common(1)[0]

        # self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        self.Q[state][action] = self.Q[state][action] + (1/self.N[state][action]) * (reward + self.Q[state][action])
        # self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.Q[state][action])
        
        logging.info(f'{self.t}, {action}, {reward}, {self.Q[state][action]}, {self.alpha}, {self.epsilon}')


    def reset(self):
        # Reset Q-table and N-table
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.N = np.ones((self.num_states, self.num_actions))