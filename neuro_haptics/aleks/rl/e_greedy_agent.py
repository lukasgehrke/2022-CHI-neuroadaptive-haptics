import numpy as np
from collections import Counter

from ucbq_agent_stateless_validation import UCBQAgent

class EGreedyAgent(UCBQAgent):
    def __init__(self, params={}):
        super().__init__(params=params)
        start_q_value = 0
        self.Q = np.full((self.num_states, self.num_actions), float(start_q_value))        

    def choose_action(self, state):
        self.t += 1

        # Begin different for this agent

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            # Take a random action
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.Q[state]
            # Select action with maximum Q value
            # Break ties randomly
            idxs_max_values = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(idxs_max_values)

        # End different

        # alpha_decay = lambda t: np.log10(t+1)/self.alpha_decay_denumerator
        # if self.alpha - alpha_decay(self.t) > self.alpha_min:
        #     self.alpha -= alpha_decay(self.t)

        # epsilon_decay = lambda t: np.log10(t+1)/self.epsilon_decay_denumerator
        # if self.epsilon  - epsilon_decay(self.t) > self.epsilon_min:
        #     self.epsilon -= epsilon_decay(self.t)

        return action

    def learn(self, state, action, reward, next_state):
        # Update N-table for action counts
        self.N[state][action] += 1

        # Update reward list
        self.rewards[action].append(reward)

        # Adjust reward
        # reward, _ = Counter(self.rewards[action]).most_common(1)[0]

        # self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        # self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.Q[state][action])
        self.Q[state][action] = self.Q[state][action] + (1/self.N[state][action]) * (reward + self.Q[state][action])
        # self.Q[state][action] = np.sum(self.rewards[action]) / self.N[state][action]
        
        # logging.info(f'{self.t}, {action}, {reward}, {self.Q[state][action]}, {self.alpha}, {self.epsilon}')            
