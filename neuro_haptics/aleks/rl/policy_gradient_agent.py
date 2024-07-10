import numpy as np
from collections import Counter

from ucbq_agent_stateless_validation import UCBQAgent

class PolicyGradientAgent(UCBQAgent):
    def __init__(self, params={}):
        super().__init__(params=params)
        self.H = np.zeros(self.num_actions)

    def softmax(self, H):
        return np.exp(H) / np.sum(np.exp(H))
    
    def choose_action(self, state):
        action = np.argmax(self.softmax(self.H))

        return action

    def learn(self, state, action, reward, next_state):
        self.rewards[action].append(reward)

        for i in range(self.num_actions):
            if i == action:
                self.H[i] += self.alpha * (reward - np.mean(self.rewards[action])) * (1 - self.softmax(self.H[i]))
            else:
                self.H[i] -= self.alpha * (reward - self.Q[state, action]) * self.softmax(self.H[i])