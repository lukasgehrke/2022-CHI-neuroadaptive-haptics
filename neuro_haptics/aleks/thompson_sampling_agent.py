import numpy as np

class ThompsonSamplingAgent:
    def __init__(self, num_actions, reward_range):
        self.num_actions = num_actions
        self.alpha = np.ones(num_actions)  # Initialize alpha (successes) to 1
        self.beta = np.ones(num_actions)   # Initialize beta (failures) to 1
        self.reward_range = reward_range
        

    def choose_action(self):
        sampled_rewards = np.random.beta(self.alpha, self.beta)
        action = np.argmax(sampled_rewards)
        # print('a: ',action)
        return action

    def observe_reward(self, action, reward):
        # print('r: ', reward)
        # Update the Beta distribution parameters based on observed reward
        normalized_reward = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        self.alpha[action] += normalized_reward
        self.beta[action] += 1 - normalized_reward

class ThompsonSamplingAgentTemporaryWrapper(ThompsonSamplingAgent):
    def __init__(self, *args):
        self.epsilon = None
        self.Q = []
        self.N = None
        self.t = None
        # Assuming we stick with 7 actions, this can remain hardcoded
        super().__init__(*args, num_actions=7, reward_range = (-6, 0))
    
    def choose_action(self, state=None):
        return super().choose_action()
    
    def learn(self, state=None, action=None, reward=None, next_state=None):
        return super().observe_reward(action=action, reward=reward)        