import numpy as np

class ThompsonSamplingAgent:
    def __init__(self, num_actions, reward_range):
        self.num_actions = num_actions
        self.alpha = np.ones(num_actions)  # Initialize alpha (successes) to 1
        self.beta = np.ones(num_actions)   # Initialize beta (failures) to 1
        self.reward_range = reward_range

        self.counts = [0 for col in range(num_actions)]
        self.values = [0.0 for col in range(num_actions)]
        

    def choose_action(self):
        sampled_rewards = np.random.beta(self.alpha, self.beta)
        action = np.argmax(sampled_rewards)
        # print('a: ',action)
        return action

    def observe_reward(self, action, reward):
        # update counts pulled for chosen action
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        
        # Update average/mean value/reward for chosen action
        value = self.values[action]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value

        # print('r: ', reward)
        # Update the Beta distribution parameters based on observed reward
        normalized_reward = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        self.alpha[action] += normalized_reward
        self.beta[action] += 1 - normalized_reward

class ThompsonSamplingAgentTemporaryWrapper(ThompsonSamplingAgent):
    def __init__(self, *args, **kwargs):
        self.epsilon = None
        self.t = None
        # Assuming we stick with 7 actions, this can remain hardcoded
        super().__init__(*args, num_actions=7, reward_range = (-6, 0))
        self.Q = self.values
        self.N = self.counts
    
    def choose_action(self, state=None):
        a = super().choose_action()
        self.N = self.counts
        return a
    
    def learn(self, state=None, action=None, reward=None, next_state=None):
        l = super().observe_reward(action=action, reward=reward)
        self.Q = self.values
        return l