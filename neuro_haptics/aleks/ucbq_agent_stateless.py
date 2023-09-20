from ucbq_agent import UCBQAgent
class UCBQAgent(UCBQAgent):
    def __init__(self, num_actions=7, alpha=0.5, gamma=0.95, epsilon=1.0, params={'alpha': 0.5, 'epsilon': 1.0}):
        super().__init__(num_states=1, num_actions=num_actions, alpha=alpha, gamma=gamma, epsilon=epsilon, params=params)