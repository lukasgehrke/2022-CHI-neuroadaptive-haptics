from rl.ucbq_agent import UCBQAgent
class UCBQAgent(UCBQAgent):
    def __init__(self, params={}):
        params['num_states'] = 1
        super().__init__(params=params)