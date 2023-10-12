import numpy as np
# random seed will only give persistent results if you re-import the script
# and restart the kernel in the notebook
np.random.seed(69)
class ModifiedRandomEnvironment:
    def __init__(self, correct_action = 6, num_states=10):
        # The total number of feedback levels
        self.num_states = num_states
        # The last feedback level sent
        # np.random.seed(69)
        self.current_state = np.random.randint(num_states)
        # The "right" level of feedback
        self.correct_action = correct_action

    def get_participant_answer(self, action):
        answer = 0 if action == self.correct_action else -abs(self.correct_action - action)

        return answer

    def step(self, action):
        reward = self.get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = action
        self.current_state = next_state
        return reward, next_state