from ucbq_environment import ModifiedRandomEnvironment

class ModifiedRandomEnvironment(ModifiedRandomEnvironment):
    def __init__(self, correct_action = 6):
        # The total number of feedback levels
        self.num_states = 1
        # The last feedback level sent
        self.current_state = 0
        # The "right" level of feedback
        self.correct_action = correct_action

    def step(self, action):
        reward = self.get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = 0
        self.current_state = next_state
        return reward, next_state