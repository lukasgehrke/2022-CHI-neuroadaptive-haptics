import numpy as np
# random seed will only give persistent results if you re-import the script
# and restart the kernel in the notebook
np.random.seed(69)
class ModifiedRandomEnvironment:
    def __init__(self, num_states=10, params={}):
        # The total number of feedback levels
        self.num_states = num_states
        # The last feedback level sent
        # np.random.seed(69)
        self.current_state = np.random.randint(num_states)
    
        self.num_actions = params.get('num_actions', 7)
        # TODO:
        # LSL here
        # this should be removed, the "correct_action" is now in the
        # participant's head
        # The "right" level of feedback
        self.correct_action = params.get('correct_action', 1)        

    def send_feedback_to_participant_and_get_participant_answer(self, action):
        # TODO
        # LSL here
        # We send the predicted `feedback` (action) to the participant and
        # wait for the participant to answer to "How off was the feedback?"
        # and assig it to the variable `answer`
        
        answer = 0 if action == self.correct_action else -abs(self.correct_action - action)

        # Simulate noise
        if np.random.rand() < 0.3:
            answer += np.random.choice([-1, 1])
        answer = np.clip(answer, -6, 0)

        return answer

    def step(self, action):
        reward = self.send_feedback_to_participant_and_get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = action
        self.current_state = next_state
        return reward, next_state

    def get_num_unique_rewards(self):
        return max(abs(self.num_actions - self.correct_action), abs(self.correct_action + 1))