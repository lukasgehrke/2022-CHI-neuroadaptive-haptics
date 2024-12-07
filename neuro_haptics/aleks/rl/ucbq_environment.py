import numpy as np

class ModifiedRandomEnvironment:
    def __init__(self, num_states=1, params={}):
        self.num_states = num_states
        # The last feedback level sent
        # np.random.seed(69)
        self.current_state = np.random.randint(num_states)
        
        # The total number of feedback levels        
        self.num_actions = params.get('num_actions', 4)

        # TODO:
        # LSL here
        # this should be removed, the "correct_action" is now in the
        # participant's head
        # The "right" level of feedback
        self.correct_action = params.get('correct_action', 1)

    # "How correct was this action?"
    def get_mock_response(self, action):
        # Mock answers
        # Assuming equally distributed means        
        answer = 1 - abs(self.correct_action - action) * 0.33
        
        answer += np.random.normal(0, 0.25)        
        answer = np.clip(answer, 0.0, 1.0)        

        return answer 

    def send_feedback_to_participant_and_get_participant_answer(self, action):
        return self.get_mock_response(action)

    def step(self, action):
        reward = self.send_feedback_to_participant_and_get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = action
        self.current_state = next_state
        return reward, next_state

    def get_num_unique_rewards(self):
        return max(abs(self.num_actions - self.correct_action), abs(self.correct_action + 1))

    def get_mock_response_negative(ai_feedback_level):
        num_actions = 5
        action = ai_feedback_level
        correct_action = 1
        response = 0 if action == correct_action else -abs(correct_action - action)
        
        # Simulate noise
        if np.random.rand() < 0.75:
            response += np.random.choice([-1, 1])
        response = np.clip(response, -(num_actions-1), 0)

        return response 