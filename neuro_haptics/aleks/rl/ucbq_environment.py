import numpy as np

class ModifiedRandomEnvironment:
    def __init__(self, num_states=10, params={}):
        self.num_states = num_states
        # The last feedback level sent
        # np.random.seed(69)
        self.current_state = np.random.randint(num_states)
        
        # The total number of feedback levels        
        self.num_actions = params.get('num_actions', 5)

        # TODO:
        # LSL here
        # this should be removed, the "correct_action" is now in the
        # participant's head
        # The "right" level of feedback
        self.correct_action = params.get('correct_action', 1)

    # Old
    def send_feedback_to_participant_and_get_participant_answer(self, action):
        # Mock answers
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

    def get_mock_response(ai_feedback_level):
        num_actions = 5
        action = ai_feedback_level
        correct_action = 1

        # Adjust for unity response
        # 1 (completely disagree)
        # 2 (disagree)
        # 3 (neither disagree nor agree)
        # 4 (agree)
        # 5 (strongly agree)
        response = 5 - abs(correct_action - action) 
        
        # Simulate noise
        if np.random.rand() < 0.3:
            response += np.random.choice([-1, 1])
        response = np.clip(response, 1, num_actions)
      

        return response        