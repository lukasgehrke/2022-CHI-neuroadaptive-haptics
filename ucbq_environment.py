import time
import random
import numpy as np

class ModifiedRandomEnvironment:
    def __init__(self, correct_action = 6, num_states=10):
        # The total number of feedback levels
        self.num_states = num_states
        # The last feedback level sent
        self.current_state = np.random.randint(num_states)
        # The "right" level of feedback
        self.correct_action = correct_action

    def get_participant_answer(self, action):
        # TODO: use this code when we'll be listenting to the 
        # actual stream
        # 
        # while True:
        #   answer = get_response_from_stream()
        #   if (answer):
        #     return answer
        #     break

        # Simulation code
        # Wait random time before giving an answer.
        # This simulates listening to stream.
        time.sleep(random.uniform(0.001, 0.002))

        # # The reward is 0 if action matches current state, otherwise -1
        # answer = 0 if action == 2 else -1 

        answer = 0 if action == self.correct_action else -1 

        return answer

    def step(self, action):
        reward = self.get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = action
        self.current_state = next_state
        return reward, next_state