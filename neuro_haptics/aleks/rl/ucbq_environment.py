# import random
# import time
# import logging
# from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Define the AI stream
# info = StreamInfo('AIStream', 'Markers', 1, 0, 'int32', 'ai_stream')
# outlet = StreamOutlet(info)
# logging.info("AI stream created.")

# # Delay to ensure Participant stream is ready
# time.sleep(5)

# # Resolve the participant stream
# logging.info("Looking for a Participant stream...")
# streams = None
# while streams is None:
#     # streams = resolve_byprop('name', 'ParticipantStream')
#     streams = resolve_byprop('name', 'LabelMaker_labels')
#     if not streams:
#         logging.info("No Participant stream found, retrying...")
#         time.sleep(1)

# inlet = StreamInlet(streams[0])
# logging.info("Participant stream found.")

import numpy as np
# from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
# import time

# random seed will only give persistent results if you re-import the script
# and restart the kernel in the notebook
np.random.seed(69)

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