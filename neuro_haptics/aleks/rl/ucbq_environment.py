import random
import time
import logging
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the AI stream
info = StreamInfo('AIStream', 'Markers', 1, 0, 'int32', 'ai_stream')
outlet = StreamOutlet(info)
logging.info("AI stream created.")

# Delay to ensure Participant stream is ready
time.sleep(5)

# Resolve the participant stream
logging.info("Looking for a Participant stream...")
streams = None
while streams is None:
    # streams = resolve_byprop('name', 'ParticipantStream')
    streams = resolve_byprop('name', 'LabelMaker_labels')
    if not streams:
        logging.info("No Participant stream found, retrying...")
        time.sleep(1)

inlet = StreamInlet(streams[0])
logging.info("Participant stream found.")

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import time

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
        
        self.ai_feedback_levels = StreamOutlet(StreamInfo('ai_feedback_levels', 'Markers', 1, 0, 'string', 'myuid34234'))
        time.sleep(1)

        self.labelmaker_labels = StreamInlet(resolve_stream('name', 'LabelMaker_labels')[0])
        time.sleep(1)

    def send_feedback_to_participant_and_get_participant_answer(self, action):
        # TODO
        # LSL here
        # We send the predicted `feedback` (action) to the participant and
        # wait for the participant to answer to "How off was the feedback?"
        # and assign it to the variable `answer`


        answer = None
        
        # Send a random number to the AI stream
        number = random.randint(0, 100)
        outlet.push_sample([number])
        logging.info(f"Sent to Participant: {number}")

        # Receive a sample from the Participant stream
        sample, timestamp = inlet.pull_sample(timeout=2)
        while sample is None:
            sample, timestamp = inlet.pull_sample(timeout=2)
            print("Waiting for response from Participant...")
        
        logging.info(f"Received from Participant: {sample[0]}")
        answer = int(sample[0])

        # Sleep to simulate time between responses
        time.sleep(1)

        return answer
        
        # # push action to stream
        # self.ai_feedback_levels.push_sample(str(action))
        # print(f"AI sent feedback: {action}")
        # time.sleep(1)

        # # pull answer from stream
        # answer = self.labelmaker_labels.pull_sample()

        # Mock answers
        # answer = 0 if action == self.correct_action else -abs(self.correct_action - action)

        # # Simulate noise
        # if np.random.rand() < 0.3:
        #     answer += np.random.choice([-1, 1])
        # answer = np.clip(answer, -6, 0)        

        # return answer

    def step(self, action):
        reward = self.send_feedback_to_participant_and_get_participant_answer(action)
        # Our case action == state, but migth consdier this separation in the future
        next_state = action
        self.current_state = next_state
        return reward, next_state

    def get_num_unique_rewards(self):
        return max(abs(self.num_actions - self.correct_action), abs(self.correct_action + 1))