import time
import logging
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop


from .ucbq_environment_stateless import ModifiedRandomEnvironment

class UCBQEnvironmentLSL(ModifiedRandomEnvironment):
    def __init__(self, params={}):
        super().__init__(params=params)

        print("UCBQEnvironmentLSL init")

        # # Setup logging
        # logging.basicConfig(level=logging.INFO)    

        # Create a logger for the environment
        self.environment_logger = logging.getLogger('environment_logger')
        self.environment_logger.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.environment_logger.addHandler(ch)            

        # Define the AI stream
        info = StreamInfo('AIStream', 'Markers', 1, 0, 'int32', 'ai_stream')
        self.outlet = StreamOutlet(info)
        self.environment_logger.info("AI stream created.")

        # Delay to ensure Participant stream is ready
        time.sleep(2)

        # Resolve the participant stream
        self.environment_logger.info("Looking for a Participant stream...")
        streams = None
        while streams is None:
            # streams = resolve_byprop('name', 'ParticipantStream')
            streams = resolve_byprop('name', 'LabelMaker_labels')
            if not streams:
                self.environment_logger.info("No Participant stream found, retrying...")
                time.sleep(1)

        self.inlet = StreamInlet(streams[0])
        self.environment_logger.info("Participant stream found.")        
        
    def send_feedback_to_participant_and_get_participant_answer(self, action):
        # TODO
        # LSL here
        # We send the predicted `feedback` (action) to the participant and
        # wait for the participant to answer to "How off was the feedback?"
        # and assign it to the variable `answer`


        answer = None
        
        # Send action to the AI stream
        self.outlet.push_sample([action])
        self.environment_logger.info(f"Sent to Participant: {action}")

        # time.sleep(2)

        # Receive a sample from the Participant stream
        sample, timestamp = self.inlet.pull_sample(timeout=10)
        # while sample is None:
        #     sample, timestamp = inlet.pull_sample(timeout=2)
        #     print("Waiting for response from Participant...")
        
        self.environment_logger.info(f"Received from Participant: {sample[0]}")
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