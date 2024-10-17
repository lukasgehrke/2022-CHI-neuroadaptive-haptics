import time
import logging
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop

from .ucbq_environment_stateless import ModifiedRandomEnvironment

class UCBQEnvironmentLSL(ModifiedRandomEnvironment):
    def __init__(self, params={}):
        super().__init__(params=params)

        # # Setup logging
        # logging.basicConfig(level=logging.INFO)    

        # Create a logger for the environment
        self.environment_logger = logging.getLogger('environment_logger')
        self.environment_logger.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.environment_logger.addHandler(ch)         

        self.environment_logger.info("UCBQEnvironmentLSL init")           

        # Define the AI stream
        info = StreamInfo('AIStream', 'Markers', 1, 0, 'string', 'ai_stream')
        self.outlet = StreamOutlet(info)
        self.environment_logger.info("AI stream created.")

        # Resolve the participant stream
        self.environment_logger.info("Looking for a Participant stream...")
        streams = None
        while streams is None:
            # streams = resolve_byprop('name', 'ParticipantStream')
            # streams = resolve_byprop('name', 'implicit_labels')
            streams = resolve_byprop('name', 'implicit_labels')
            if not streams:
                self.environment_logger.info("No Participant stream found, retrying...")
                time.sleep(1)

        self.inlet = StreamInlet(streams[0])
        self.environment_logger.info("Participant stream found.")

        # Delay to ensure Participant stream is ready
        time.sleep(5)        
        
    def send_feedback_to_participant_and_get_participant_answer(self, action):
        # We send the predicted `feedback` (action) to the participant and
        # wait for the participant to answer to "How off was the feedback?"
        # and assign it to the variable `answer`

        outgoing_sample = [str(action)]
        self.outlet.push_chunk(outgoing_sample)
        # print(f"Sent to Participant: {outgoing_sample}")
        self.environment_logger.info(f"t: {self.t} - > {action}") # sent to Participant
        
        incoming_sample = None
        timestamp = None

        while incoming_sample is None or len(incoming_sample) == 0:
            incoming_sample, timestamp = self.inlet.pull_chunk(timeout=0.0)

        # print(f"Received from Participant: {incoming_sample} at {timestamp}")

        # 1 (completely disagree)
        # 2 (disagree)
        # 3 (neither disagree nor agree)
        # 4 (agree)
        # 5 (strongly agree)
        answer = int(incoming_sample[0][0])
      

        # print(f"Received from Participant: {answer} at {timestamp[0]}")
        self.environment_logger.info(f"t: {self.t} - < {answer}")

        # Sleep to simulate time between responses
        time.sleep(1)

        # Convert answer to format expected by the agent (negative values)
        num_actions = 5
        reward = answer - num_actions

        return reward