from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import time, random, json, os
import logging
import numpy as np

from SimPhysDataStreamer import SimPhysDataStreamer
from Classifier import Classifier
from LabelMaker import LabelMaker

# Setup logging
logging.basicConfig(level=logging.INFO)

class NahEnvironment():

    def __init__(self, source, environment) -> None:
        """Simulates the experimental environment of the neuro_haptics project.

        Args:
            source (string): either 'explicit' or 'implicit', describes the source of the ratings
        """

        # !! This is just here to simulate the stream coming from the unity scene that sends the questionnaire answers after every trial
        if environment == 'explicit' and data_source == 'simulated':
            self.sim_labels = StreamOutlet(StreamInfo('Explicit_Labels', 'Markers', 1, 0, 'string', 'myuid34234'))
            time.sleep(2)
        
            logging.info("Participant stream created.")


        # init EEG classifiers that predict the label -> blind to whats going on in unity scene
        elif environment == 'implicit':

            # !! This is just here to simulate the BrainVision RDA data source
            if data_source == 'simulated':
                self.phys_data_streamer = SimPhysDataStreamer()
                self.phys_data_streamer.start()

            # init classifier(s)
            # TODO will need to fix paths and stream names later -> the loaded config here gets generated in an external jupyter notebook that trains the classifier and saves the model
            debug = True
            model_path_eeg = 'example_data'+os.sep+'model_sub-016_eeg.sav'
            with open('example_data'+os.sep+'bci_params.json', 'r') as f:
                bci_params = json.load(f)
            # eeg = Classifier('BrainVision RDA', 'eeg_classifier', bci_params['classifier_update_rate'], bci_params['data_srate'], model_path_eeg, 
            #     bci_params['target_class'], bci_params['chans'], bci_params['threshold'], bci_params['windows'], bci_params['baseline'],
            #     debug)
            self.eeg = Classifier('SimPhysDataStream_Lukas', 'eeg_classifier', bci_params['classifier_updat e_rate'], bci_params['data_srate'], model_path_eeg, 
                bci_params['target_class'], bci_params['chans'], bci_params['threshold'], bci_params['windows'], bci_params['baseline'],
                debug)            
            self.eeg.start()
            
        # init label maker
        self.labels = LabelMaker(environment)
        self.labels.start()

        # resolve event stream
        # self.inlet = StreamInlet(resolve_stream('name', 'Unity_Events')[0])


if __name__ == "__main__":

    # data_source = input("Real data or simulated data? Enter 'real' or 'simulated': ")
    # environment = input("Enter 'explicit' or 'implicit' for environment to be simulated: ")
    environment = 'explicit'
    data_source = 'simulated'

    nah = NahEnvironment(data_source, environment)

    # # Define the participant stream
    # info = StreamInfo('ParticipantStream', 'Markers', 1, 0, 'int32', 'participant_stream')
    # outlet = StreamOutlet(info)
    # logging.info("Participant stream created.")

    # Resolve the AI stream
    logging.info("Looking for an AI stream...")
    streams = None
    while streams is None:
        streams = resolve_byprop('name', 'AIStream')
        if not streams:
            logging.info("No AI stream found, retrying...")
            time.sleep(1)

    inlet = StreamInlet(streams[0])
    logging.info("AI stream found.")    
    
    while True: # This will be then changed to wait for an experiment marker from the lsl marker stream coming from unity

        # # !! This is just here to simulate the questionnaire labels coming from the unity scene
        # if environment == 'explicit' and data_source == 'simulated':

        while True:            
            # Receive a sample from the AI stream
            sample, timestamp = inlet.pull_sample(timeout=2)
            if sample is not None:
                ai_feedback_level = int(sample[0])
                logging.info(f"Received from AI: {ai_feedback_level}")

                # Mock response
                action = ai_feedback_level
                correct_action = 1
                response = 0 if action == correct_action else -abs(correct_action - action)

                # Simulate noise
                if np.random.rand() < 0.3:
                    response += np.random.choice([-1, 1])
                response = np.clip(response, -6, 0)

                response = [str(response)]
                
                nah.sim_labels.push_sample(response)
                       
                logging.info(f"Sent to AI: {response}")

            # Sleep to simulate time between responses
            time.sleep(1)        