from pylsl import StreamInfo, StreamOutlet
import time, random, json, os

from SimPhysDataStreamer import SimPhysDataStreamer
from Classifier import Classifier
from LabelMaker import LabelMaker

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
            self.eeg = Classifier('SimPhysDataStream_Lukas', 'eeg_classifier', bci_params['classifier_update_rate'], bci_params['data_srate'], model_path_eeg, 
                bci_params['target_class'], bci_params['chans'], bci_params['threshold'], bci_params['windows'], bci_params['baseline'],
                debug)            
            self.eeg.start()
            
        # init label maker
        self.labels = LabelMaker(environment)
        self.labels.start()

        # resolve event stream
        # self.inlet = StreamInlet(resolve_stream('name', 'Unity_Events')[0])

if __name__ == "__main__":
    
    data_source = input("Real data or simulated data? Enter 'real' or 'simulated': ")
    environment = input("Enter 'explicit' or 'implicit' for environment to be simulated: ")
    
    nah = NahEnvironment(data_source, environment)
    
    while True: # This will be then changed to wait for an experiment marker from the lsl marker stream coming from unity

        # !! This is just here to simulate the questionnaire labels coming from the unity scene
        if environment == 'explicit' and data_source == 'simulated':
            rand_label = str(random.randint(1,7))
            nah.sim_labels.push_sample(rand_label)
            time.sleep(1)