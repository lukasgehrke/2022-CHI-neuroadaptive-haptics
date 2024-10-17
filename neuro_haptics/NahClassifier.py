from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import pickle, time, os, json
import numpy as np
import mne

class NahClassifier:
    def __init__(self, model_path):

        # Load the pre-trained model
        self.model = pickle.load(open(model_path, 'rb'))
        # load boundaries for the classifier
        self.boundaries = pickle.load(open(model_path.replace('model', 'boundaries'), 'rb'))

        # load bci_params json file from the model_path
        with open(model_path.replace('model', 'bci_params').replace('.sav', '.json')) as f:
            bci_params = json.load(f)
        self.target_class = bci_params['target_class']
    
        # resolve streams
        streams = None
        while streams is None:

            streams = resolve_byprop('name', 'BrainVision RDA')
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)

            # init EEG stream inlet
            self.eeg_inlet = StreamInlet(streams[0])

            streams = resolve_byprop('name', 'NAH_GazeBehavior')
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)

            # init EYE stream inlet
            self.eye_inlet = StreamInlet(streams[0])

            streams = resolve_byprop('name', 'NAH_Unity3DEvents')
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)

            # init marker stream inlet
            self.marker_inlet = StreamInlet(streams[0])

        # set up outlet for sending predictions
        self.labels = StreamOutlet(StreamInfo('implicit_labels', 'Markers', 1, 0, 'string', 'myuid34234'))

        # in order to use MNE create empty raw info object
        # self.mne_raw_info = mne.create_info(ch_names=[f"EEG{n:01}" for n in range(1, 66)],  ch_types=["eeg"] * 65, sfreq=self.srate)
    
    def predict(self, features):
        
        prediction = int(self.model.predict(features)[0]) #predicted class
        probs = self.model.predict_proba(features) #probability for class prediction
        probs_target_class = probs[0][int(self.target_class)]
        score = self.model.transform(features)[0][0]

        return prediction, probs_target_class, score
        
    def discretize(self, prediction, probability, score):

        if prediction == self.target_class:

            # find the boundaries for the classifier
            for i, boundary in enumerate(self.boundaries):
                if score < boundary:
                    bin = i
                    break

        return bin
        
    def get_data(self):

        # Initialize an empty list to store the pulled samples
        all_eeg_data = []
        all_eye_data = []

        # Continue pulling data until we have exactly 250 samples
        grab_time = time.time()
        while len(all_eeg_data) < 108:
            eeg_data, _ = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=108 - len(all_eeg_data))
            all_eeg_data.extend(eeg_data)
            
            eye_data, _ = [] #self.eye_inlet.pull_chunk(timeout=0.0, max_samples=108 - len(all_eeg_data))
            all_eye_data.extend(eye_data)

            marker_sample, _ = self.marker_inlet.pull_sample(timeout=0.0)
            if marker_sample and 'focus:in;object: PlacementPos' in marker_sample[0]:
                fix_delay = time.time() - grab_time

        # Convert the list to a numpy array
        eeg_data = np.array(all_eeg_data).T
        eye_data = np.array(eye_data).T

        return eeg_data , eye_data, fix_delay

    def compute_features(self, data, modality):
        
        # mne_data = mne.io.RawArray(data.T, self.mne_raw_info) # ! if MNE object is needed for feature extraction

        # data indeces for baseline correction
        baseline_start = 0
        baseline_end = 12

        # windows and window size for feature extraction
        num_windows = 8
        window_size = 12

        # channels for eye data
        gaze_direction_chans = np.arange(2,5)
        gaze_validity_chan = 10

        # eeg processing and feature extraction
        if modality == 'eeg':
            
            # Filter the data, data will be 64 channels x 250 samples
            # data = self.filter_data(data)

            # baseline correction
            baseline = data[:,baseline_start:baseline_end].mean(axis=1)
            data = data - baseline[:,None]

            # discard baseline
            data = data[:,baseline_end:]

            # Compute the features
            reshaped_erp = data.reshape(data.shape[0], num_windows, window_size)
            eeg_features = reshaped_erp.mean(axis=2).flatten() #.reshape(1,-1)

            return eeg_features

        # eye processing and feature extraction
        elif modality == 'eye':

            gaze_velocity = np.zeros((data.shape[0], data.shape[1] - 1))

            tmp = np.diff(data[gaze_direction_chans, :], axis=1)
            gaze_velocity = np.sqrt(np.sum(tmp**2, axis=0))

            invalid_samples = data[gaze_validity_chan, :-1] == 1
            gaze_velocity[invalid_samples] = np.nan
            # repead last value to keep the same length
            gaze_velocity = np.append(gaze_velocity, gaze_velocity[-1])

            # Example: Compute windowed means
            gaze_velocity = gaze_velocity[baseline_end:]
            reshaped_gaze = gaze_velocity.reshape(num_windows, window_size)
            gaze_features = reshaped_gaze.mean(axis=1).flatten() #.reshape(1, -1)

            return gaze_features

    # def choose_nah_label(self):

    #     # Get the data
    #     data = self.get_data()

    #     # Compute the features
    #     features = self.compute_features(data)

    #     # Predict the class
    #     prediction = self.predict(features)

    #     discrete_prediction = self.discretize(prediction)

    #     return prediction
    
    def send_nah_label_to_ai(self, prediction):
        
        # use LSL to send the prediction to the AI
        # LabelMaker_labels

        # does it wait for a certain event to send the prediction? so is there a specific condition for the agent
        # to only listen to the prediction at a certain time?
        self.labels.push_sample(prediction)



# main
if __name__ == "__main__":

    id = 1
    pID = 'sub-' + "%01d" % (id)
    # path = '/Volumes/Lukas_Gehrke/NAH/data/5_single-subject-EEG-analysis'
    path = r'P:\Lukas_Gehrke\NAH\data\5_single-subject-EEG-analysis'

    model_path = path+os.sep+pID+os.sep+'model.sav'
    
    classifier = NahClassifier(model_path)

    # Define a flag to track execution state
    has_executed = False

    while True:

        marker = classifier.marker_inlet.pull_sample()[0]

        # what if there are two grab markers
        if marker and 'What:grab' in marker[0] and not has_executed:
            
            # eeg = classifier.get_data()
            eeg, eye, fix_delay = classifier.get_data()
            eeg_feat = classifier.compute_features(eeg, 'eeg')
            eye_feat = classifier.compute_features(eeg, 'eye')

            # test_fix_delay = np.array([0.4])

            # concatenate eeg, eye, fix_delay
            feature_vector = np.concatenate((eeg_feat, eye_feat, fix_delay), axis=0).reshape(1, -1)

            # pred
            prediction = classifier.predict(feature_vector)

            classifier.send_nah_label_to_ai(prediction)

            # Set the flag to indicate the code has been executed
            has_executed = True