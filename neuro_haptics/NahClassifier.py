from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import pickle, time, os, json
import numpy as np
import mne
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class NahClassifier:
    def __init__(self, model_path):

        # Initialize the model and imputer
        self.model = LinearDiscriminantAnalysis()
        self.imputer = SimpleImputer(strategy='mean')
        self.expected_num_features = 521  # Set this to the number of features the model expects
        
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
            print("Looking for EEG stream...")	
            streams = resolve_byprop('name', 'BrainVision RDA')
            
            # Don't think this is necessary, since resolve_prop will wait until it finds the stream
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)
            
            print("EEG stream found!")

            # init EEG stream inlet
            self.eeg_inlet = StreamInlet(streams[0])

            
            print("Looking for EYE stream...")
            streams = resolve_byprop('name', 'NAH_GazeBehavior')
            
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)

            print("EYE stream found!")

            # init EYE stream inlet
            self.eye_inlet = StreamInlet(streams[0])

                        
            print("Looking for Marker stream...")
            streams = resolve_byprop('name', 'NAH_Unity3DEvents')
            if not streams:
                print("No EEG stream found, retrying...")
                time.sleep(1)

            
            print("Marker stream found!")
            # init marker stream inlet
            self.marker_inlet = StreamInlet(streams[0])

        # set up outlet for sending predictions
        self.labels = StreamOutlet(StreamInfo('labels', 'Markers', 1, 0, 'string', 'myuid34234'))

        # in order to use MNE create empty raw info object
        # self.mne_raw_info = mne.create_info(ch_names=[f"EEG{n:01}" for n in range(1, 66)],  ch_types=["eeg"] * 65, sfreq=self.srate)
    
    def predict(self, features):
        # Impute missing values
        features = self.imputer.fit_transform(features)

        # Adjust the number of features to match the model's expected input shape
        if features.shape[1] < self.expected_num_features:
            # Pad with zeros if there are fewer features
            padding = np.zeros((features.shape[0], self.expected_num_features - features.shape[1]))
            features = np.hstack((features, padding))
        elif features.shape[1] > self.expected_num_features:
            # Truncate if there are more features
            features = features[:, :self.expected_num_features]

        # Predict the class
        prediction = int(self.model.predict(features)[0])  # predicted class
        probs = self.model.predict_proba(features)  # probability for class prediction
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
        fix_delay = 0
        grab_time = time.time()

        # Cannot test this right now because only restreaming one stream at a time and not all three in parallel in my test environment
        while len(all_eeg_data) < 108:
            eeg_data, _ = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=108 - len(all_eeg_data))
            all_eeg_data.extend(eeg_data)
            
            eye_data, _ = self.eye_inlet.pull_chunk(timeout=0.0, max_samples=108 - len(all_eye_data))
            all_eye_data.extend(eye_data)

            marker_sample, _ = self.marker_inlet.pull_sample(timeout=0.0)
            if marker_sample and 'focus:in;object: PlacementPos' in marker_sample[0]:
                fix_delay = time.time() - grab_time

        # Convert the list to a numpy array
        eeg_data = np.array(all_eeg_data).T
        eye_data = np.array(all_eye_data).T

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

            # Example values for num_windows and window_size
            num_windows = 8
            window_size = 12

            # Compute the required size
            required_size = num_windows * window_size

            # Print the current size of gaze_velocity
            print(f"Size of gaze_velocity: {gaze_velocity.size}")

            # Pad gaze_velocity if its size is less than the required size
            if gaze_velocity.size < required_size:
                padding_size = required_size - gaze_velocity.size
                gaze_velocity = np.pad(gaze_velocity, (0, padding_size), 'constant', constant_values=np.nan)
                print(f"Padded gaze_velocity to size: {gaze_velocity.size}")

            # Truncate gaze_velocity if its size is greater than the required size
            elif gaze_velocity.size > required_size:
                gaze_velocity = gaze_velocity[:required_size]
                print(f"Truncated gaze_velocity to size: {gaze_velocity.size}")

            # Reshape gaze_velocity
            reshaped_gaze = gaze_velocity.reshape(num_windows, window_size)
            print(f"Reshaped gaze_velocity to shape: {reshaped_gaze.shape}")

            # Compute gaze features
            gaze_features = reshaped_gaze.mean(axis=1).flatten()
            print(f"Gaze features: {gaze_features}")

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
        prediction = str(prediction)
        self.labels.push_sample(prediction)



# main
if __name__ == "__main__":

    id = 1
    pID = 'sub-' + "%01d" % (id)
    # path = '/Volumes/Lukas_Gehrke/NAH/data/5_single-subject-EEG-analysis'
    # path = '/Users/lukasgehrke/data/NAH/data/5_single-subject-EEG-analysis/'
    path = r'P:\Lukas_Gehrke\NAH\data\5_single-subject-EEG-analysis'

    model_path = path+os.sep+pID+os.sep+'model.sav'
    
    classifier = NahClassifier(model_path)

    last_grab_number = -1

    start_print_time = time.time()
    last_print_time = start_print_time
    
    while True:
        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_print_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            formatted_time = f"{minutes:02}:{seconds:02}"
            print(f"{formatted_time} - Waiting for grab marker...")
            last_print_time = current_time

        marker = classifier.marker_inlet.pull_sample()[0]

        # what if there are two grab markers
        if marker and 'What:' in marker[0]:
            marker_data = marker[0].split(';')
            marker_dict = {item.split(':')[0]: item.split(':')[1] for item in marker_data}
            what = marker_dict.get('What')
            
            if what == 'grab':
                current_grab_number = int(marker_dict.get('Number', 0))

                if current_grab_number > last_grab_number:           
                    print(f"Grab {current_grab_number} detected: ", marker)

                    eeg, eye, fix_delay = classifier.get_data()
                    
                    eeg_feat = classifier.compute_features(eeg, 'eeg')
                    eye_feat = classifier.compute_features(eye, 'eye')

                    # for tests
                    # eeg = classifier.get_data()
                    # eye = classifier.get_data()            
                    # eye_feat = np.zeros(8)
                    # test_fix_delay = np.array([0.4])

                    # concatenate eeg, eye, fix_delay
                    feature_vector = np.concatenate((eeg_feat, eye_feat, [fix_delay]), axis=0).reshape(1, -1)

                    # pred
                    prediction, probs_target_class, score = classifier.predict(feature_vector)

                    print(f'Classifier stuff {prediction}, {probs_target_class}, {score}')
                    
                    # Map probs_target_class to an int value in the range 1 to 5
                    mapped_label = max(1, int(np.ceil(probs_target_class * 5)))

                    classifier.send_nah_label_to_ai(mapped_label)
                    
                    print("Label sent to AI: ", mapped_label)