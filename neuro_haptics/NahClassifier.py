from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import pickle, time, os, json
import numpy as np
import mne
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from bci_funcs import windowed_mean, compute_gaze_velocity

import concurrent.futures

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
        
    def discretize(self, prediction, score):

        # if prediction == self.target_class:

        # find the boundaries for the classifier
        for i, boundary in enumerate(self.boundaries):
            if score < boundary:
                bin = i
                break

        return bin
        
    def get_data(self, type):

        if type == 'eeg':
            all_eeg_data = np.empty((0, 64))
        elif type == 'eye':
            all_eye_data = np.empty((0, 11))
        elif type == 'marker':
            fix_delay = 0

        # Continue pulling data until we have exactly 1s of samples
        pull_time = time.time()
        while time.time() - pull_time < 1.0: # 1 second window to grab data

            # Pull data
            if type == 'eeg':
                eeg_data, _ = self.eeg_inlet.pull_sample(timeout=0.1)
                eeg_data = np.array(eeg_data).reshape(1, -1)
                all_eeg_data = np.vstack([all_eeg_data, eeg_data])
            elif type == 'eye':
                eye_data, _ = self.eye_inlet.pull_sample(timeout=0.1)
                eye_data = np.array(eye_data).reshape(1, -1)
                all_eye_data = np.vstack([all_eye_data, eye_data])
            elif type == 'marker':
                marker_sample, _ = self.marker_inlet.pull_sample(timeout=0.1)
                if marker_sample and 'focus:in;object: PlacementPos' in marker_sample[0]:
                    fix_delay = time.time() - pull_time

        if type == 'eeg':
            return all_eeg_data.T
        elif type == 'eye':
            return all_eye_data.T
        elif type == 'marker':
            return fix_delay

    def compute_features(self, data, modality):
        
        # mne_data = mne.io.RawArray(data.T, self.mne_raw_info) # ! if MNE object is needed for feature extraction

        window_size = 50 # ms

        # eeg processing and feature extraction
        if modality == 'eeg':

            srate = 250
            windowed_means = windowed_mean(data, srate, window_size)
            
            # select features of interest
            windowed_mean = windowed_mean[:,1:9].flatten()

            return windowed_mean

        # eye processing and feature extraction
        elif modality == 'eye':

            # channels for eye data
            gaze_direction_chans = np.arange(2,5)
            gaze_validity_chan = 10

            # Use the function in the compute_features method
            gaze_velocity = compute_gaze_velocity(data, gaze_direction_chans, gaze_validity_chan)

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

    def choose_nah_label(self):

        # this needs to pull data directly after the grab marker
        # needs to pull exactly 108 samples for eeg

        tic = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            eeg = executor.submit(self.get_data, 'eeg')
            eye = executor.submit(self.get_data, 'eye')
            fix_delay = self.get_data('marker')
        print(f"EEG data shape: {eeg.result().shape}, Eye data shape: {eye.result().shape}")
        toc = time.time()
        print(toc - tic)

        # compute features and predict    
        eeg_feat = classifier.compute_features(eeg, 'eeg') # this needs to pull data directly after the grab marker
        eye_feat = classifier.compute_features(eye, 'eye') # this needs to pull data directly after the grab marker

        feature_vector = np.concatenate((eeg_feat, eye_feat, [fix_delay]), axis=0).reshape(1, -1)

        # Predict the class
        prediction, probs_target_class, score = classifier.predict(feature_vector)

        discrete_prediction = self.discretize(prediction, score)

        return discrete_prediction
    
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

    while True:

        tic = time.time()
        choose_label = classifier.choose_nah_label()
        print(choose_label)
        print(f"Time to choose label: {time.time() - tic}")

    # TODO clean this up? but the -1 here had some logic, ask aleks
    # last_grab_number = -1

    # start_print_time = time.time()
    # last_print_time = start_print_time
    
    # while True:
    #     current_time = time.time()
    #     if current_time - last_print_time >= 1:
    #         elapsed_time = current_time - start_print_time
    #         minutes, seconds = divmod(int(elapsed_time), 60)
    #         formatted_time = f"{minutes:02}:{seconds:02}"
    #         print(f"{formatted_time} - Waiting for grab marker...")
    #         last_print_time = current_time

    #     marker = classifier.marker_inlet.pull_sample()[0]

    #     # what if there are two grab markers
    #     if marker and 'What:' in marker[0]:
    #         marker_data = marker[0].split(';')
    #         marker_dict = {item.split(':')[0]: item.split(':')[1] for item in marker_data}
    #         what = marker_dict.get('What')
            
    #         if what == 'grab':
    #             current_grab_number = int(marker_dict.get('Number', 0))

    #             if current_grab_number > last_grab_number:           
    #                 print(f"Grab {current_grab_number} detected: ", marker)

    #                 eeg, eye, fix_delay = classifier.get_data()
                    
    #                 eeg_feat = classifier.compute_features(eeg, 'eeg')
    #                 eye_feat = classifier.compute_features(eye, 'eye')

    #                 # for tests
    #                 # eeg = classifier.get_data()
    #                 # eye = classifier.get_data()            
    #                 # eye_feat = np.zeros(8)
    #                 # test_fix_delay = np.array([0.4])

    #                 # concatenate eeg, eye, fix_delay
    #                 feature_vector = np.concatenate((eeg_feat, eye_feat, [fix_delay]), axis=0).reshape(1, -1)

    #                 # pred
    #                 prediction, probs_target_class, score = classifier.predict(feature_vector)

    #                 print(f'Classifier stuff {prediction}, {probs_target_class}, {score}')
                    
    #                 # Map probs_target_class to an int value in the range 1 to 5
    #                 mapped_label = max(1, int(np.ceil(probs_target_class * 5)))

    #                 classifier.send_nah_label_to_ai(mapped_label)
                    
    #                 print("Label sent to AI: ", mapped_label)