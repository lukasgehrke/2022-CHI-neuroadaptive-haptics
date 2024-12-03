from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import pickle, time, os, json
import numpy as np
from bci_funcs import windowed_mean, calculate_velocity, bandpass_filter_fft, gaze_remove_invalid_samples
import concurrent.futures
import argparse

class NahClassifier:
    def __init__(self, model_path):
        
        self.model = pickle.load(open(model_path+'model.sav', 'rb'))
        self.scaler = pickle.load(open(model_path+'scaler.sav', 'rb'))

        with open(model_path+'bci_params.json', 'r') as f:
            bci_params = json.load(f)
        self.mean_fix_delay = bci_params['mean_fix_delay']
        
        with open(model_path+'top_channels.json', 'r') as f:
            self.top_channels = json.load(f)
        with open(model_path+'boundaries.json', 'r') as f:
            self.boundaries = json.load(f)
    
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

            print("Looking for Hand motion stream...")
            streams = resolve_byprop('name', 'NAH_rb_handRight')
            
            if not streams:
                print("No Hand motion stream found, retrying...")
                time.sleep(1)

            print("Hand motion stream found!")

            # init Hand motion stream inlet
            self.motion_inlet = StreamInlet(streams[0])

            
            # print("Looking for EYE stream...")
            # streams = resolve_byprop('name', 'NAH_GazeBehavior')
            
            # if not streams:
            #     print("No EEG stream found, retrying...")
            #     time.sleep(1)

            # print("EYE stream found!")

            # # init EYE stream inlet
            # self.eye_inlet = StreamInlet(streams[0])

                        
            print("Looking for Marker stream...")
            streams = resolve_byprop('name', 'NAH_Unity3DEvents')
            if not streams:
                print("No Marker stream found, retrying...")
                time.sleep(1)

            
            print("Marker stream found!")
            # init marker stream inlet
            self.marker_inlet = StreamInlet(streams[0])

            print("Looking for Fixations stream...")
            streams = resolve_byprop('name', 'NAH_FocusedObjectEvents')
            if not streams:
                print("No Marker stream found, retrying...")
                time.sleep(1)

            
            print("Fixations stream found!")
            # init marker stream inlet
            self.fixations_inlet = StreamInlet(streams[0])

        # set up outlet for sending predictions
        self.labels = StreamOutlet(StreamInfo('labels', 'Markers', 1, 0, 'string', 'myuid34234'))
    
    def predict(self, features):
        """
        Predict the class of the given features using the classifier model.

        Parameters:
        features (numpy.ndarray): The features to be used for prediction

        Returns:
        int: The predicted class
        float: The probability of the target class
        float: The score of the prediction
        """

        features = self.scaler.transform(features)
        score = self.model.transform(features)[0][0]
        
        prediction = int(self.model.predict(features)[0])  # predicted class

        # probs = self.model.predict_proba(features)  # probability for class prediction
        # probs_target_class = probs[0][int(self.target_class)]

        return prediction, score #probs_target_class
    
    def normalize_to_boundaries(self, score):
        """
        Normalize the score to the range [0, 1] using min-max normalization
        based on boundaries specified in the JSON file.

        Parameters:
        score (float): The score to be normalized

        Returns:
        float: The normalized score in the range [0, 1]
        """
        
        # Get the min and max boundaries from the JSON
        min_boundary = self.boundaries[0]
        max_boundary = self.boundaries[-1]

        # Perform min-max normalization
        normalized_score = (score - min_boundary) / (max_boundary - min_boundary)

        # Ensure the normalized score is within the range [0, 1]
        if normalized_score < 0:
            return 0
        elif normalized_score > 1:
            return 1
        else:
            return normalized_score
        
    def discretize(self, score):

        # find the boundaries for the classifier
        for i, boundary in enumerate(self.boundaries):
            if score < boundary:
                bin = i
                break
            else:
                bin = len(self.boundaries)

        bin += 1        
        return bin
        
    def get_data(self, data_type, grab_ts):
        """
        Pull data from the specified inlet for a duration of 1 second.

        Parameters:
        data_type (str): The type of data to pull ('eeg', 'eye', 'motion', 'marker').

        Returns:
        numpy.ndarray or float: The pulled data as a numpy array (for 'eeg', 'eye', 'motion') 
                                or the fix delay as a float (for 'marker').
        """

        if data_type == 'eeg':
            all_data = np.empty((0, 64))
            inlet = self.eeg_inlet
        elif data_type == 'eye':
            all_data = np.empty((0, 11))
            inlet = self.eye_inlet
        elif data_type == 'motion':
            all_data = np.empty((0, 7))
            inlet = self.motion_inlet
        elif data_type == 'marker':
            fix_delay = 0
            inlet = self.fixations_inlet
        else:
            raise ValueError("Invalid data type. Choose from 'eeg', 'eye', 'motion', 'marker'.")
        
        ts_tmp = grab_ts
        while ts_tmp - grab_ts <= 1.0:  # 1 second window to grab data

            # Pull data
            if data_type in ['eeg', 'eye', 'motion']:
                tmp_data, ts_tmp = inlet.pull_sample()
                tmp_data = np.array(tmp_data).reshape(1, -1)
                all_data = np.vstack([all_data, tmp_data])
            
            elif data_type == 'marker':
                marker_sample, ts_tmp = inlet.pull_sample()
                if marker_sample and 'focus:in;object: PlacementPos' in marker_sample[0]:
                    fix_delay = ts_tmp - grab_ts # check value
                    break

        if data_type in ['eeg', 'eye', 'motion']:
            return all_data.T
        elif data_type == 'marker':
            if fix_delay == 0:
                fix_delay = self.mean_fix_delay
            return fix_delay

    def compute_features(self, data, modality):

        # Consider only data from the first 450 ms
        srate = data.shape[1] / 1.0  # Assuming data is collected for 1 second
        num_samples_450ms = int(0.45 * srate)
        data = data[:, :num_samples_450ms]

        # eeg processing and feature extraction
        if modality == 'eeg':

            erp_selected = data[self.top_channels, :]# TODO channel indices correct from 0 index?
            erp_selected = np.expand_dims(erp_selected, axis=2)
            erp_selected = np.squeeze(bandpass_filter_fft(erp_selected, 0.1, 15, 250))

            data = np.array_split(erp_selected, 9, axis=1)
            windowed_means = [np.mean(part, axis=1) for part in data]
            windowed_means = np.array(windowed_means).T
            baseline = windowed_means[:,0]

            windowed_means = windowed_means - baseline[:,np.newaxis] # correct for baseline
            windowed_means = windowed_means[:,1:9].flatten()

            return windowed_means
        
        elif modality == 'motion':

            cart_motion_chans = np.arange(0,3)
            velocity = calculate_velocity(data, cart_motion_chans, srate)
            velocity = np.expand_dims(velocity, axis=[0,2])
            velocity = np.squeeze(bandpass_filter_fft(velocity, 0.1, 15, srate))

            data = np.array_split(velocity, 9)
            windowed_means = [np.mean(part) for part in data]
            windowed_means = np.array(windowed_means)
            windowed_means = windowed_means[1:9]

            return windowed_means

        # eye processing and feature extraction
        elif modality == 'eye':

            srate = data.shape[1]

            # channels for eye data
            gaze_direction_chans = np.arange(2,5)
            gaze_validity_chan = 10
            
            # TODO test when needed
            velocity = calculate_velocity(data, gaze_direction_chans, srate)
            velocity = gaze_remove_invalid_samples(velocity, data[gaze_validity_chan,:])

            # these are not used 
            gaze_velocity = np.expand_dims(gaze_velocity, axis=0)
            windowed_means = windowed_mean(gaze_velocity, srate, window_size)
            windowed_means = windowed_means[:,1:9].flatten()

            return windowed_means

    def choose_nah_label(self, grab_ts):
        """
        Pull data directly after the grab marker, compute features, and predict the NAH label.

        Returns:
        float: The predicted score.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_data, 'eeg', grab_ts): 'eeg',
                executor.submit(self.get_data, 'motion', grab_ts): 'motion',
                executor.submit(self.get_data, 'marker', grab_ts): 'marker'
            }

            results = {}
            for future in concurrent.futures.as_completed(futures):
                data_type = futures[future]
                try:
                    result = future.result()
                    results[data_type] = result
                except Exception as exc:
                    print(f"{data_type} generated an exception: {exc}")

        # Compute features and predictx
        eeg_feat = self.compute_features(results['eeg'], 'eeg')
        motion_feat = self.compute_features(results['motion'], 'motion')
        fix_delay = results['marker']
        
        feature_vector = np.concatenate((eeg_feat, motion_feat, [fix_delay]), axis=0).reshape(1, -1)
        _, score = self.predict(feature_vector)
        print(f"Score: {score:.2f}")

        # Normalize the score to the range [0, 1]
        score = self.normalize_to_boundaries(score)
        print(f"Normalized score: {score:.2f}")

        return score
    
    def send_nah_label_to_ai(self, label):
        """
        Send the NAH label to the AI system.

        Parameters:
        label (int or str): The label to be sent. It will be converted to a string before sending.

        Returns:
        None
        """
        label = str(label)
        self.labels.push_sample([label])

# main
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run NahClassifier for a specific participant.")
    parser.add_argument('--id', type=int, required=True, help='Participant ID')
    args = parser.parse_args()
    pID = 'sub-' + "%01d" % (args.id)

    # path = '/Volumes/Lukas_Gehrke/NAH/data/5_single-subject-EEG-analysis'
    # path = '/Users/lukasgehrke/data/NAH/data/5_single-subject-EEG-analysis/'
    path = r'P:\Lukas_Gehrke\NAH\data\5_single-subject-EEG-analysis'

    model_path = path+os.sep+pID+os.sep
    classifier = NahClassifier(model_path)

    time.sleep(4) # wait for the streams buffers to fill up
    last_grab_number = -1
    
    while True:

        marker, grab_ts = classifier.marker_inlet.pull_sample()
        print(marker)

        # what if there are two grab markers
        if marker and 'What:' in marker[0]:
            marker_data = marker[0].split(';')
            marker_dict = {item.split(':')[0]: item.split(':')[1] for item in marker_data}
            what = marker_dict.get('What')
            
            if what == 'grab':
                current_grab_number = int(marker_dict.get('Number', 0))

                if current_grab_number > last_grab_number:           
                    print(f"Grab {current_grab_number} detected: ", marker)

                    tic = time.time()
                    label = classifier.choose_nah_label(grab_ts)
                    print(f"Time to choose label: {time.time() - tic}")

                    classifier.send_nah_label_to_ai(label)
                    print("Label sent to AI: ", label)