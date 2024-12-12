from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
import pickle, time, os, json
import numpy as np
from bci_funcs import windowed_mean, calculate_velocity, bandpass_filter_fft, gaze_remove_invalid_samples
# import concurrent.futures
from bsl import StreamReceiver

class NahClassifier:
    def __init__(self, path, pID):

        self.pID = pID
        self.path = path

        model_path = path+os.sep+self.pID+os.sep
        self.model = pickle.load(open(model_path+'model.sav', 'rb'))
        self.scaler = pickle.load(open(model_path+'scaler.sav', 'rb'))

        with open(model_path+'bci_params.json', 'r') as f:
            bci_params = json.load(f)
        self.mean_fix_delay = bci_params['mean_fix_delay']
        
        with open(model_path+'top_channels.json', 'r') as f:
            self.top_channels = json.load(f)
        with open(model_path+'lda_scores.json', 'r') as f:
            self.lda_scores = json.load(f)
            self.lda_scores = [item for sublist in self.lda_scores for item in sublist]
        self.min_boundary = np.percentile(self.lda_scores, 5)
        self.max_boundary = np.percentile(self.lda_scores, 95)

        # resolve streams
        streams = None
        while streams is None:

            # print("Looking for EEG stream...")	
            # streams = resolve_byprop('name', 'BrainVision RDA')
            
            # # Don't think this is necessary, since resolve_prop will wait until it finds the stream
            # if not streams:
            #     print("No EEG stream found, retrying...")
            #     time.sleep(1)
            
            # print("EEG stream found!")

            # # init EEG stream inlet
            # self.eeg_inlet = StreamInlet(streams[0])

            # print("Looking for Hand motion stream...")
            # streams = resolve_byprop('name', 'NAH_rb_handRight')
            
            # if not streams:
            #     print("No Hand motion stream found, retrying...")
            #     time.sleep(1)

            # print("Hand motion stream found!")

            # # init Hand motion stream inlet
            # self.motion_inlet = StreamInlet(streams[0])

            
            # # print("Looking for EYE stream...")
            # # streams = resolve_byprop('name', 'NAH_GazeBehavior')
            
            # # if not streams:
            # #     print("No EEG stream found, retrying...")
            # #     time.sleep(1)

            # # print("EYE stream found!")

            # # # init EYE stream inlet
            # # self.eye_inlet = StreamInlet(streams[0])

                        
            print("Looking for Marker stream...")
            streams = resolve_byprop('name', 'NAH_Unity3DEvents')
            if not streams:
                print("No Marker stream found, retrying...")
                time.sleep(1)          
            print("Marker stream found!")
            # init marker stream inlet
            self.marker_inlet = StreamInlet(streams[0])

            # print("Looking for Fixations stream...")
            # streams = resolve_byprop('name', 'NAH_FocusedObjectEvents')
            # if not streams:
            #     print("No Marker stream found, retrying...")
            #     time.sleep(1)

            
            # print("Fixations stream found!")
            # # init marker stream inlet
            # self.fixations_inlet = StreamInlet(streams[0])

            # Connects to EEG stream
            self.sr = StreamReceiver(bufsize=1, winsize=1, stream_name='BrainVision RDA')
            time.sleep(1)
            

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
    
    def normalize_to_boundaries(self, score, update_boundaries=False):
        """
        Normalize the score to the range [0, 1] using min-max normalization
        based on boundaries specified in the JSON file.

        Parameters:
        score (float): The score to be normalized

        Returns:
        float: The normalized score in the range [0, 1]
        """

        if update_boundaries:
            # add score to the list of lda scores
            self.lda_scores.append(score)
            
            self.min_boundary = np.percentile(self.lda_scores, 5)
            self.max_boundary = np.percentile(self.lda_scores, 95)

            # save lda_scores with self.pID in filename in subfolder lda_scores
            with open(self.path + os.sep + self.pID + os.sep + 'lda_scores.json', 'w') as f:
                json.dump(self.lda_scores , f)
            print(f"Updated boundaries: {self.min_boundary:.2f}, {self.max_boundary:.2f}")
        
        # Perform min-max normalization
        normalized_score = (score - self.min_boundary) / (self.max_boundary - self.min_boundary)

        # Ensure the normalized score is within the range [0, 1]
        normalized_score = np.clip(normalized_score, 0, 1)

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

    def get_data_bsl(self):

        time.sleep(1) # wait for 1s to fill up the buffer

        # Update each stream buffer with new data
        self.sr.acquire()
        print("Buffer acquired at time: ", time.time())
        
        # Retrieve buffer/window for the stream named 'StreamPlayer'
        data, _ = self.sr.get_window('BrainVision RDA')
        print("Data acquired at time: ", time.time())
        data = np.delete(data, 0, 1) # channels info

        return data

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

        if data_type in ['eeg', 'eye', 'motion']:

            _, ts1 = inlet.pull_sample(timeout=0.0)
            ts_tmp = ts1

            while ts_tmp - ts1 <= 1.0:  # 1 second window to grab data
                tmp_data, ts_tmp = inlet.pull_sample()
                tmp_data = np.array(tmp_data).reshape(1, -1)
                all_data = np.vstack([all_data, tmp_data])
            
            return all_data.T    
        
        elif data_type == 'marker':
            
            #_, ts1 = inlet.pull_sample()
            # ts_tmp = ts1
            # while ts_tmp - ts1 <= 1.0:  # 1 second window to grab data
            
            ts_tmp = grab_ts
            while ts_tmp - grab_ts <= 1.0:  # 1 second window to grab data

                marker_sample, ts_pull = inlet.pull_sample(timeout=0.0)
                if marker_sample is not None:
                    ts_tmp = ts_pull                                
                    print(marker_sample)

                if marker_sample and 'focus:in;object: PlacementPos' in marker_sample[0]:
                    fix_delay = ts_tmp - grab_ts
                    print(f"Fixation delay: {fix_delay:.2f}")
                    break

            if fix_delay == 0:
                fix_delay = self.mean_fix_delay

            return fix_delay

    def compute_features(self, data, modality):

        # # Consider only data from the first 450 ms
        # srate = data.shape[0] / 1.0  # Assumes data is collected for 1 second
        # num_samples_600ms = int(0.6 * srate)

        data = data[:144, :].T

        # if num_samples_600ms > 144:
        #     data = data[:144, :].T
        # else:
        #     data = data[:num_samples_600ms, :].T

        # eeg processing and feature extraction
        if modality == 'eeg':

            erp_selected = data[self.top_channels, :]
            erp_selected = np.expand_dims(erp_selected, axis=2)
            erp_selected = bandpass_filter_fft(erp_selected, 0.1, 15, 250)

            # if num_samples_600ms > 144:
            #     data = np.array_split(erp_selected, 12, axis=1)
            #     erp_selected = [np.mean(part, axis=1) for part in data]
            #     erp_selected = np.array(erp_selected).T
            # else:
            erp_selected = erp_selected.reshape(erp_selected.shape[0], 12, 12, erp_selected.shape[2])
            erp_selected = np.squeeze(np.mean(erp_selected, axis=2))
            
            baseline = erp_selected[:,0]
            erp_corrected = erp_selected - baseline[:, np.newaxis]
            erp_corrected = erp_corrected[:,2:].flatten()

            return erp_corrected
        
        elif modality == 'motion':

            cart_motion_chans = np.arange(0,3)
            velocity = calculate_velocity(data, cart_motion_chans, srate)
            velocity = np.expand_dims(velocity, axis=[0,2])
            velocity = np.squeeze(bandpass_filter_fft(velocity, 0.1, 15, srate))

            return np.max(velocity)

            # data = np.array_split(velocity, 12)
            # windowed_means = [np.mean(part) for part in data]
            # windowed_means = np.array(windowed_means)
            # windowed_means = windowed_means[0:-1]
            # return windowed_means

        # eye processing and feature extraction
        elif modality == 'eye':

            srate = data.shape[1]

            # channels for eye data
            gaze_direction_chans = np.arange(2,5)
            gaze_validity_chan = 10
            
            # TODO test when needed, currently not included in the model
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

        # old version running with futures threads -> now using BSL LSL app
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(self.get_data, 'eeg', grab_ts): 'eeg',
        #         executor.submit(self.get_data, 'motion', grab_ts): 'motion',
        #         # executor.submit(self.get_data, 'marker', grab_ts): 'marker'
        #     }

        #     results = {}
        #     for future in concurrent.futures.as_completed(futures):
        #         data_type = futures[future]
        #         try:
        #             result = future.result()
        #             results[data_type] = result
        #         except Exception as exc:
        #             print(f"{data_type} generated an exception: {exc}")
        
        # # Compute features and predict
        # eeg_feat = self.compute_features(results['eeg'], 'eeg')
        # motion_feat = self.compute_features(results['motion'], 'motion')
        # fix_delay = results['marker']        
        # feature_vector = np.concatenate((eeg_feat, [motion_feat], [fix_delay]), axis=0).reshape(1, -1)

        tic_data_pull = time.time()
        eeg_data = self.get_data_bsl()
        print(f"Time to get data: {time.time() - tic_data_pull}")
        eeg_feat = self.compute_features(eeg_data, 'eeg')

        feature_vector = eeg_feat.reshape(1, -1)
        _, score = self.predict(feature_vector)
        print(f"Score: {score:.2f}")

        # Normalize the score to the range [0, 1]
        score = self.normalize_to_boundaries(score, update_boundaries=False)
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
   
    pID = 'sub-' + "9"
    path = r'P:\Lukas_Gehrke\NAH\data\5_single-subject-EEG-analysis'

    classifier = NahClassifier(path, pID)

    time.sleep(2) # wait for the streams buffers to fill up
    last_grab_number = -1
    
    while True:

        marker, grab_ts = classifier.marker_inlet.pull_sample()

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