import numpy as np
from pylsl import StreamInfo, StreamOutlet
import pyxdf, time, threading


class SimPhysDataStreamer(threading.Thread):

    def __init__(self):
        """Simulates the physiological data streams available in the network when running the neuro_haptics project.
        """

        threading.Thread.__init__(self)

        # Load EEG and Eye data
        # fname = 'example_data/fastReach_s16_Baseline.xdf'
        fname = r'C:\Users\Lukas\Desktop\nah\tst-eeg\data\0_source-data\sub-001\nah.xdf'
        self.data, header = pyxdf.load_xdf(fname)

        self.eeg_stream_name = 'BrainVision RDA'
        self.eye_stream_name = 'NAH_GazeBehavior'
        self.events_stream_name = 'NAH_Unity3DEvents'

        self.eeg_ix = 0
        self.eye_ix = 0
        self.events_ix = 0

        for stream in self.data:

            stream_name = stream['info']['name'][0]

            if stream_name == self.eeg_stream_name:
                self.eeg_time_stamps = stream['time_stamps']
                self.sample_interval = np.diff(self.eeg_time_stamps)
                self.eeg_data = stream['time_series']
                stream_info = StreamInfo(self.eeg_stream_name, stream['info']['type'][0],
                                        int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                                        stream['info']['channel_format'][0], stream['info']['uid'][0])
                # stream_info = StreamInfo('SimPhysDataStream_Lukas', stream['info']['type'][0],
                #                         int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                #                         stream['info']['channel_format'][0], stream['info']['uid'][0])                
                self.eeg_outlet = StreamOutlet(stream_info)

            elif stream_name == self.eye_stream_name:
                self.eye_time_stamps = stream['time_stamps']
                self.eye_data = stream['time_series']
                stream_info = StreamInfo(self.eye_stream_name, stream['info']['type'][0],
                                        int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                                        stream['info']['channel_format'][0], stream['info']['uid'][0])
                self.eye_outlet = StreamOutlet(stream_info)

            elif stream_name == self.events_stream_name:
                self.events_time_stamps = stream['time_stamps']
                self.events_data = stream['time_series']
                stream_info = StreamInfo(self.events_stream_name, stream['info']['type'][0],
                                        int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                                        stream['info']['channel_format'][0], stream['info']['uid'][0])
                self.events_outlet = StreamOutlet(stream_info)                

    def send_eeg(self):
        self.eeg_outlet.push_sample(self.eeg_data[self.eeg_ix,:])
        self.eeg_ix += 1

    def send_eye(self):
        # different sample rates
        if self.eeg_time_stamps[self.eeg_ix] > self.eye_time_stamps[self.eye_ix]:
            self.eye_outlet.push_sample(self.eye_data[self.eye_ix,:])
            self.eye_ix += 1
    
    def send_events(self):
        if self.eeg_time_stamps[self.eeg_ix] > self.events_time_stamps[self.events_ix]:
            print("Pushed event: ", self.events_data[self.events_ix])
            self.events_outlet.push_sample(self.events_data[self.events_ix])
            self.events_ix += 1

    def run(self):

        last_formatted_timestamp = None
        
        while True:
            # Print currently elapsed min:sec
            formatted_timestamp = time.strftime("%M:%S", time.gmtime(self.eeg_time_stamps[self.eeg_ix]))
            if formatted_timestamp != last_formatted_timestamp:
                print(formatted_timestamp)
                last_formatted_timestamp = formatted_timestamp

            # push to self.labels
            self.send_eeg()

            if self.eye_stream_name != '':
                self.send_eye()

            self.send_events()

            # wait for 1 second
            time.sleep(self.sample_interval[self.eeg_ix])

if __name__ == "__main__":

    sim = SimPhysDataStreamer()
    sim.start()