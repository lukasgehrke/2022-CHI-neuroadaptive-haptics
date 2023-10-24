import pyxdf
from pylsl import StreamInfo, StreamOutlet
import time
import numpy as np

import matplotlib.pyplot as plt

fname = 'example_data/fastReach_s16_Baseline.xdf'
eeg_stream_name = 'BrainVision RDA'
eye_stream_name = ''
marker_stream_name = 'fastReach'

data, header = pyxdf.load_xdf(fname)

for stream in data:

    stream_name = stream['info']['name'][0]

    if stream_name == eeg_stream_name:
        eeg_time_stamps = stream['time_stamps']
        sample_interval = np.diff(eeg_time_stamps)
        eeg_data = stream['time_series']
        stream_info = StreamInfo(eeg_stream_name, stream['info']['type'][0],
                                 int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                                 stream['info']['channel_format'][0], stream['info']['uid'][0])
        eeg_outlet = StreamOutlet(stream_info)

    elif stream_name == eye_stream_name:
        eye_time_stamps = stream['time_stamps']
        eye_data = stream['time_series']
        stream_info = StreamInfo(eye_stream_name, stream['info']['type'][0],
                                 int(stream['info']['channel_count'][0]), int(stream['info']['nominal_srate'][0]), 
                                 stream['info']['channel_format'][0], stream['info']['uid'][0])
        eye_outlet = StreamOutlet(stream_info)
    
    elif stream_name == marker_stream_name:
        marker_time_stamps = stream['time_stamps']
        marker_data = stream['time_series']
        stream_info = StreamInfo(marker_stream_name, stream['info']['type'][0],
                                 int(stream['info']['channel_count'][0]), 0, 
                                 stream['info']['channel_format'][0], stream['info']['uid'][0])
        marker_outlet = StreamOutlet(stream_info)

print('Now transmitting data...')

eye_i = 0
eye_exists = 'eye_time_stamps' in locals() or 'eye_time_stamps' in globals()

marker_i = 0
i = 0

while True:
    
    eeg_outlet.push_sample(eeg_data[i,:])

    if eye_exists and eeg_time_stamps[i] > eye_time_stamps[eye_i]:
        eye_outlet.push_sample(eye_data[eye_i,:])
        eye_i += 1
    
    if not marker_i > len(marker_time_stamps) and eeg_time_stamps[i] > marker_time_stamps[marker_i]:
        print(marker_data[marker_i][0])
        marker_outlet.push_sample(marker_data[marker_i])
        marker_i += 1

    time.sleep(sample_interval[i])

    i += 1
    if i > len(sample_interval):
        i = 0
        eye_i = 0
        marker_i = 0


#         # list of strings, draw one vertical line for each marker
#         for timestamp, marker in zip(stream['time_stamps'], y):
#             plt.axvline(x=timestamp)
#             print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
#     elif isinstance(y, np.ndarray):
#         # numeric data, draw as lines
#         plt.plot(stream['time_stamps'], y)
#     else:
#         raise RuntimeError('Unknown stream format')

# plt.show()