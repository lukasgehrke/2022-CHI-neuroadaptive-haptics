import numpy as np
from scipy import stats


def bandpass_filter_fft(data, lowcut, highcut, fs):
    """
    Apply a bandpass filter to 3D ERP data using FFT.

    Parameters:
    data (numpy.ndarray): 3D array of ERP data with shape (channels, samples, trials)
    lowcut (float): Low cutoff frequency in Hz
    highcut (float): High cutoff frequency in Hz
    fs (float): Sampling frequency in Hz

    Returns:
    numpy.ndarray: Filtered ERP data
    """
    # Perform FFT on the data
    fft_data = np.fft.fft(data, axis=1)
    freqs = np.fft.fftfreq(data.shape[1], d=1/fs)

    # Create a frequency mask
    mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    mask = mask.reshape((1, -1, 1))  # Reshape mask to match the dimensions of fft_data


    # Apply the mask to the FFT data
    fft_data *= mask

    # Perform the inverse FFT to get the filtered data
    filtered_data = np.fft.ifft(fft_data, axis=1)

    # Return the real part of the filtered data
    return np.real(filtered_data)

def windowed_mean(data, srate, window_size):
    
    n_windows = int(np.floor(1000 / window_size)) # int(np.floor(srate / window_size))
    window_size = int(np.floor(window_size * 0.001 * srate)) # ms to samples

    if len(data.shape) < 3:
        baseline = data[:,0:window_size].mean(axis=1)
        data = data[:,0:n_windows*window_size]
        data = data - baseline[:,None]
        reshaped = data.reshape(data.shape[0], n_windows, window_size)
        windowed_mean = reshaped.mean(axis=-1)
    else:
        baseline = data[:,0:window_size,:].mean(axis=1)
        data = data[:,0:n_windows*window_size,:]
        data = data - baseline[:,np.newaxis,:]
        reshaped = data.reshape(data.shape[0], n_windows, window_size, data.shape[2])
        windowed_mean = reshaped.mean(axis=2)

    return windowed_mean

def gaze_remove_invalid_samples(data, gaze_validity):
    
    invalid_samples = gaze_validity == 1
    data[invalid_samples] = np.nan
    
    return data

def calculate_velocity(data, cart_motion_chans=np.arange(0,3), fs=250):
    """
    Calculate the velocity in meters per second (m/s) from the motion data.

    Parameters:
    data (numpy.ndarray): 2D array of motion data with shape (channels, samples)
    cart_motion_chans (list): List of indices for the Cartesian motion channels (x, y, z)
    fs (float): Sampling frequency in Hz

    Returns:
    numpy.ndarray: Velocity in meters per second (m/s)
    """
    # Initialize the velocity array
    velocity = np.zeros(data.shape[1] - 1)

    # Calculate the differences in the Cartesian coordinates
    tmp = np.diff(data[cart_motion_chans, :], axis=1)

    # Calculate the Euclidean distance (displacement) between consecutive samples
    displacement = np.sqrt(np.sum(tmp**2, axis=0))

    # Calculate the time interval between samples
    dt = 1 / fs

    # Calculate the velocity in meters per second (m/s)
    velocity = displacement / dt

    # Repeat the last value to keep the same length
    velocity = np.append(velocity, velocity[-1])

    return velocity

# # TODO fix this with real-time data
# def compute_velocity(data, srate=250, cart_motion_chans = np.arange(0,3)):

#     # window_size = int(np.floor(window_size * 0.001 * srate)) # ms to samples
#     # n_windows = int(np.floor(srate / window_size))

#     # data = data[0:n_windows*window_size,:]

#     velocity = np.zeros(data.shape[1] - 1)

#     tmp = np.diff(data[cart_motion_chans, :], axis=1)
#     velocity = np.sqrt(np.sum(tmp**2, axis=0))

#     # repeat last value to keep the same length
#     velocity = np.append(velocity, velocity[-1])
    
#     return velocity

# def windowed_mean(data, n_windows = 10):
#     """Computes windowed mean of data

#     Args:
#         data (_type_): 2D array, usually chans x time
#         windows (int, optional): Number of windows data is split into . Defaults to 10.

#     Returns:
#         _type_: Windowed mean of data, of shape chans x n_windows
#     """
#     stepsize = data.shape[1] // n_windows
#     win_data = np.reshape(data, (data.shape[0], n_windows, stepsize))
#     return np.mean(win_data, axis = 2)

def select_mean(data, n_windows, window):
    """_summary_

    Args:
        data (_type_): _description_
        n_windows (_type_): _description_
        window (_type_): _description_

    Returns:
        _type_: _description_
    """

    win_means = windowed_mean(data, n_windows)
    return win_means[:,window-1]

def slope(data, function):

    slopes = np.array([])

    for i in range(data.shape[0]):

        if function == 'linear':
            slopes = np.append(slopes, stats.linregress(data[i,:], np.arange(0, data.shape[1]))[0])

        elif function == 'exp':
            log_clean_data = np.ma.masked_invalid(np.log(data[i,:] - np.min(data[i,:]))).compressed() # log of data, excluding -inf
            slopes = np.append(slopes, stats.linregress(log_clean_data, np.arange(0, log_clean_data.shape[0]))[0])

    return slopes

    


def base_correct(data, baseline_end_ix):
    """Subtracts baseline from data

    Args:
        data (_type_): 2D array, usually chans x time

    Returns:
        _type_: Baseline corrected 2D array, usually chans x time
    """
    base = np.mean(data[:,:int(baseline_end_ix)], axis = 1)
    data_base_correct = data - base[:,None]
    return data_base_correct

def drop_baseline(data, baseline_end_ix):
    """Drops baseline from data"""
    data = data[:,baseline_end_ix:]
    return data

def reshape_trial(data):
    data_reshaped_trial = np.reshape(data, (data.shape[2], data.shape[0] * data.shape[1]), order = 'F')
    # data_reshaped_trial = np.reshape(data, (data.shape[0] * data.shape[1]), order = 'F')
    return data_reshaped_trial