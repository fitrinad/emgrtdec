import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
import os
from scipy import interpolate
from scipy.io import loadmat
from scipy.signal import butter, lfilter, iirnotch
from emgdecompy.preprocessing import flatten_signal


def import_emg_data(empty_dict={}, name="P00", n_training=1, n_objects=3, fs=2000,
                    electrode_placements=["ext", "int"],
                    data_types=["10_train", "30_train", "10_mov", "30_mov"],
                    gestures=["rest_calibration", "close", "pinch", "tripod"]):
    """
    # Hanning window
    windowSize = fs * filt_size
    window = np.hanning(windowSize)
    window = window / window.sum()
    """
    for electrode_placement in electrode_placements:
        for data_type in data_types:
            folder_path = "../data/" + name + "/" + name + "_" + electrode_placement + "_" + data_type
            key_name = electrode_placement + "_" + data_type
            empty_dict[key_name] = {}
            for gesture in gestures:
                # Rest calibration from Training data 10% MVC
                if gesture == "rest_calibration":
                    if data_type == "10_train":
                        # Importing rest calibration data
                        file_name = folder_path + "/" + gesture + ".csv"
                        file_name_force = folder_path + "/" + gesture + "_force.csv"
                        emg_signal = np.array( pd.read_csv(file_name).T )
                        empty_dict[key_name][gesture] = emg_signal
                        force_signal = np.array( pd.read_csv(file_name_force).T )
                        force_signal = interpolate_1d(force_signal, emg_signal[0]).flatten()
                        # force_signal = np.convolve(window, force_signal, mode='same')
                        empty_dict[key_name][gesture+"_force"] = force_signal
                else:
                    if (data_type == "10_train") or (data_type == "30_train"):
                        # Importing training data
                        for i in range(1, n_training+1):
                            training_gesture = gesture + str(i)
                            file_name = folder_path + "/" + training_gesture + ".csv"
                            file_name_force = folder_path + "/" + training_gesture + "_force.csv"
                            emg_signal = np.array( pd.read_csv(file_name).T )
                            empty_dict[key_name][training_gesture] = emg_signal
                            force_signal = np.array( pd.read_csv(file_name_force).T )
                            force_signal = interpolate_1d(force_signal, emg_signal[0]).flatten()
                            # force_signal = np.convolve(window, force_signal, mode='same')
                            empty_dict[key_name][training_gesture+"_force"] = force_signal
                        if data_type == "10_train":
                            # Importing mvc data
                            mvc_gesture = "mvc_" + gesture
                            file_name = folder_path + "/" + mvc_gesture + ".csv"
                            file_name_force = folder_path + "/" + mvc_gesture + "_force.csv"
                            emg_signal = np.array( pd.read_csv(file_name).T )
                            empty_dict[key_name][mvc_gesture] = emg_signal
                            force_signal = np.array( pd.read_csv(file_name_force).T )
                            force_signal = interpolate_1d(force_signal, emg_signal[0]).flatten()
                            # force_signal = np.convolve(window, force_signal, mode='same')
                            empty_dict[key_name][mvc_gesture+"_force"] = force_signal
                    elif (data_type == "10_mov") or (data_type == "30_mov"):
                        # Importing movement data
                        for i in range(1, n_objects+1):
                            movement_gesture = gesture + str(i)
                            file_name = folder_path + "/" + movement_gesture + ".csv"
                            file_name_force = folder_path + "/" + movement_gesture + "_force.csv"
                            emg_signal = np.array( pd.read_csv(file_name).T )
                            empty_dict[key_name][movement_gesture] = emg_signal
                            force_signal = np.array( pd.read_csv(file_name_force).T )
                            force_signal = interpolate_1d(force_signal, emg_signal[0]).flatten()
                            # force_signal = np.convolve(window, force_signal, mode='same')
                            empty_dict[key_name][movement_gesture+"_force"] = force_signal
    empty_dict["fs"] = fs
    
    return empty_dict


def import_emg_data_otb(empty_dict={}, name="P01", 
                        n_training=1, n_objects=4, fs=2000, cont_length=30.0,
                        electrode_placements=["ext", "int"],
                        mvc_levels=["10", "30"],
                        gestures=["close", "pinch", "tripod"]):
    for electrode_placement in electrode_placements:
        if electrode_placement == "ext":
            muscle_names = ["flexor", "extensor"]
        elif electrode_placement == "int":
            muscle_names = ["fdi", "second_di"]
        for mvc_level in mvc_levels:
            key_name_train = electrode_placement + "_" + mvc_level + "_train"
            key_name_mov = electrode_placement + "_" + mvc_level + "_mov"
            empty_dict[key_name_train] = {}
            empty_dict[key_name_mov] = {}
            for gesture in gestures:
                folder_name = name + "_" + electrode_placement + "_" + mvc_level + "_" + gesture
                folder_path = "../../data/" + name + "/" + folder_name
                # Importing force data
                file_name_force = folder_path + "/" + folder_name + "_force.csv"
                force_data = np.array( pd.read_csv(file_name_force, delimiter=";").T)
                force_data = force_data[1:, :].flatten()

                # Importing voltage measurement data
                file_name_volt = folder_path + "/" + folder_name + "_volt.csv"
                volt_data = np.array( pd.read_csv(file_name_volt, delimiter=";").T)
                volt_data = volt_data[1:, :].flatten()
                
                # Importing EMG data
                file_name_emg_0 = folder_path + "/" + folder_name + "_" + muscle_names[0] + ".csv"
                emg_data_0 = np.array( pd.read_csv(file_name_emg_0, delimiter=";").T )
                emg_data_0 = emg_data_0[1:,:]
                nx_0 = emg_data_0.shape[1]
                file_name_emg_1 = folder_path + "/" + folder_name + "_" + muscle_names[1] + ".csv"
                emg_data_1 = np.array( pd.read_csv(file_name_emg_1, delimiter=";").T )
                emg_data_1 = emg_data_1[1:,:]
                nx_1 = emg_data_1.shape[1]
                if nx_0 > nx_1:
                    emg_data_0 = emg_data_0[:, :nx_1]
                elif nx_1 > nx_0:
                    emg_data_1 = emg_data_1[:, :nx_0]
                emg_data = np.vstack((emg_data_0, emg_data_1))

                # Separating training data and movement data
                ## Training data
                for i in range(1, n_training+1):
                    # Cropping training data from recorded EMG
                    train_signal = crop_data(data=emg_data, 
                                                start = (i-1)*2*cont_length,
                                                end = i*2*cont_length,
                                                fs = fs)
                    train_gesture = gesture + str(i)
                    empty_dict[key_name_train][train_gesture] = train_signal
                    # Cropping training data from force signal
                    train_force = crop_data(data=force_data, 
                                                start = (i-1)*2*cont_length,
                                                end = i*2*cont_length,
                                                fs = fs)
                    empty_dict[key_name_train][train_gesture+"_force"] = train_force
                    # Cropping training data from volt signal 
                    train_volt = crop_data(data=volt_data, 
                                            start = (i-1)*2*cont_length,
                                            end = i*2*cont_length,
                                            fs = fs)
                    empty_dict[key_name_train][train_gesture+"_volt"] = train_volt

                ## Movement data while holding objects
                for i in range(1, n_objects+1):
                    # Cropping movement data from recorded EMG
                    mov_signal = crop_data(data=emg_data,
                                            start = (n_training * 2*cont_length) + ((i-1)*cont_length),
                                            end = (n_training * 2*cont_length) + (i*cont_length),
                                            fs = fs)
                    mov_gesture = gesture + str(i)
                    empty_dict[key_name_mov][mov_gesture] = mov_signal
                    # Cropping movement data from force signal
                    mov_force = crop_data(data=force_data,
                                            start = (n_training * 2*cont_length) + ((i-1)*cont_length),
                                            end = (n_training * 2*cont_length) + (i*cont_length),
                                            fs = fs)
                    empty_dict[key_name_mov][mov_gesture+"_force"] = mov_force
                    # Cropping training data from volt signal 
                    mov_volt = crop_data(data=volt_data,
                                            start = (n_training * 2*cont_length) + ((i-1)*cont_length),
                                            end = (n_training * 2*cont_length) + (i*cont_length),
                                            fs = fs)
                    empty_dict[key_name_mov][mov_gesture+"_force"] = mov_volt
    empty_dict["fs"] = fs
    
    return empty_dict


def fftPlot(sig, dt=None, plot=True):
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequency axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return sigFFTPos, freqAxisPos


def interpolate_1d(data, data_ref):
    """
    Interpolates a 1D array, data, to the size of 1D array data_ref.
    
    Args:
    	data	        : numpy.ndarray
            1D array containing data to be interpolated
        data_ref        : numpy.ndarray
            1D array being the size reference of the resulting interpolation
        
    Returns:
        data_result     : numpy.ndarray
            1D array containing data after interpolated to fit the size of data_ref
    """
    data_tmp = interpolate.interp1d(np.arange(data.size), data)
    data_interpolated = data_tmp(np.linspace(0, data.size-1, data_ref.size))

    return data_interpolated


def calc_snr(data, noise=None, noise_start=0.0, noise_end=3.0, fs=2048):
    """
    Calculates the signal-to-noise ratio (SNR) of each channel in data.
    Noise signal can be: 
    - another signal, noise; or
    - a specified part of data, starting from noise_start until noise_end (seconds).
    
    Args:
    	data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel 
            (if the channels are in a grid of (13,5))
        noise           : numpy.ndarray
            1D array containing noise reference for SNR calculation
        noise_start     : float
            If noise is not given, a specified part of data is used as noise reference, 
            starting from noise_start (seconds) until noise_end (seconds)
        noise_end       : float
            If noise is not given, a specified part of data is used as noise reference, 
            starting from noise_start (seconds) until noise_end (seconds)
        fs              : float
            Sampling frequency (Hz)
        
    Returns:
        snr             : numpy.ndarray
            1D array containing the SNR of each channel in data.
    """
    # Flattening data, excluding the empty channel, if the channels are in a grid of (13, 5)
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    n_x = data.shape[1]
    n_ch = data.shape[0]
    snr = np.zeros(n_ch, dtype="float")
    
    # If noise is not given, uses a specified part of data; 
    # starting from noise_start until noise_end (seconds); as the noise reference.
    if noise is None:
        for i in range(n_ch):
            noise = data[i][int(noise_start*fs):int(noise_end*fs)]
            snr[i] = np.square(data[i]).mean() / np.square(noise).mean()
    else:
        for i in range(n_ch):
            snr[i] = np.square(data[i]).mean() / np.square(noise).mean()
    return snr  


def search_zero_ch(data):
    """
    Locates channels in data that contain only zeros.
    
    Args:
    	data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel 
            (if the channels are in a grid of (13,5))
        
    Returns:
        zero_ch         : numpy.ndarray
            1D array containing channels in data that contain only zeros.
    """
    # Flattening data, excluding the empty channel, if the channels are in a grid of (13, 5)
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    n_x = data.shape[1]
    n_ch = data.shape[0]
    zero_ch = []
    
    # Checking for channels containing only zeros
    for i in range(n_ch):
        if np.all(data[i] == 0):
            zero_ch.append(i)
    zero_ch = np.array(zero_ch, dtype="int")
    return zero_ch


def bad_channels(data, signal="signal", thd_snr=2.0, noise = None, noise_start=0.0, noise_end=3.0, fs=2048):
    """
    Locates bad channels in data which:
    - have a SNR < thd_snr, or
    - contain only zeros.
    
    Args:
    	data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel 
            (if the channels are in a grid of (13,5))
        signal          : string
            String to name the signal in data
        thd_snr         : float
            SNR threshold of each channel in data
        noise           : numpy.ndarray
            1D array containing noise reference for SNR calculation
        noise_start     : float
            If noise is not given, a specified part of data is used as noise reference, 
            starting from noise_start (seconds) until noise_end (seconds)
        noise_end       : float
            If noise is not given, a specified part of data is used as noise reference, 
            starting from noise_start (seconds) until noise_end (seconds)
        fs              : float
            Sampling frequency (Hz)      
        
    Returns:
        zero_ch         : numpy.ndarray
            1D array containing channels in data that contain only zeros.
    """
    
    # Checking for channels with SNR < thd_snr
    snr = calc_snr(data, noise=noise, noise_start=noise_start, noise_end=noise_end, fs=fs)
    bad_ch = np.argwhere(snr < thd_snr)
    bad_ch = bad_ch.flatten()

    # Checking for channels with only zeros
    zero_ch = search_zero_ch(data)

    bad_ch = np.unique(np.append(bad_ch, zero_ch))
    print(f"Bad channels in {signal}: {bad_ch}")
    return bad_ch


def discard_ch_flat(discard_ch_vec, data):
    """
    Converts a boolean array (discard_ch_vec) containing values of whether or not to discard each channel, including an empty channel;
    to an integer array (disc_ch_flat) containing the channels to discard, excluding the empty channel;
    using the location of the empty channel in data.
    
    Args:
        discard_ch_vec  : numpy.ndarray
            1D or 2D boolean array containing values of whether or not to discard a channel in data
    	data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        data_ref        : numpy.ndarray
            1D array being the size reference of the resulting interpolation
        
    Returns:
        disc_ch_flat   : numpy.ndarray
            1D integer array containing the channels to discard
    """
    # Flattening data if the channels are in a 2D grid
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = data.flatten()
    # Location of empty channel in data:
    empty_ch = [channel for channel, signal in enumerate(data) if 0 in signal.shape]
    # Flattening discard_ch_vec if it is a 2D array 
    if discard_ch_vec.ndim == 2 :
        discard_ch_vec = discard_ch_vec.flatten()
    # Excluding empty channel and locating which channels to discard
    disc_ch_flat = np.delete(discard_ch_vec, empty_ch, axis=0)
    disc_ch_flat = np.argwhere(disc_ch_flat == 1).squeeze()
    return disc_ch_flat


def show_ext_factor(data, discard_ch, const=1000):
    """
    Calculates the extension factor that would be used after bad channels in discard_ch are discarded from data.
       
    Args:
        data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        discard_ch      : numpy.ndarray or list
            1D int array or list containing channel numbers to be discarded
        const           : int
            The number of measurements acquired after data has been extended
        
    Returns:
        ext_factor      : int
            Extension factor
    """
    # Calculating and showing extension factor
    if np.all(discard_ch) is not None and (data.shape[0] - discard_ch.size != 0):
        ext_factor = int(np.ceil(const/(data.shape[0] - discard_ch.size)))
    else:
        ext_factor = int(np.ceil(const/(data.shape[0])))
    print(f"Extension factor: {ext_factor}")
    print(discard_ch)

    if (data.shape[0] - discard_ch.size == 0):
        warnings.warn("Warning: 0 channels remaining!")
        print("Warning: 0 channels remaining!")
    return ext_factor


def notch_filter(data, freq_cutoff=50, fs=2048, q_factor=30): 
    b, a = iirnotch(w0=freq_cutoff, Q=q_factor, fs=fs)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def apply_filter(signal, b, a):
    filtered_data = lfilter(b=b, a=a, x=signal)
    return filtered_data


def env_data(data, fs=2048, order=4, l_bpf=10, h_bpf=900, lpf_cut=.2):
    """
    Signal from each channel in data is band-pass filtered, rectified, and low-pass filtered,
    resulting in a rectified envelope of each signal.
    These envelopes are averaged, resulting in a 1D array containing rectified envelope of the data.
    
    Args:
    	data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        fs		        : float
            Sampling frequency (Hz)
        order	        : int
            Order of filter
        l_bpf	        : float
            Lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	        : float
            Higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	        : float
            Cutoff frequency of the low-pass filter (Hz)
    
    Returns:
        envelope_data   : numpy.ndarray
            1D array containing rectified envelope of the EMG data, averaged
    """
    # Flattening data if the channels are in a 2D grid
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
        n_ch = flatten_signal(data).shape[0]
        data_flt = flatten_signal(data)
    else:
        n_x = data.shape[1]
        n_ch = data.shape[0]
        data_flt = data
    
    # Calculating envelope from each channel
    envelope_data = np.zeros((n_ch, n_x), dtype="float")    
    ## Bandpass filter
    b0, a0 = butter(order, [l_bpf, h_bpf], fs=fs, btype="band")
    ## Lowpass filter
    b1, a1 = butter(order, lpf_cut, fs=fs, btype="low")        
    for i in range(n_ch):
        # Bandpass filter
        bpfiltered_data = lfilter(b0, a0, data_flt[i])
        # Rectifying signal
        rectified_data = abs(bpfiltered_data)
        # Lowpass filter
        envelope_data[i] = lfilter(b1, a1, rectified_data)
    
    # Calculating average
    envelope_data_avg = envelope_data.mean(axis=0)

    return envelope_data_avg


def acquire_remove_ind(data, fs=2048, order=4, l_bpf=10, h_bpf=900, lpf_cut=.2, tol=5e-6):
    """
    Retrieves indices of ramps in-between flat EMG signal that are to be removed.
    Envelope of data is calculated, then the starting and end points of ramps are marked.
    
    Args:
    	data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        fs		    : float
            Sampling frequency (Hz)
        order	    : int
            Order of filter
        l_bpf	    : float
            Lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	    : float
            Higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	    : float
            Cutoff frequency of the low-pass filter (Hz)
        tol		    : float
            Tolerated value of gradient ~ 0 for marking the start/end of a ramp
    
    Returns:
    	remove_ind	: numpy.ndarray
            contains the indices of the ramps to be removed:
            - between data[remove_ind[0]] and data[remove_ind[1]]
            - between data[remove_ind[2]] and data[remove_ind[3]]
    """
    envelope_data = env_data(data=data, 
                             fs=fs, 
                             order=order, 
                             l_bpf=l_bpf, 
                             h_bpf=h_bpf, 
                             lpf_cut=lpf_cut)
    
    font_large = 16
    font_medium = 12
    font_small = 10
    
    # Plotting results
    x = np.arange(0, len(envelope_data))
    time = x / fs
    plt.rcParams['figure.figsize'] = [15,5]
    plt.rc('xtick', labelsize=font_small)
    plt.rc('ytick', labelsize=font_small)
    plt.plot(time, envelope_data)
    
    # Finding plateau and start/end of ramp (grad ~ 0)
    envelope_grad = np.gradient(envelope_data)
    flat_ind = np.argwhere(abs(envelope_grad)<=tol)
    
    thr = .25 * (np.max(envelope_data) - np.min(envelope_data))
    
    # Finding indices of start and end of ramps
    ramp_ind = np.array([], dtype="int64")
    for i in range(flat_ind.shape[0] - 1):
        if abs(envelope_data[flat_ind[i+1][0]] - envelope_data[flat_ind[i][0]]) > thr:
            ramp_ind = np.append( ramp_ind, 
                                 [flat_ind[i], flat_ind[i+1]] )
    # Removing duplicate points
    tmp = np.asarray([], dtype="int64")
    for i in range(ramp_ind.shape[0] - 1):
        if ( ramp_ind[i+1] - ramp_ind[i] < 5e2 or 
            (envelope_data[ramp_ind[i]] < thr and envelope_data[ramp_ind[i+1]] < thr ) ): 
            tmp = np.append(tmp, i)
    ramp_ind = np.delete(ramp_ind, tmp)
    
    # Marking start and end of ramp
    plt.scatter(time[flat_ind], envelope_data[flat_ind], c='r', s=40)
    plt.scatter(time[ramp_ind], envelope_data[ramp_ind], c='g', s=100)
    plt.show()
    
    # Indices to remove from the signal
    rm_ind = np.asarray([], dtype="int64")
    for i in range(1, len(ramp_ind) - 1):
        if ( (envelope_data[ramp_ind[i+1]] - envelope_data[ramp_ind[i]] > thr) and 
            (envelope_data[ramp_ind[i+1]] - envelope_data[ramp_ind[i]] > thr) ):
            rm_ind = np.append(rm_ind, [ramp_ind[i - 1], ramp_ind[i + 1]])
    
    # Marking parts to remove
    plt.plot(time, envelope_data)
    plt.scatter(time[rm_ind], envelope_data[rm_ind], c='r', s=40)
    plt.show()

    return rm_ind
    


def modify_signal(data, remove_ind):
    """
    Removes the following elements of data:
        data[remove_ind[0] : remove_ind[1]]
        data[remove_ind[2] : remove_ind[3]]
        ...
    and returns the remaining elements as data_mod.
    
    Args:
    	data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        remove_ind  : numpy.ndarray
            Array containing indices of parts in data that would be removed
            
    Returns:
        data_mod    : numpy.ndarray
            Array containing modified EMG data
    """

    # Acquiring indices to remove from data
    rm_ind_flt = np.ndarray.flatten(remove_ind)
    indices_to_remove = np.asarray([], dtype="int64")
    for i in range(len(rm_ind_flt) // 2):
        indices_to_remove = np.append( indices_to_remove, 
                                        [ np.arange(rm_ind_flt[2*i], rm_ind_flt[2*i + 1]) ] )
    
    # Removing indices from data
    if data.ndim == 1:
        data_mod = np.zeros(len(data)-len(indices_to_remove), dtype='float')
        if len(data) != 0:
            tmp = np.delete(data, indices_to_remove)
            data_mod = np.asarray([tmp])
    elif (data.ndim == 2 and (data[0][0].size != 0 and data[12][0].size != 0)):
        data_mod = np.zeros((data.shape[0], data.shape[1]-len(indices_to_remove)), dtype='float')
        for i in range(data.shape[0]):
            if len(data[i]) != 0:
                tmp = np.delete(data[i], indices_to_remove)
                data_mod[i] = np.asarray([tmp]) 
    elif (data[0][0].size == 0 or
          data[12][0].size == 0 and data.ndim == 2) or data.ndim == 3:
        data_mod = copy.deepcopy(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if len(data[i][j]) != 0:
                    tmp = np.delete(data[i][j][0], indices_to_remove)
                    data_mod[i][j] = np.asarray([tmp]) 

    return data_mod


def crop_data(data, start=0.0, end=40.0, fs=2048):
    """
    Returns cropped data between start and end (in seconds).
    
    Args:
    	data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        start       : float
            Starting time (in seconds) of the cropped data
        end         : float
            End time (in seconds) of the cropped data
        fs          : int
            Sampling frequency (Hz)
            
    Returns:
        data_crop   : numpy.ndarray
            Array containing EMG data in the specified window
    """
    if data.ndim == 1:
        n_x = data.size
        if end > (n_x / fs):
            end = n_x / fs
        nx_start = int(start*fs)
        nx_end = int(end*fs)
        data_crop = np.array(data[nx_start: nx_end])
    
    elif data.shape[0] > 12:
        if (data.ndim == 2 and (data[0][0].size != 0 and data[12][0].size != 0)):
            n_x = data.shape[1]
            n_ch = data.shape[0]
            if end > (n_x / fs):
                end = n_x / fs
            nx_start = int(start*fs)
            nx_end = int(end*fs)
            data_crop = np.zeros((n_ch, nx_end - nx_start), dtype="float")
            for i in range(0, data_crop.shape[0]):
                if np.size(data_crop[i]) != 0:
                    data_crop[i] = np.asarray([data[i][nx_start : nx_end]])

            data_crop = data[:, int(start*fs): int(end*fs)]

        elif (data[0][0].size == 0 or
            data[12][0].size == 0 and data.ndim == 2) or data.ndim == 3:
            # Flattening data if the channels are in a 2D grid
            n_x = data[0][1].shape[1]
            if end > (n_x / fs):
                end = n_x / fs
            data_crop = copy.deepcopy(data)
            for i in range(0, data_crop.shape[0]):
                for j in range(0, data_crop.shape[1]):
                    if np.size(data_crop[i][j]) != 0:
                        data_crop[i][j] = np.asarray([data[i][j][0][int(start*fs): int(end*fs)]])
    else:
        n_x = data.shape[1]
        n_ch = data.shape[0]
        if end > (n_x / fs):
            end = n_x / fs
        nx_start = int(start*fs)
        nx_end = int(end*fs)
        data_crop = np.zeros((n_ch, nx_end - nx_start), dtype="float")
        for i in range(0, data_crop.shape[0]):
            if np.size(data_crop[i]) != 0:
                data_crop[i] = np.asarray([data[i][nx_start : nx_end]])

        data_crop = data[:, int(start*fs): int(end*fs)]
    
    return data_crop

    

def crop_ind_pt(ind_pt, data, start=0.0, end=40.0, fs=2048):
    """
    Returns the cropped pulse train (ind_pt) and cropped data between start and end (in seconds).
    
    Args:
    	ind_pt  	: numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        start       : float
            Starting time (in seconds) of the cropped data
        end         : float
            End time (in seconds) of the cropped data
        fs          : int
            Sampling frequency (Hz)
            
    Returns:
        ind_pt_crop : numpy.ndarray
            Indices of motor units' pulse trains in the specified window
        data_crop   : numpy.ndarray
            (13, 5) array containing EMG data in the specified window
    """
    # Cropping data
    data_crop = crop_data(data, start=start, end=end, fs=fs)
    n_mu = ind_pt.shape[0]
    # Flattening data if the channels are in a 2D grid
    if ((data_crop[0][0].size == 0 or
         data_crop[12][0].size == 0) and data_crop.ndim == 2) or data_crop.ndim == 3:
        ind_pt = ind_pt.squeeze()
        n_mu = ind_pt.shape[0]
        x = data_crop[0][1].shape[1]
    else:
        x = data_crop.shape[1]

    start_idx = int(start * fs)
    end_idx = start_idx + x
    
    # Acquiring indices from each pulse train that are between start_idx and end_idx
    ind_pt_crop = []
    for i in range(n_mu):
        tmp = np.where( np.logical_and( ind_pt[i] > start_idx, ind_pt[i] < end_idx ) )
        if tmp[0].size == 0:
            add_ind = []
        else:
            add_ind = ind_pt[i][tmp] - start_idx
        ind_pt_crop.append(np.array(add_ind, dtype="int64"))
    
    ind_pt_crop = np.array(ind_pt_crop, dtype="object")
    
    return ind_pt_crop, data_crop


def ind_to_pt(ind_pt, data):
    """
    Converts array containing indices where the motor units' pulse trains are firing (ind_pt)
    into pulse trains (pt).
    
    Args:
    	ind_pt  : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))

    Returns:
        pt      : numpy.ndarray
            Array containing pulse trains    
    """
    
    n_mu = ind_pt.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Pulse train
    pt = np.zeros((n_mu, n_x), dtype="int64")
    for i in range(n_mu):
        pt[i][ind_pt[i]] = 1
    
    return pt

