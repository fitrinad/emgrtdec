from emgdecompy.preprocessing import *
from emgdecompy.decomposition import *

from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import numpy as np
from functions.preprocessing import *

#############################################################################################################################
### Decomposition module  ###################################################################################################
############################################################################################################################# 
def rt_decomp(data, B_realtime, 
              mean_tm=None, discard=None, R=16, 
              bandpass=True, lowcut=10, highcut=900, order=6, l=31, 
              random_seed = None, 
              classify_mu=False, thd_sil=0.85, 
              use_pps=False, thd_pps=5, 
              sc_tm=None, nc_tm=None, fs=2048):    
    """
    Applies the realtime separation matrix to the training data and classifies detected peaks to obtain signal centroids and
    noise centroids of each MU used for the realtime decomposition.
    
    Args:
        data        : numpy.ndarray
            Raw EMG signal
        B_realtime  : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x
        bandpass    : bool
            Applies a Butterworth bandpass filter before extending data if True
        lowcut      : float
            Lower cutoff frequency of the bandpass filter
        highcut     : float
            Higher cutoff frequency of the bandpass filter
        order       : int
            Order of bandpass filter
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
        classify_mu : bool
            Uses silhouette score (SIL) to classify spikes if True
        thd_sil     : float
            Threshold SIL value used to classify spikes
        use_pps     : bool
            Uses the firing rate (pps/pulse per second) to classify spikes if True
        thd_pps     : int
            Threshold pps value used to classify spikes
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU
        fs          : float
            Sampling frequency

    Returns:
        MUPulses            : numpy.ndarray
            Firing indices for each motor unit
        signal_centroids    : numpy.ndarray
            Signal centroids calculated from the k-means clustering in the sort_peaks step
        noise_centroids     : numpy.ndarray
            Noise centroids calculated from the k-means clustering in the sort_peaks step
    """
    # 1. Source extraction
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    ## Discarding channels
    if np.all(discard) is not None:
        data = np.delete(data, discard, axis=0)
    ## Band-pass filter
    if bandpass:
        data = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=data,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)
    ## Extend
    data_ext = extend_all_channels(data, R)
    if mean_tm is not None: # Using mean from training module
        # data_ext - mean_tm
        data_ext = data_ext.T - mean_tm.T
        data_ext = data_ext.T
    else:                   # Using mean from realtime data
        # data_ext - data.mean
        data_ext = center_matrix(data_ext)
    ## Source extraction
    s = np.dot(B_realtime.T, data_ext)

    ## Squared source vectors
    s2 = np.square(s)
    if s2.ndim == 1:
        s2 = np.array([s2])
    peak_indices = []
    MUPulses = []
    sil_scores = []
    signal_centroids = []
    noise_centroids = []
    
    for i in range(s2.shape[0]):
        # 2. Peak extraction
        ## Detecting peaks in s2
        peak_indices_i , _ = find_peaks(s2[i], distance=l)
        peak_indices.append(peak_indices_i.astype("int64"))

        signal_i, noise_i = sort_peaks(s2_i=s2[i], peak_indices_i=peak_indices[i],
                                    random_seed=random_seed)
        peak_indices_signal_i = signal_i["peak_indices_signal"] 
        signal_centroids.append(signal_i["signal_centroid"])
        noise_centroids.append(noise_i["noise_centroid"])
        
        # 3. Spike classification
        add_peaks = np.array(peak_indices_signal_i)
        if classify_mu: ## whether MU i is firing or not, based on SIL
            # Silhouette score
            sil = silhouette_score(s2[i], add_peaks)
            sil_scores.append(sil)
            if sil < thd_sil:
                add_peaks = np.array([], dtype="int64")
        ## Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        ## adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                            abs(s2[i][add_peaks] - (sc_tm[i])) )
            add_peaks = add_peaks[add_indices]
        
        if use_pps: ## Rejecting detected pulses if < thd_spikes
            thd_spikes = int(thd_pps * s2.shape[1] / fs)
            ### number of spikes in current window
            n_spikes_curr_i = len(add_peaks)
            if (n_spikes_curr_i < thd_spikes):
                add_peaks = np.array([], dtype="int64")
            else:
                add_peaks = np.array(add_peaks, dtype="int64")

        MUPulses.append(add_peaks.squeeze())
    
    MUPulses = np.array(MUPulses, dtype="object")
    sil_scores = np.array(sil_scores, dtype="float")
    signal_centroids = np.array(signal_centroids, dtype="float")
    noise_centroids = np.array(noise_centroids, dtype="float")

    return MUPulses, signal_centroids, noise_centroids


def rt_decomp_live(data, B_realtime, batch_size, 
                   n_updates, prev_timediff, time_diff, 
                   prev_MUPulses, 
                   mean_tm=None, discard=None, R=16, l=31,
                   classify_mu=False, thd_sil=0.85, 
                   use_pps=False, thd_pps=5, 
                   sc_tm=None, nc_tm=None, fs=2048):    
    """
    Used for realtime decomposition in the decomposition module window (without visualization).
    
    Args:
        data            : numpy.ndarray
            Raw EMG signal
        B_realtime      : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        batch_size      : float
            The signal duration taken for each computation (in seconds)
        n_updates       : int
            The number of batches that has been processed
        prev_timediff   : float
            Time difference between the previous batch and the starting time
        time_diff       : float
            Time difference between the current batch and the starting time
        prev_MUPulses   : numpy.ndarray
            Matrix containing firing indices of each MU from the starting time up to the previous batch
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
        classify_mu : bool
            Uses silhouette score (SIL) to classify spikes if True
        thd_sil     : float
            Threshold SIL value used to classify spikes
        use_pps     : bool
            Uses the firing rate (pps/pulse per second) to classify spikes if True
        thd_pps     : int
            Threshold pps value used to classify spikes
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU
        fs          : float
            Sampling frequency

    Returns:
        MUPulses    : numpy.ndarray
            Firing indices for each motor unit
        n_updates   : int
            The number of batches that has been processed
    """
    # 1. Source extraction
    ## Discarding channels
    if np.all(discard) is not None:
        data = np.delete(data, discard, axis=0)
    ## Extend
    data_ext = extend_all_channels(data, R)
    if mean_tm is not None: # Using mean from training module
        # data_ext - mean_tm
        data_ext = data_ext.T - mean_tm.T
        data_ext = data_ext.T
    else:                   # Using mean from realtime data
        # data_ext - data.mean
        data_ext = center_matrix(data_ext)
    ## Source extraction
    s = np.dot(B_realtime.T, data_ext)

    ## Squared source vectors
    s2 = np.square(s)
    peak_indices = []
    MUPulses = []
    sil_scores = []
    
    for i in range(s2.shape[0]):
        # 2. Peak extraction
        ## Detecting peaks in s2
        peak_indices_i , _ = find_peaks(s2[i], distance=l)
        peak_indices.append(peak_indices_i.astype("int64"))
        peak_indices_signal_i = peak_indices_i 
        
        # 3. Spike classification
        add_peaks = np.array(peak_indices_signal_i)
        if classify_mu: ## whether MU i is firing or not, based on SIL
            # Silhouette score
            sil = silhouette_score(s2[i], add_peaks)
            sil_scores.append(sil)
            if sil < thd_sil:
                add_peaks = np.array([], dtype="int64")
        ## Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        ## adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                            abs(s2[i][add_peaks] - (sc_tm[i])) )
            add_peaks = add_peaks[add_indices]
        
        if use_pps: ## Rejecting detected pulses if < thd_spikes
            thd_spikes = int(thd_pps * s2.shape[1] / fs)
            ### number of spikes in current window
            n_spikes_curr_i = len(add_peaks)
            if (n_spikes_curr_i < thd_spikes):
                add_peaks = np.array([], dtype="int64")
            else:
                add_peaks = np.array(add_peaks, dtype="int64")
        
        if n_updates == 0:  # First batch
            MUPulses.append(add_peaks.squeeze())
        
        else: 
            # Adding peaks to pulse train
            add_peaks = add_peaks + int((time_diff - batch_size) * fs)
            add_MUPulses_i = add_peaks[(add_peaks >= (prev_timediff) * fs)]
            tmp = np.concatenate((prev_MUPulses[i], add_MUPulses_i), axis=None, dtype="int64")
            MUPulses.append(tmp)
    n_updates += 1
            
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
    sil_scores = np.array(sil_scores, dtype="float")

    return MUPulses, n_updates


def rt_decomp_plotpg(data, B_realtime, batch_size, 
                     n_updates, prev_timediff, time_diff, max_nsamples,
                     current_pt, line_handler, time_axis, prev_MUPulses, 
                     mean_tm=None, discard=None, R=16, l=31, 
                     classify_mu=False, thd_sil=0.85, 
                     use_pps=False, thd_pps=5, 
                     sc_tm=None, nc_tm=None, fs=2048):    
    """
    Used for realtime decomposition in the decomposition module window (with visualization using pyqtgraph).
    
    Args:
        data            : numpy.ndarray
            Raw EMG signal
        B_realtime      : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        batch_size      : float
            The signal duration taken for each computation (in seconds)
        n_updates       : int
            The number of batches that has been processed
        prev_timediff   : float
            Time difference between the previous batch and the starting time
        time_diff       : float
            Time difference between the current batch and the starting time
        max_nsamples    : int
            Maximum number of data points that can be displayed on the user interface
        current_pt      : numpy.ndarray
            Pulse train of the current batch that is to be computed in the current step
        line_handler    : pyqtgraph.plot
            Used to set the data being visualized on each row of the plots 
        time_axis       : numpy.ndarray
            Time axis displayed on the x-axis of the plots
                    
        prev_MUPulses   : numpy.ndarray
            Matrix containing firing indices of each MU from the starting time up to the previous batch
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
        classify_mu : bool
            Uses silhouette score (SIL) to classify spikes if True
        thd_sil     : float
            Threshold SIL value used to classify spikes
        use_pps     : bool
            Uses the firing rate (pps/pulse per second) to classify spikes if True
        thd_pps     : int
            Threshold pps value used to classify spikes
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU
        fs          : float
            Sampling frequency

    Returns:
        MUPulses        : numpy.ndarray
            Firing indices for each motor unit
        line_handler    : pyqtgraph.plot
            Used to set the data being visualized on each row of the plots 
        n_updates       : int
            The number of batches that has been processed
        current_pt      : numpy.ndarray
            Pulse train of the current batch that is to be computed in the current step
    """
    
    # 1. Source extraction
    ## Discarding channels
    if np.all(discard) is not None:
        data = np.delete(data, discard, axis=0)
    ## Extend
    data_ext = extend_all_channels(data, R)
    if mean_tm is not None: # Using mean from training module
        # data_ext - mean_tm
        data_ext = data_ext.T - mean_tm.T
        data_ext = data_ext.T
    else:                   # Using mean from realtime data
        # data_ext - data.mean
        data_ext = center_matrix(data_ext)
    ## Source extraction
    s = np.dot(B_realtime.T, data_ext)

    ## Squared source vectors
    s2 = np.square(s)
    peak_indices = []
    MUPulses = []
    sil_scores = []
     
    for i in range(s2.shape[0]):
        # 2. Peak extraction
        ## Detecting peaks in s2
        peak_indices_i , _ = find_peaks(s2[i], distance=l)
        peak_indices.append(peak_indices_i.astype("int64"))
        peak_indices_signal_i = peak_indices_i # signal_i["peak_indices_signal"] 
        
        # 3. Spike classification
        add_peaks = np.array(peak_indices_signal_i)
        if classify_mu: ## whether MU i is firing or not, based on SIL
            # Silhouette score
            sil = silhouette_score(s2[i], add_peaks)
            sil_scores.append(sil)
            if sil < thd_sil:
                add_peaks = np.array([], dtype="int64")
        ## Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        ## adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                            abs(s2[i][add_peaks] - (sc_tm[i])) )
            add_peaks = add_peaks[add_indices]
        
        if use_pps: ## Rejecting detected pulses if < thd_spikes
            thd_spikes = int(thd_pps * s2.shape[1] / fs)
            ### number of spikes in current window
            n_spikes_curr_i = len(add_peaks)
            if (n_spikes_curr_i < thd_spikes):
                add_peaks = np.array([], dtype="int64")
            else:
                add_peaks = np.array(add_peaks, dtype="int64")
        
        if n_updates == 0: # First batch
            MUPulses.append(add_peaks)
            # Adding peaks to pulse train
            current_pt[i][add_peaks] = 1
            # Adding pulse train to plot
            line_handler[i].setData(time_axis, current_pt[i])  
        else:
            # Adding peaks to pulse train
            add_peaks = add_peaks + int((time_diff - batch_size) * fs)
            add_MUPulses_i = add_peaks[(add_peaks >= (prev_timediff) * fs)]
            tmp = np.concatenate((prev_MUPulses[i], add_MUPulses_i), axis=None, dtype="int64")
            MUPulses.append(tmp)
            if (time_diff * fs) <= max_nsamples:
                current_pt[i][tmp] = 1
            else:
                tmp = (tmp[tmp > (time_diff * fs) - max_nsamples] - 
                    int((time_diff*fs) - max_nsamples))
                current_pt[i] = np.zeros(max_nsamples)
                current_pt[i][tmp] = 1
            # Adding pulse train to plot
            line_handler[i].setData(time_axis, current_pt[i])
    n_updates += 1
            
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
    sil_scores = np.array(sil_scores, dtype="float")

    return MUPulses, line_handler, n_updates, current_pt




#############################################################################################################################
### Simulation of realtime decomposition  ###################################################################################
#############################################################################################################################
def source_extraction(x, B_realtime, mean_tm=None, discard=None, 
                      bandpass=True, lowcut=10, highcut=900, fs=2048, order=6, 
                      R=16):
    """
    Returns matrix containing source vectors estimated from the EMG signal (x).

    Args:
        x           : numpy.ndarray
            Raw EMG signal
        B_realtime  : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        bandpass    : bool
            Applies a Butterworth bandpass filter before extending data if True
        lowcut      : float
            Lower cutoff frequency of the bandpass filter
        highcut     : float
            Higher cutoff frequency of the bandpass filter
        fs          : float
            Sampling frequency
        order       : int
            Order of bandpass filter
        R           : int
            How far to extend x

    Returns:
        s           : numpy.ndarray
            Matrix containing estimated source vectors
        x_ext       : numpy.ndarray
            Extended EMG signal
    """    

    # Flattening data, excluding the empty channel, if the channels are in a grid of (13, 5)
    if ((x[0][0].size == 0 or
         x[12][0].size == 0) and x.ndim == 2) or x.ndim == 3:    
        x = flatten_signal(x)

    # Discarding channels
    if np.all(discard) is not None:
        x = np.delete(x, discard, axis=0)

    # band-pass filter
    if bandpass:
        x = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)

    # Extend
    x_ext = extend_all_channels(x, R)

    if mean_tm is not None: # Use mean from training module
        # x_ext - mean_tm
        x_ext = x_ext.T - mean_tm.T
        x_ext = x_ext.T
    else: # Use mean from realtime data
        # x_ext - x.mean
        x_ext = center_matrix(x_ext)

    # Source extraction
    s = np.dot(B_realtime.T, x_ext)

    return s, x_ext


def peak_extraction(s, l=31):
    """
    Detects and extracts peaks from each squared source vector s[i] from matrix s.

    Args:
        s           : numpy.ndarray
            Matrix containing estimated source vectors
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
    Returns:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
    """
    # Squared source vectors
    s2 = np.square(s)
    
    peak_indices = []
    # Detecting peaks in s2
    for i in range(s2.shape[0]):
        peak_indices_i , _ = find_peaks(s2[i], distance=l)
        peak_indices.append(peak_indices_i.astype("int64"))

    length = len(peak_indices[0])
    if any(len(arr) != length for arr in peak_indices):
        peak_indices = np.array(peak_indices, dtype="object")
    else:
        peak_indices = np.array(peak_indices, dtype="int64")

    return s2, peak_indices


def sort_peaks(s2_i, peak_indices_i, random_seed=None):
    """
    Separates extracted peaks into relatively large peaks (signal) and relatively small peaks (noise) 
    using k-means clustering.

    Args:
        s2_i            : numpy.ndarray
            Source vector for MU i
        peak_indices_i  : numpy.ndarray
            Vector containing indices of detected peaks from s2_i
        
    Returns:
        signal  : dict
            peak_indices_signal  : indices of peaks that are classified into the signal cluster
            signal_centroid      : centroid (mean) of the signal cluster
        noise   : dict
            peak_indices_noise  : indices of peaks that are classified into the noise cluster
            noise_centroid      : centroid (mean) of the noise cluster
    """
    peak_indices_signal = []
    peak_indices_noise = []

    # Separating large peaks from relatively small peaks (noise)
    kmeans = KMeans(n_clusters=2, random_state=random_seed)
    kmeans.fit(s2_i[peak_indices_i].reshape(-1,1))

    # Signal cluster centroid (sc_i)
    sc_i = np.argmax(kmeans.cluster_centers_)
    # Determining which cluster contains large peaks (signal)
    signal_indices = kmeans.labels_.astype(bool) # if sc_i == 1
    if sc_i == 0:
        signal_indices = ~signal_indices      
    
    # Indices of the peaks in signal cluster
    peak_indices_signal_i = peak_indices_i[signal_indices]
    peak_indices_noise_i = peak_indices_i[~signal_indices]
    peak_indices_signal.append(peak_indices_signal_i)
    peak_indices_noise.append(peak_indices_noise_i)

    # Signal cluster and Noise cluster
    signal_cluster = s2_i[peak_indices_signal_i]
    noise_cluster = np.delete(s2_i, peak_indices_signal_i)

    # Centroids
    signal_centroid = signal_cluster.mean()
    noise_centroid = noise_cluster.mean()

    signal = {"peak_indices_signal": peak_indices_signal,
            "signal_centroid": signal_centroid}
    noise = {"peak_indices_noise": peak_indices_noise,
            "noise_centroid": noise_centroid}

    return signal, noise


def spike_classification(s2, peak_indices,
                         classify_mu=True,
                         thd_sil=0.9,
                         sc_tm=None, nc_tm=None):
    """
    Returns a matrix of motor unit pulses.

    Args:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
        
        thd_sil         : float
            Threshold of the silhouette score
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU
    
    Returns:
        MUPulses        : numpy.ndarray
            Matrix containing indices of motor unit pulses
    """
    
    peak_indices_signal = peak_indices
    MUPulses = []
    sil_scores = []
    for i in range(s2.shape[0]):
        add_peaks = peak_indices_signal[i]
        if classify_mu: # whether MU i is firing or not, based on SIL or centroid distance
            add_peaks = peak_indices_signal[i]
            # Silhouette score
            sil = silhouette_score(s2[i], peak_indices_signal[i])
            sil_scores.append(sil)
            if sil < thd_sil:
                add_peaks = []
            
        try:
            # Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
            # adding peaks closer to signal centroid
            if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):             
                add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                                abs(s2[i][add_peaks] - (sc_tm[i])) )
                add_peaks = add_peaks[add_indices] 
        except IndexError:
            pass
        MUPulses.append(np.array(add_peaks, dtype="int64"))
    
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")  
    
    return MUPulses


def batch_decomp(data, B_realtime, mean_tm=None, discard=None, R=16,
                 bandpass=True, lowcut=10, highcut=900, order=6, l=31,
                 classify_mu=True, thd_sil=0.9,
                 sc_tm=None, nc_tm=None,
                 batch_size=4.0, overlap=0.0, fs=2048):
    """
    Used for simulated realtime decomposition.
    
    Args:
        data            : numpy.ndarray
            Raw EMG signal
        B_realtime      : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x
               
        bandpass    : bool
            Applies a Butterworth bandpass filter before extending data if True
        lowcut      : float
            Lower cutoff frequency of the bandpass filter
        highcut     : float
            Higher cutoff frequency of the bandpass filter
        order       : int
            Order of bandpass filter
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
        classify_mu : bool
            Uses silhouette score (SIL) to classify spikes if True
        thd_sil     : float
            Threshold SIL value used to classify spikes
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU

        batch_size  : float
            The signal duration taken for each computation (in seconds)
        overlap     : float
            Time overlap between two batches
        fs          : float
            Sampling frequency

    Returns:
        decomp          : dict
            MUPulses    : numpy.ndarray
                Firing indices for each motor unit
    """

    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        end_time = data[0][1].shape[1] / fs
    else:
        end_time = data.shape[1] / fs

    time = 0.0
    while True:
        raw = crop_data(data, start = time, end = time+batch_size)

        # Source extraction
        s, _ = source_extraction(x = raw, 
                                 B_realtime = B_realtime,
                                 mean_tm = mean_tm, 
                                 discard=discard,
                                 bandpass = bandpass,
                                 lowcut=lowcut,
                                 highcut=highcut,
                                 fs=fs,
                                 order=order,
                                 R=R)
        if s.ndim == 1:
            s = np.array([s])

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        if s2.ndim == 1:
            s2 = np.array([s2])
        
        # Spike classification
        if time == 0.0:
            MUPulses = spike_classification(s2=s2, 
                                            peak_indices=peak_indices, 
                                            classify_mu=classify_mu, thd_sil=thd_sil, 
                                            sc_tm=sc_tm, nc_tm=nc_tm)
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr = spike_classification(s2=s2, 
                                                 peak_indices=peak_indices, 
                                                 classify_mu=classify_mu, thd_sil=thd_sil, 
                                                 sc_tm=sc_tm, nc_tm=nc_tm)
            MUPulses_curr = MUPulses_curr + int(time*fs)
           
            for j in range(MUPulses.shape[0]):
                add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (time+overlap)*fs]
                tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
                
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time-overlap:
            break

    decomp = {"MUPulses": MUPulses}

    return decomp


def batch_decomp_window(data, B_realtime, mean_tm=None, discard=None, 
                        bandpass=True, lowcut=10, highcut=900, order=6, l=31,
                        classify_mu=True, thd_sil=0.9, thd_pps=5,
                        sc_tm=None, nc_tm=None,
                        batch_size=0.6, overlap=0.3, fs=2048, R=16):
    """
    Used for simulated realtime decomposition, with a pulse per second (pps) thresholding step.
    
    Args:
        data            : numpy.ndarray
            Raw EMG signal
        B_realtime      : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        mean_tm     : numpy.ndarray
            Vector containing mean values of each channel from the training data
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x
               
        bandpass    : bool
            Applies a Butterworth bandpass filter before extending data if True
        lowcut      : float
            Lower cutoff frequency of the bandpass filter
        highcut     : float
            Higher cutoff frequency of the bandpass filter
        order       : int
            Order of bandpass filter
        l           : int
            Minimum distance between detected peaks when applying the peak-finding algorithm
        
        classify_mu : bool
            Uses silhouette score (SIL) to classify spikes if True
        thd_sil     : float
            Threshold SIL value used to classify spikes
        thd_pps     : int
            Threshold pps value used to classify spikes            
        sc_tm       : numpy.ndarray
            Vector containing signal centroids for each MU
        nc_tm       : numpy.ndarray
            Vector containing noise centroids for each MU

        batch_size  : float
            The signal duration taken for each computation (in seconds)
        overlap     : float
            Time overlap between two batches
        fs          : float
            Sampling frequency

    Returns:
        decomp          : dict
            MUPulses    : numpy.ndarray
                Firing indices for each motor unit
    """
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        end_time = data[0][1].shape[1] / fs
    else:
        end_time = data.shape[1] / fs
    time = 0.0
    while True:
        raw = crop_data(data, start = time, end = time+batch_size)

        # Source extraction
        s, _ = source_extraction(x = raw, 
                                 B_realtime = B_realtime,
                                 mean_tm = mean_tm, 
                                 discard=discard,
                                 bandpass = bandpass,
                                 lowcut=lowcut,
                                 highcut=highcut,
                                 fs=fs,
                                 order=order,
                                 R=R)

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses = spike_classification(s2=s2, 
                                            peak_indices=peak_indices, 
                                            classify_mu=classify_mu, thd_sil=thd_sil, 
                                            sc_tm=sc_tm, nc_tm=nc_tm)
            
            thd_spikes = int(thd_pps * s2.shape[1]/fs)
                
            for i in range(MUPulses.shape[0]):
                # number of spikes in current window
                n_spikes_curr_i = MUPulses[i].shape[0]
                if (n_spikes_curr_i < thd_spikes):
                    MUPulses[i] = np.array([], dtype="int64")
                
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr = spike_classification(s2=s2, 
                                                 peak_indices=peak_indices, 
                                                 classify_mu=classify_mu, thd_sil=thd_sil, 
                                                 sc_tm=sc_tm, nc_tm=nc_tm)
             
            MUPulses_curr = MUPulses_curr + int(time*fs)

            thd_spikes = int(thd_pps * s2.shape[1]/fs)

            for i in range(MUPulses.shape[0]):
                # number of spikes in current window
                n_spikes_curr_i = MUPulses_curr[i].shape[0]
                
                if (n_spikes_curr_i >= thd_spikes) and (n_spikes_curr_i >= thd_pps):
                    add_MUPulses = MUPulses_curr[i][MUPulses_curr[i] >= (time+overlap)*fs]
                else:
                    add_MUPulses = np.array([], dtype="int64")
                MUPulses_curr_i = np.append(MUPulses[i], add_MUPulses)
                tmp.append( np.array( MUPulses_curr_i, dtype="int64" ) ) 
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time-overlap:
            break

    decomp = {"MUPulses": MUPulses}
    return decomp



#############################################################################################################################
### Plotting ################################################################################################################
#############################################################################################################################
def plot_extracted_peaks(s2, peak_indices, fs=2048, title="extracted peaks"):
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    time = np.arange(0, s2.shape[1], dtype="float") / float(fs)
    
    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=40)
    for i in range(s2.shape[0]):
        y = s2[i]
        ax[i].plot(time, y)
        ax[i].set_ylabel(f"MU {i}", fontsize=20)
        if len(peak_indices[i]) != 0:
            ax[i].scatter(peak_indices[i]/fs, s2[i][peak_indices[i]], c='r', s=40)
    plt.show()


def plot_classified_spikes(s2, peak_indices, MUPulses, fs=2048, 
                           title="classified spikes", label1="detected peaks", label2="MUPulses"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    time = np.arange(0, s2.shape[1], dtype="float") / float(fs)
    
    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[i].plot(time, s2[i])
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if len(peak_indices[i]) != 0:
            ax[i].scatter(peak_indices[i]/fs, s2[i][peak_indices[i]], c='g', s=70, label=label1)
        if len(MUPulses[i]) != 0:
            ax[i].scatter(MUPulses[i]/fs, s2[i][MUPulses[i]], c='r', s=40, label=label2)
        if i == 0:
            ax[i].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()


def plot_peaks(s2, noise, signal, centroid_dists, fs=2048, title="extracted peaks"):
    font_large = 30
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    t_axis = np.arange(0, s2.shape[1], dtype="float") / float(fs)

    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[i].plot(t_axis, s2[i], label=r"$s^2$")
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if noise["peak_indices_noise"][i].size != 0:
            ax[i].scatter(noise["peak_indices_noise"][i]/fs, s2[i][noise["peak_indices_noise"][i]], c='r', s=40, label="noise")
        ax[i].scatter(signal["peak_indices_signal"][i]/fs, s2[i][signal["peak_indices_signal"][i]], c='g', s=40, label="signal")
        ax[i].scatter(signal["peak_indices_signal"][i][signal["signal_clusters"][i].argmax()]/fs, signal["signal_clusters"][i].max(), c='c', s=80)
        ax[i].xaxis.set_tick_params(labelsize=font_small)
        ax[i].yaxis.set_tick_params(labelsize=font_small)
        ax[i].text(.02,.95, f"centroid_dist={centroid_dists[i]:.4f}", fontsize=font_medium, transform=ax[i].transAxes)
        ax[i].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()


def plot_peaks_pulses(s2, noise, signal, sc_tm, nc_tm, MUPulses, fs=2048, title="extracted peaks"):
    font_large = 30
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0] * 2
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    t_axis = np.arange(0, s2.shape[1], dtype="float") / float(fs)

    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[2*i].plot(t_axis, s2[i], label=r"$s^2$")
        ax[2*i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if noise["peak_indices_noise"][i].size != 0:
            ax[2*i].scatter(noise["peak_indices_noise"][i]/fs, s2[i][noise["peak_indices_noise"][i]], c='r', s=40, label="noise")
        ax[2*i].scatter(signal["peak_indices_signal"][i]/fs, s2[i][signal["peak_indices_signal"][i]], c='g', s=40, label="signal")
        ax[2*i].scatter(signal["peak_indices_signal"][i][signal["signal_clusters"][i].argmax()]/fs, signal["signal_clusters"][i].max(), c='c', s=80)
        ax[2*i].xaxis.set_tick_params(labelsize=font_small)
        ax[2*i].yaxis.set_tick_params(labelsize=font_small)
        ax[2*i].text(.02,.95, f"sc={sc_tm[i]:.4f}\nnc={nc_tm[i]:.4f}", fontsize=font_medium, transform=ax[2*i].transAxes)
        ax[2*i].legend(loc='upper right', shadow=False, fontsize=font_medium)

        ax[2*i+1].plot(t_axis, s2[i])
        ax[2*i+1].set_ylabel(f"MU {i}", fontsize=20)
        ax[2*i+1].scatter(MUPulses[i]/fs, s2[i][MUPulses[i]], c='g', s=40, label="MUPulses")
        ax[2*i+1].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()

