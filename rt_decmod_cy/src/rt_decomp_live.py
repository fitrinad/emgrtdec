from emgdecompy.preprocessing import (flatten_signal, butter_bandpass_filter, extend_all_channels, center_matrix)
from emgdecompy.decomposition import silhouette_score

from scipy.signal import find_peaks
import numpy as np
from rt_decmod_cy.sort_peaks_rt_cy import sort_peaks_rt_cy

def rt_decomp_live_cy(data, B_realtime, batch_size, 
                   n_updates, prev_timediff, time_diff, 
                   prev_MUPulses, 
                   mean_tm=None, discard=None, R=16, 
                   l=31, random_seed = None, 
                   classify_mu=True, thd_sil=0.85, 
                   use_pps=False, thd_pps=5, 
                   sc_tm=None, nc_tm=None, fs=2048):    
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
    signal_centroids = []
    noise_centroids = []
    
    if n_updates == 0: # First batch
        n_updates += 1
        for i in range(s2.shape[0]):
            # 2. Peak extraction
            ## Detecting peaks in s2
            peak_indices_i , _ = find_peaks(s2[i], distance=l)
            peak_indices.append(peak_indices_i.astype("int64"))

            signal_i, noise_i = sort_peaks_rt_cy(s2_i=s2[i], peak_indices_i=peak_indices[i], 
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
            
    else: 
        n_updates += 1
        for i in range(s2.shape[0]):
            # 2. Peak extraction
            ## Detecting peaks in s2
            peak_indices_i , _ = find_peaks(s2[i], distance=l)
            peak_indices.append(peak_indices_i.astype("int64"))

            signal_i, noise_i = sort_peaks_rt_cy(s2_i=s2[i], peak_indices_i=peak_indices[i],
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
            # Adding peaks to pulse train
            add_peaks = add_peaks + int((time_diff - batch_size) * fs)
            add_MUPulses_i = add_peaks[(add_peaks >= (prev_timediff) * fs)]
            tmp = np.concatenate((prev_MUPulses[i], add_MUPulses_i), axis=None, dtype="int64")
            MUPulses.append(tmp)
            
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
    sil_scores = np.array(sil_scores, dtype="float")
    signal_centroids = np.array(signal_centroids, dtype="float")
    noise_centroids = np.array(noise_centroids, dtype="float")

    return MUPulses, n_updates




def rt_decomp_live(data, B_realtime, batch_size, 
                   n_updates, prev_timediff, time_diff, 
                   prev_MUPulses, 
                   mean_tm=None, discard=None, R=16, 
                   bandpass=True, lowcut=10, highcut=900, order=6, l=31, 
                   random_seed = None, 
                   classify_mu=True, thd_sil=0.85, 
                   use_pps=False, thd_pps=5, 
                   sc_tm=None, nc_tm=None, fs=2048):    
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

        signal_i, noise_i = sort_peaks_rt(s2_i=s2[i], peak_indices_i=peak_indices[i],
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
    signal_centroids = np.array(signal_centroids, dtype="float")
    noise_centroids = np.array(noise_centroids, dtype="float")

    return MUPulses, n_updates