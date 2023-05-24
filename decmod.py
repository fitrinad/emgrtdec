from emgdecompy.preprocessing import *
from emgdecompy.decomposition import *

from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import numpy as np
from functions import *



def rt_decomp(data, B_realtime, current_time, mean_tm=None, discard=None,
              bandpass=True, lowcut=10, highcut=900, order=6, l=31, 
              use_kmeans=False, thd_noise=0.38, 
              classify_mu=True, sil_dist = True, 
              thd_sil=0.85, thd_cent_dist=0.6, 
              sc_tm=None, nc_tm=None, 
              batch_size=4.0, overlap=3.0, fs=2048):
    # Source extraction
    s, _ = source_extraction(x = data, 
                             B_realtime = B_realtime, 
                             bandpass = bandpass, 
                             lowcut=lowcut, 
                             highcut=highcut, 
                             order=order, 
                             fs=fs, 
                             mean_tm = mean_tm, 
                             discard=discard)

    # Peak extraction
    s2, peak_indices = peak_extraction(s=s, l=l)
    
    # Spike classification
    if current_time == 0.0:
        MUPulses, cls_values, signal, noise, _ = spike_classification(s2=s2, 
                                                    peak_indices=peak_indices, 
                                                    use_kmeans=use_kmeans, 
                                                    thd_noise=thd_noise,
                                                    classify_mu=classify_mu, 
                                                    sil_dist = sil_dist, 
                                                    thd_sil=thd_sil, 
                                                    thd_cent_dist=thd_cent_dist,
                                                    sc_tm=sc_tm, nc_tm=nc_tm)
        sil_scores = np.tile(cls_values["sil_scores"], (s2.shape[1], 1)).T
        signal_centroids = np.tile(signal["signal_centroids"], (s2.shape[1], 1)).T
        noise_centroids = np.tile(noise["noise_centroids"], (s2.shape[1], 1)).T
    else:
        tmp = []
        MUPulses_curr, cls_values_curr, signal_curr, noise_curr, _ = spike_classification(s2=s2, 
                                                            peak_indices=peak_indices, 
                                                            use_kmeans=use_kmeans, 
                                                            thd_noise=thd_noise,
                                                            classify_mu=classify_mu, 
                                                            sil_dist = sil_dist, 
                                                            thd_sil=thd_sil, 
                                                            thd_cent_dist=thd_cent_dist,
                                                            sc_tm=sc_tm, nc_tm=nc_tm)
        MUPulses_curr = MUPulses_curr + int(current_time*fs)

        sil_scores_curr = np.tile(cls_values_curr["sil_scores"], (s2.shape[1], 1)).T
        sil_scores_curr = batch_sil(sil_scores, 
                                    sil_scores_curr, 
                                    overlap=overlap, 
                                    fs=fs)
        signal_centroids_curr = np.tile(signal_curr["signal_centroids"], (s2.shape[1], 1)).T
        signal_centroids_curr = batch_sil(signal_centroids, 
                                          signal_centroids_curr, 
                                          overlap=overlap, 
                                          fs=fs)
        noise_centroids_curr = np.tile(noise_curr["noise_centroids"], (s2.shape[1], 1)).T
        noise_centroids_curr = batch_sil(noise_centroids, 
                                         noise_centroids_curr, 
                                         overlap=overlap, 
                                         fs=fs)

        sil_scores = sil_scores_curr
        signal_centroids = signal_centroids_curr
        noise_centroids = noise_centroids_curr

        for j in range(MUPulses.shape[0]):
            add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (current_time+overlap)*fs]
            tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
            
        tmp = np.array(tmp, dtype="object")
        MUPulses_curr = tmp
        MUPulses = MUPulses_curr    

    decomp = {"MUPulses": MUPulses, "sil_scores": sil_scores, 
              "signal_centroids": signal_centroids, 
              "noise_centroids": noise_centroids}
    return decomp



#############################################################################################################################
### Decomposition module  ###################################################################################################
#############################################################################################################################
def source_extraction(x, B_realtime, mean_tm=None, discard=None, 
                      bandpass=True, lowcut=10, highcut=900, fs=2048, order=6, 
                      R=16):
    """
    Returns matrix containing source vectors estimation from the EMG signal (x).

    Args:
        x           : numpy.ndarray
            Raw EMG signal
        B_realtime  : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x

    Returns:
        s           : numpy.ndarray
            Matrix containing source vectors
        x_ext       : numpy.ndarray
            Extended EMG signal
    """    

    # Flatten signal
    if ((x[0][0].size == 0 or
         x[12][0].size == 0) and x.ndim == 2) or x.ndim == 3:    
        x = flatten_signal(x)

    # Discarding channels
    if discard is not None:
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
    Detects and extracts peaks from the each squared source vector s[i] from matrix s.

    Args:
        x_ext       : numpy.ndarray
            Extended EMG signal
        s           : numpy.ndarray
            Matrix containing source vectors
        l           : int
            Minimal horizontal distance between peaks in the peak-finding algorithm
            (default: l=31, approximately 15 ms for fs = 2048 Hz)
        
    Returns:
        peak_indices  : numpy.ndarray
            Matrix containing separation vectors for realtime decomposition
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

#############################################################################################################################
### kmeans_sil_cp  ##########################################################################################################
#############################################################################################################################
def sort_peaks(s2, peak_indices, use_kmeans=True, random_seed=None, thd_noise=0.38):
    signal_clusters = []
    signal_centroids = []
    max_signals = []
    noise_clusters = []
    noise_centroids = []
    n_signal = []
    n_noise = []
    peak_indices_signal = []
    peak_indices_noise = []

    for i in range(s2.shape[0]):
        if use_kmeans:
            # Separating large peaks from relatively small peaks (noise)
            kmeans = KMeans(n_clusters=2, random_state=random_seed)
            kmeans.fit(s2[i][peak_indices[i]].reshape(-1,1))

            # Signal cluster centroid (sc_i)
            sc_i = np.argmax(kmeans.cluster_centers_)
            # Determining which cluster contains large peaks (signal)
            signal_indices = kmeans.labels_.astype(bool) # if sc_i == 1
            if sc_i == 0:
                signal_indices = ~signal_indices      
        else:
            # noise: peaks < thd_noise*s2.max()
            signal_indices = s2[i][peak_indices[i]] > thd_noise*s2[i].max()
            # signal_indices = signal_indices.flatten()
        n_signal_idx = signal_indices.sum()
        n_noise_idx = (~signal_indices).sum()
        
        # Indices of the peaks in signal cluster
        peak_indices_signal_i = peak_indices[i][signal_indices]
        peak_indices_noise_i = peak_indices[i][~signal_indices]
        peak_indices_signal.append(peak_indices_signal_i)
        peak_indices_noise.append(peak_indices_noise_i)

        # Signal cluster and Noise cluster
        signal_cluster = s2[i][peak_indices_signal_i]
        noise_cluster = np.delete(s2[i], peak_indices_signal_i)

        # Centroids
        signal_centroid = signal_cluster.mean()
        noise_centroid = noise_cluster.mean()
        
        signal_clusters.append(signal_cluster)
        signal_centroids.append(signal_centroid)
        max_signals.append(signal_cluster.max())
        noise_clusters.append(noise_cluster)
        noise_centroids.append(noise_centroid)
        n_signal.append(n_signal_idx)
        n_noise.append(n_noise_idx)

    n_signal = np.array(n_signal, dtype="int")
    n_noise = np.array(n_noise, dtype="int")
    peak_indices_signal = np.array(peak_indices_signal, dtype="object")
    peak_indices_noise = np.array(peak_indices_noise, dtype="object")
    
    signal_centroids = np.array(signal_centroids, dtype="float")
    max_signals = np.array(max_signals, dtype="float")
    max_sc = signal_centroids.max()
    max_signal = max_signals.max()
    # denom = max_sc
    denom = 1
    signal_centroids = signal_centroids / denom
    signal_clusters = np.array(signal_clusters, dtype="object")
    signal_clusters = signal_clusters / denom
    max_signals = max_signals / denom
    
    noise_clusters = np.array(noise_clusters, dtype="object")
    noise_clusters = noise_clusters / denom
    noise_centroids = np.array(noise_centroids, dtype="float")
    noise_centroids = noise_centroids / denom
    
    # Distance between centroids
    centroid_dists = signal_centroids - noise_centroids 
    s2 = s2 / denom

    signal = {"n_signal": n_signal, 
            "peak_indices_signal": peak_indices_signal, 
            "signal_clusters": signal_clusters,
            "signal_centroids": signal_centroids}
    noise = {"n_noise": n_noise, 
            "peak_indices_noise": peak_indices_noise, 
            "noise_clusters": noise_clusters,
            "noise_centroids": noise_centroids}

    return signal, noise, centroid_dists, s2, max_sc



def spike_classification(s2, peak_indices,
                         use_kmeans=True, random_seed=None, thd_noise=0.38, 
                         classify_mu=True, sil_dist=False, 
                         thd_sil=0.9, thd_cent_dist=0.6,
                         sc_tm=None, nc_tm=None):
    """
    Returns a matrix of motor unit pulses.

    Args:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
        use_kmeans      : bool
            Separates large peaks from small peaks using kmeans clustering if True
        random_seed     : int
            Used to initialize the pseudo-random processes in the function
        thd_sil         : float
            Threshold of the silhouette score
        thd_cent_dist   : float
            Threshold of the centroid distance
        sil_dist        : bool
            Classifies peaks as motor unit pulses according to the silhouette score (if True) 
            or peak distance ratio (if False) 
    
    Returns:
        MUPulses        : numpy.ndarray
            Matrix containing indices of motor unit pulses
    """
    
    signal, noise, centroid_dists, s2, _ = sort_peaks(s2=s2, peak_indices=peak_indices, 
                                                           use_kmeans=use_kmeans, random_seed=random_seed, 
                                                           thd_noise=thd_noise)

    peak_indices_signal = signal["peak_indices_signal"]
    MUPulses = []
    sil_scores = []
    cent_dists = []
    for i in range(s2.shape[0]):
        add_peaks = peak_indices_signal[i]
        if classify_mu: # whether MU i is firing or not, based on SIL or centroid distance
            add_peaks = peak_indices_signal[i]
            if sil_dist:
                # Silhouette score
                sil = silhouette_score(s2[i], peak_indices_signal[i])
                sil_scores.append(sil)
                if sil < thd_sil:
                    add_peaks = []
            else:
                cent_dist = centroid_dists[i] 
                cent_dists.append(cent_dist)
                if cent_dist < thd_cent_dist:
                    add_peaks = []

        # Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        # adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            # s2_max = s2[i][add_peaks].max()
            # add_indices = ( abs(s2[i][add_peaks]/s2_max - nc_tm[i]) > 
            #                abs(s2[i][add_peaks]/s2_max - sc_tm[i]) )
            
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                            abs(s2[i][add_peaks] - (sc_tm[i])) )
            add_peaks = add_peaks[add_indices] 
        
        MUPulses.append(np.array(add_peaks, dtype="int64"))
    
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
        
    cls_values = {"centroid_dists": np.array(cent_dists, dtype="float"),
                  "sil_scores": np.array(sil_scores, dtype="float")}

    return MUPulses, cls_values, signal, noise, s2


def batch_sil(sil_scores_prev, sil_scores, overlap=1.0, fs=2048):
    n_mu = sil_scores_prev.shape[0]
    end_1 = sil_scores_prev.shape[1]
    end_2 = sil_scores.shape[1]
    ol = int(overlap * fs)
    curr_size = end_1 + end_2 - ol
    if end_2 < ol:
        curr_size = end_1
    sil_scores_curr = np.zeros((n_mu, curr_size), dtype="float")
    
    sil_scores_curr[:, :end_1] = sil_scores_prev
    if end_2 <= ol:
        sil_scores_curr[:, end_1-end_2:curr_size] = sil_scores_prev[:, end_1-end_2:end_1]
    else:
        sil_scores_curr[:, end_1:curr_size] = sil_scores[:, ol:end_2]
        """if ol!=0:
            for i in range(n_mu):
                tmp = np.arange(sil_scores_prev[i][end_1-1-ol],
                                sil_scores[i][ol],
                                (sil_scores[i][ol]-sil_scores_prev[i][end_1-1-ol])/ol)
                sil_scores_curr[i, end_1-ol:end_1] = tmp"""
    return sil_scores_curr


def batch_decomp(data, B_realtime, mean_tm=None, discard=None, 
                 bandpass=True, lowcut=10, highcut=900, order=6, l=31,
                 use_kmeans=False, thd_noise=0.38,
                 classify_mu=True, sil_dist = True,
                 thd_sil=0.9, thd_cent_dist=0.6,
                 sc_tm=None, nc_tm=None,
                 batch_size=4.0, overlap=0.0, fs=2048):
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
                                 bandpass = bandpass,
                                 lowcut=lowcut,
                                 highcut=highcut,
                                 order=order,
                                 fs=fs,
                                 mean_tm = mean_tm,
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, cls_values, signal, noise, _ = spike_classification(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            sil_scores = np.tile(cls_values["sil_scores"], (s2.shape[1], 1)).T
            signal_centroids = np.tile(signal["signal_centroids"], (s2.shape[1], 1)).T
            noise_centroids = np.tile(noise["noise_centroids"], (s2.shape[1], 1)).T
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, cls_values_curr, signal_curr, noise_curr, _ = spike_classification(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
            MUPulses_curr = MUPulses_curr + int(time*fs)

            sil_scores_curr = np.tile(cls_values_curr["sil_scores"], (s2.shape[1], 1)).T
            sil_scores_curr = batch_sil(sil_scores, 
                                        sil_scores_curr, 
                                        overlap=overlap, 
                                        fs=fs)
            signal_centroids_curr = np.tile(signal_curr["signal_centroids"], (s2.shape[1], 1)).T
            signal_centroids_curr = batch_sil(signal_centroids, 
                                                signal_centroids_curr, 
                                                overlap=overlap, 
                                                fs=fs)
            noise_centroids_curr = np.tile(noise_curr["noise_centroids"], (s2.shape[1], 1)).T
            noise_centroids_curr = batch_sil(noise_centroids, 
                                                noise_centroids_curr, 
                                                overlap=overlap, 
                                                fs=fs)

            sil_scores = sil_scores_curr
            signal_centroids = signal_centroids_curr
            noise_centroids = noise_centroids_curr

            for j in range(MUPulses.shape[0]):
                add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (time+overlap)*fs]
                tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
                
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time-overlap:
            break

    decomp = {"MUPulses": MUPulses, "sil_scores": sil_scores, 
              "signal_centroids": signal_centroids, 
              "noise_centroids": noise_centroids}
    return decomp



def batch_decomp_window(data, B_realtime, mean_tm=None, discard=None, 
                        bandpass=True, lowcut=10, highcut=900, order=6, l=31,
                        use_kmeans=False, thd_noise=0.38,
                        classify_mu=True, sil_dist = True,
                        thd_sil=0.9, thd_cent_dist=0.6,
                        thd_pps=5,
                        sc_tm=None, nc_tm=None,
                        batch_size=0.6, overlap=0.3, fs=2048):
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
                                 bandpass = bandpass,
                                 lowcut=lowcut,
                                 highcut=highcut,
                                 order=order,
                                 fs=fs,
                                 mean_tm = mean_tm,
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, cls_values, _, _, _ = spike_classification(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            sil_scores = np.tile(cls_values["sil_scores"], (s2.shape[1], 1)).T
            thd_spikes = int(thd_pps * s2.shape[1]/fs)
                
            for i in range(MUPulses.shape[0]):
                # number of spikes in current window
                n_spikes_curr_i = MUPulses[i].shape[0]
                if (n_spikes_curr_i < thd_spikes):
                    MUPulses[i] = np.array([], dtype="int64")
                
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, cls_values_curr, _, _, _ = spike_classification(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
             
            MUPulses_curr = MUPulses_curr + int(time*fs)

            sil_scores_curr = np.tile(cls_values_curr["sil_scores"], (s2.shape[1], 1)).T
            sil_scores_curr = batch_sil(sil_scores, 
                                        sil_scores_curr, 
                                        overlap=overlap, 
                                        fs=fs)
            sil_scores = sil_scores_curr
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

    decomp = {"MUPulses": MUPulses, "sil_scores": sil_scores}
    return decomp


#############################################################################################################################
### kmeans_cp_sil  ##########################################################################################################
#############################################################################################################################
def sort_peaks2(s2, peak_indices, use_kmeans=True, random_seed=None, thd_noise=0.38):
    signal_clusters = []
    signal_centroids = []
    noise_clusters = []
    noise_centroids = []
    n_signal = []
    n_noise = []
    peak_indices_signal = []
    peak_indices_noise = []

    for i in range(s2.shape[0]):
        if use_kmeans:
            # Separating large peaks from relatively small peaks (noise)
            kmeans = KMeans(n_clusters=2, random_state=random_seed)
            kmeans.fit(s2[i][peak_indices[i]].reshape(-1,1))

            # Signal cluster centroid (sc_i)
            sc_i = np.argmax(kmeans.cluster_centers_)
            # Determining which cluster contains large peaks (signal)
            signal_indices = kmeans.labels_.astype(bool) # if sc_i == 1
            if sc_i == 0:
                signal_indices = ~signal_indices      
        else:
            # noise: peaks < thd_noise*s2.max()
            signal_indices = s2[i][peak_indices[i]] > thd_noise*s2[i].max()
            # signal_indices = signal_indices.flatten()
        n_signal_idx = signal_indices.sum()
        n_noise_idx = (~signal_indices).sum()
        
        # Indices of the peaks in signal cluster
        peak_indices_signal_i = peak_indices[i][signal_indices]
        peak_indices_noise_i = peak_indices[i][~signal_indices]
        peak_indices_signal.append(peak_indices_signal_i)
        peak_indices_noise.append(peak_indices_noise_i)

        # Signal cluster and Noise cluster
        signal_cluster = s2[i][peak_indices_signal_i]
        noise_cluster = np.delete(s2[i], peak_indices_signal_i)

        # Centroids
        signal_centroid = signal_cluster.mean()
        noise_centroid = noise_cluster.mean()
        
        signal_clusters.append(signal_cluster)
        signal_centroids.append(signal_centroid)
        noise_clusters.append(noise_cluster)
        noise_centroids.append(noise_centroid)
        n_signal.append(n_signal_idx)
        n_noise.append(n_noise_idx)

    n_signal = np.array(n_signal, dtype="int")
    n_noise = np.array(n_noise, dtype="int")
    peak_indices_signal = np.array(peak_indices_signal, dtype="object")
    peak_indices_noise = np.array(peak_indices_noise, dtype="object")
    
    signal_centroids = np.array(signal_centroids, dtype="float")
    signal_clusters = np.array(signal_clusters, dtype="object")
    
    noise_clusters = np.array(noise_clusters, dtype="object")
    noise_centroids = np.array(noise_centroids, dtype="float")
    
    # Distance between centroids
    centroid_dists = signal_centroids - noise_centroids 

    signal = {"n_signal": n_signal, 
            "peak_indices_signal": peak_indices_signal, 
            "signal_clusters": signal_clusters,
            "signal_centroids": signal_centroids}
    noise = {"n_noise": n_noise, 
            "peak_indices_noise": peak_indices_noise, 
            "noise_clusters": noise_clusters,
            "noise_centroids": noise_centroids}

    return signal, noise, centroid_dists, s2


def spike_classification2(s2, peak_indices,
                         use_kmeans=True, random_seed=None, thd_noise=0.38, 
                         classify_mu=True, sil_dist=False, 
                         thd_sil=0.9, thd_cent_dist=0.6,
                         sc_tm=None, nc_tm=None):
    """
    Returns a matrix of motor unit pulses.

    Args:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
        use_kmeans      : bool
            Separates large peaks from small peaks using kmeans clustering if True
        random_seed     : int
            Used to initialize the pseudo-random processes in the function
        thd_sil         : float
            Threshold of the silhouette score
        thd_cent_dist   : float
            Threshold of the centroid distance
        sil_dist        : bool
            Classifies peaks as motor unit pulses according to the silhouette score (if True) 
            or peak distance ratio (if False) 
    
    Returns:
        MUPulses        : numpy.ndarray
            Matrix containing indices of motor unit pulses
    """
    signal, noise, centroid_dists, s2, _ = sort_peaks(s2=s2, peak_indices=peak_indices, 
                                                           use_kmeans=use_kmeans, random_seed=random_seed, 
                                                           thd_noise=thd_noise)

    peak_indices_signal = signal["peak_indices_signal"]
    MUPulses = []
    sil_scores = []
    cent_dists = []
    for i in range(s2.shape[0]):
        add_peaks = peak_indices_signal[i]
        
        # Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        # adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            # s2_max = s2[i][add_peaks].max()
            # add_indices = ( abs(s2[i][add_peaks]/s2_max - nc_tm[i]) > 
            #                abs(s2[i][add_peaks]/s2_max - sc_tm[i]) )
            # add_peaks = add_peaks[add_indices] 
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i]/sc_tm.max())) > 
                            abs(s2[i][add_peaks] - (sc_tm[i]/sc_tm.max())) )
            add_peaks = add_peaks[add_indices] 
        
        if classify_mu: # whether MU i is firing or not, based on SIL or centroid distance
            if sil_dist:
                # Silhouette score
                sil = silhouette_score(s2[i], peak_indices_signal[i])
                sil_scores.append(sil)
                if sil < thd_sil:
                    add_peaks = []
            else:
                cent_dist = centroid_dists[i] 
                cent_dists.append(cent_dist)
                if cent_dist < thd_cent_dist:
                    add_peaks = []
        
        MUPulses.append(np.array(add_peaks, dtype="int64"))
    
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
        
    cls_values = {"centroid_dists": np.array(cent_dists, dtype="float"),
                  "sil_scores": np.array(sil_scores, dtype="float")}

    return MUPulses, cls_values, signal, noise, s2



def batch_decomp2(data, B_realtime, mean_tm=None, discard=None, l=31,
                 use_kmeans=False, thd_noise=0.38,
                 classify_mu=True, sil_dist = True,
                 thd_sil=0.9, thd_cent_dist=0.6,
                 sc_tm=None, nc_tm=None,
                 batch_size=4.0, overlap=0.0, fs=2048):
    if (data[0][0].size == 0 and data.ndim == 2) or data.ndim == 3:
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
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, _, _, _, _ = spike_classification2(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, _, _, _, _ = spike_classification2(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
            MUPulses_curr = MUPulses_curr + int(time*fs)

            for j in range(MUPulses.shape[0]):
                add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (time+overlap)*fs]
                tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
                
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time:
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

