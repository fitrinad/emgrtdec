from sklearn.cluster import KMeans
import numpy as np

def sort_peaks_rt(s2_i, peak_indices_i, random_seed=None):
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