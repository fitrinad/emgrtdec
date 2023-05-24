from emgdecompy.decomposition import *
from scipy import linalg
import numpy as np

#############################################################################################################################
### Training module  ########################################################################################################
#############################################################################################################################
"""
Decomposition functions for the training module from:
https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
"""
def silhouette_score_tmod(s_i, peak_indices):
    """
    Silhouette score function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Calculates silhouette score on the estimated source.

    Defined as the difference between within-cluster sums of point-to-centroid distances
    and between-cluster sums of point-to-centroid distances.
    Measure is normalized by dividing by the maximum of these two values (Negro et al. 2016).

    Parameters
    ----------
        s_i: numpy.ndarray
            Estimated source. 1D array containing K elements, where K is the number of samples.
        peak_indices_a: numpy.ndarray
            1D array containing the peak indices.

    Returns
    -------
        sil: float
            Silhouette score.
        peak_centroid: float
            Peak centroid value of estimated source.
        noise_centroid: float
            Noise centroid value of estimated source.
    """
    # Create clusters
    peak_cluster = s_i[peak_indices]
    noise_cluster = np.delete(s_i, peak_indices)

    # Create centroids
    peak_centroid = peak_cluster.mean()
    noise_centroid = noise_cluster.mean()

    # Calculate within-cluster sums of point-to-centroid distances
    intra_sums = (
        abs(peak_cluster - peak_centroid).sum()
        + abs(noise_cluster - noise_centroid).sum()
    )

    # Calculate between-cluster sums of point-to-centroid distances
    inter_sums = (
        abs(peak_cluster - noise_centroid).sum()
        + abs(noise_cluster - peak_centroid).sum()
    )

    diff = inter_sums - intra_sums

    sil = diff / max(intra_sums, inter_sums)

    return sil, peak_centroid, noise_centroid



def refinement_tmod(
    w_i, z, i, l=31, sil_pnr=True, thresh=0.9, max_iter=10, random_seed=None, verbose=False
):
    """
    Refinement function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Refines the estimated separation vectors determined by the `separation` function
    as described in Negro et al. (2016). Uses a peak-finding algorithm combined
    with K-Means clustering to determine the motor unit spike train. Updates the 
    estimated separation vector accordingly until regularity of the spike train is
    maximized. Steps 4, 5, and 6 in Negro et al. (2016).

    Parameters
    ----------
        w_i: numpy.ndarray
            Current separation vector to refine.
        z: numpy.ndarray
            Centred, extended, and whitened EMG data.
        i: int
            Decomposition iteration number.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
           If true, refinement information is printed.

    Returns
    -------
        w_i: numpy.ndarray
            Separation vector if SIL/PNR is above threshold.
            Otherwise return empty vector.
        s_i: numpy.ndarray
            Estimated source obtained from dot product of separation vector and z.
            Empty array if separation vector not accepted.
        peak_indices_a: numpy.ndarray
            Peak indices for peaks in cluster "a" of the squared estimated source.
            Empty array if separation vector not accepted.
        sil: float
            Silhouette score if SIL/PNR is above threshold.
            Otherwise return 0.
        pnr_score: float
            Pulse-to-noise ratio if SIL/PNR is above threshold.
            Otherwise return 0.
        peak_centroid: float
            Peak centroid value of estimated source.
        noise_centroid: float
            Noise centroid value of estimated source.
    """
    cv_curr = np.inf # Set it to inf so there isn't a chance the loop breaks too early

    for iter in range(max_iter):
        
        w_i = normalize(w_i) # Normalize separation vector

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i2 = np.square(s_i)

        # Peak-finding algorithm
        peak_indices, _ = find_peaks(
            s_i2, distance=l
        )

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s_i2[peak_indices].reshape(-1, 1))
        
        # Determine which cluster contains large peaks
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )
        
        # Determine which peaks are large (part of cluster a)
        peak_a = ~kmeans.labels_.astype(
            bool
        )

        if centroid_a == 1: # If cluster a corresponds to kmeans label 1, change indices correspondingly
            peak_a = ~peak_a

        
        # Get the indices of the peaks in cluster a
        peak_indices_a = peak_indices[
            peak_a
        ]

        # c. Update inter-spike interval coefficients of variation
        isi = np.diff(peak_indices_a)  # inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)

        if np.isnan(cv_curr): # Translate nan to 0
            cv_curr = 0

        if (
            cv_curr > cv_prev
        ):
            break
            
        elif iter != max_iter - 1: # If we are not on the last iteration
            # d. Update separation vector for next iteration unless refinement doesn't converge
            j = len(peak_indices_a)
            w_i = (1 / j) * z[:, peak_indices_a].sum(axis=1)

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil, peak_centroid, noise_centroid = silhouette_score_tmod(
        s_i2, peak_indices_a
    )
    pnr_score = pnr(s_i2, peak_indices_a)
    
    if isi.size > 0 and verbose:
        print(f"Cov(ISI): {cv_curr / isi.mean() * 100}")

    if verbose:
        print(f"PNR: {pnr_score}")
        print(f"SIL: {sil}")
        print(f"cv_curr = {cv_curr}")
        print(f"cv_prev = {cv_prev}")
        
        if cv_curr > cv_prev:
            print(f"Refinement converged after {iter} iterations.")

    if sil_pnr:
        score = sil # If using SIL as acceptance criterion
    else:
        score = pnr_score # If using PNR as acceptance criterion
    
    # Don't accept if score is below threshold or refinement doesn't converge
    if score < thresh or cv_curr < cv_prev or cv_curr == 0: 
        w_i = np.zeros_like(w_i) # If below threshold, reject estimated source and return nothing
        return w_i, np.zeros_like(s_i), np.array([]), 0, 0, 0, 0
    else:
        print(f"Extracted source at iteration {i}.")
        return w_i, s_i, peak_indices_a, sil, pnr_score, peak_centroid, noise_centroid


def decomposition_tmod(
    x,
    discard=None,
    R=16,
    M=64,
    bandpass=True,
    lowcut=10,
    highcut = 900,
    fs=2048,
    order=6,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter_sep=10,
    l=31,
    sil_pnr=True,
    thresh=0.9,
    max_iter_ref=10,
    random_seed=None,
    verbose=False
):
    """
    Decomposition function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Blind source separation algorithm that utilizes the functions
    in EMGdecomPy to decompose raw EMG data. Runs data pre-processing, separation,
    and refinement steps to extract individual motor unit activity from EMG data. 
    Runs steps 1 through 6 in Negro et al. (2016).

    Parameters
    ----------
        x: numpy.ndarray
            Raw EMG signal.
        discard: slice, int, or array of ints
            Indices of channels to discard.
        R: int
            How far to extend x.
        M: int
            Number of iterations to run decomposition for.
        bandpass: bool
            Whether to band-pass filter the raw EMG signal or not.
        lowcut: float
            Lower range of band-pass filter.
        highcut: float
            Upper range of band-pass filter.
        fs: float
            Sampling frequency in Hz.
        order: int
            Order of band-pass filter. 
        Tolx: float
            Tolerance for element-wise comparison in separation.
        contrast_fun: function
            Contrast function to use.
            skew, og_cosh or exp_sq
        ortho_fun: function
            Orthogonalization function to use.
            gram_schmidt or deflate
        max_iter_sep: int > 0
            Maximum iterations for fixed point algorithm.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter_ref: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
            If true, decomposition information is printed.

    Returns
    -------
        decomp_results: dict
            Dictionary containing:
                B: numpy.ndarray
                    Matrix whose columns contain the accepted separation vectors.
                MUPulses: numpy.ndarray
                    Firing indices for each motor unit.
                SIL: numpy.ndarray
                    Corresponding silhouette scores for each accepted source.
                PNR: numpy.ndarray
                    Corresponding pulse-to-noise ratio for each accepted source.
                peak_centroids: numpy.ndarray
                    Peak centroid value for each accepted source.
                noise_centroids: numpy.ndarray
                    Noise centroid value for each accepted source.
                s: numpy.ndarray
                    Estimated source.
                discarded_channels: numpy.ndarray
                    Array of discarded channels.  
    """

    # Flatten
    if ((x[0][0].size == 0 or
         x[12][0].size == 0) and x.ndim == 2) or x.ndim == 3:
        x_flt = flatten_signal(x)
    else:
        x_flt = x
    # Discard unwanted channels
    if discard is not None:
        x_flt = np.delete(x_flt, discard, axis=0)

    # Apply band-pass filter
    if bandpass:
        x_flt = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x_flt,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)

    # Center
    x_flt = center_matrix(x_flt)

    print("Centred.")

    # Extend
    x_ext = extend_all_channels(x_flt, R)

    print("Extended.")

    # Whiten
    z = whiten(x_ext)

    print("Whitened.")

    decomp_results = {}  # Create output dictionary

    B = np.zeros((z.shape[0], z.shape[0]))  # Initialize separation matrix
    
    z_peak_indices, z_peak_heights = initial_w_matrix(z)  # Find highest activity columns in z
    z_peaks = z[:, z_peak_indices] # Index the highest activity columns in z

    MUPulses = []
    sils = []
    pnrs = []
    peak_centroids = []
    noise_centroids = []
    s = []


    for i in range(M):

        z_highest_peak = (
            z_peak_heights.argmax()
        )  # Determine which column of z has the highest activity

        w_init = z_peaks[
            :, z_highest_peak
        ]  # Initialize the separation vector with this column

        if verbose and (i + 1) % 10 == 0:
            print(i)

        # Separate
        w_i = separation(
            z, w_init, B, Tolx, contrast_fun, ortho_fun, max_iter_sep, verbose
        )

        # Refine
        w_i, s_i, mu_peak_indices, sil, pnr_score, peak_centroid, noise_centroid = refinement_tmod(
            w_i, z, i, l, sil_pnr, thresh, max_iter_ref, random_seed, verbose
        )
    
        B[:, i] = w_i # Update i-th column of separation matrix

        if mu_peak_indices.size > 0:  # Only save information for accepted vectors
            MUPulses.append(np.array(mu_peak_indices, dtype="int64"))
            sils.append(sil)
            pnrs.append(pnr_score)
            peak_centroids.append(peak_centroid)
            noise_centroids.append(noise_centroid)
            s.append(s_i)

        # Update initialization matrix for next iteration
        z_peaks = np.delete(z_peaks, z_highest_peak, axis=1)
        z_peak_heights = np.delete(z_peak_heights, z_highest_peak)
    
    decomp_results["B"] = B[:, B.any(0)] # Only save columns of B that have accepted vectors
    if len(MUPulses) > 1:
        decomp_results["MUPulses"] = np.array(MUPulses, dtype="object")
    else:
        decomp_results["MUPulses"] = np.array(MUPulses, dtype="int64")
    decomp_results["SIL"] = np.array(sils)
    decomp_results["PNR"] = np.array(pnrs)
    decomp_results["peak_centroids"] = np.array(peak_centroids)
    decomp_results["noise_centroids"] = np.array(noise_centroids)
    decomp_results["s"] = np.array(s)
    decomp_results["discarded_channels"] = np.array(discard)
    decomp_results["SIG"] = x
    decomp_results["fsamp"] = fs
    decomp_results["ext_factor"] = R

    return decomp_results



def sep_realtime(x, B, discard=None, center=True, 
                 bandpass=True, lowcut=10, highcut=900, fs=2048, order=6, 
                 R=16):
    """
    Returns matrix containing separation vectors for realtime decomposition.

    Args:
        x           : numpy.ndarray
            Raw EMG signal
        B           : numpy.ndarray
            Matrix containing separation vectors from training module
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x

    Returns:
        B_realtime  : numpy.ndarray
            Separation matrix for realtime decomposition
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

    # Center
    if center: 
        x_cent = center_matrix(x)
    else:
        x_cent = x

    # Extend
    x_ext = extend_all_channels(x_cent, R)

    # Whitening Matrix: wzca
    #   Calculate covariance matrix
    cov_mat = np.cov(x_ext, rowvar=True, bias=True)
    #   Eigenvalues and eigenvectors
    w, v = linalg.eig(cov_mat)
    #   Apply regularization factor, replacing eigenvalues smaller than it with the factor
    reg_factor = w[round(len(w) / 2):].mean()
    w = np.where(w < reg_factor, reg_factor, w)
    #   Diagonal matrix inverse square root of eigenvalues
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real
    #   Whitening using zero component analysis: v diagw v.T x
    wzca = np.dot(v, np.dot(diagw, v.T))

    # 1. Realtime separation matrix: 
    #    B_realtime = wzca . B
    B_realtime = np.dot(wzca, B)
    #   Normalized separation matrix
    for i in range(B_realtime.shape[0]):
        B_realtime[i] = normalize(B_realtime[i])

    # 2. Mean of training data
    x_ext_tm = extend_all_channels(x, R=R)
    mean_tm = x_ext_tm.mean(axis=1)

    # 3. Signal and noise centroids
    #    normalized signal and noise centroids (centroid / max(signal_centroids_tm))
    
    return B_realtime, mean_tm


