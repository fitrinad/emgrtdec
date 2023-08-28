from emgdecompy.decomposition import *
from functions.preprocessing import apply_filter
from functions.analysis import unique_mu
from scipy.signal import butter, iirnotch
from scipy import linalg
import numpy as np

#############################################################################################################################
### Training module  ########################################################################################################
#############################################################################################################################
"""
Decomposition functions for the training module are from:
https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
"""
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
    use_notch_filter=True,
    notch_cutoffs=[50],
    q_factor=30,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter_sep=10,
    l=31,
    sil_pnr=True,
    thresh=0.9,
    max_iter_ref=10,
    random_seed=None,
    verbose=False,
    similarity_thd = 0.9
):
    """
    Decomposition function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    with some changes to adjust to the data structure of the input.
    
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
                B               : numpy.ndarray
                    Matrix whose columns contain the accepted separation vectors
                MUPulses        : numpy.ndarray
                    Firing indices for each motor unit
                SIL             : numpy.ndarray
                    Corresponding silhouette scores for each accepted source
                PNR             : numpy.ndarray
                    Corresponding pulse-to-noise ratio for each accepted source
                s               : numpy.ndarray
                    Estimated source
                discard_ch      : numpy.ndarray
                    Array of discarded channels
                SIG             : numpy.ndarray
                    Raw EMG signal
                fsamp           : float
                    Sampling frequency (Hz)
                ext_factor      : int
                    Extension factor
                min_distance    : int
                    Required minimal horizontal distance between peaks in peak-finding algorithm
                unique_mu       : list
                    List of unique extracted motor units
    """
    # Flattening data, excluding the empty channel, if the channels are in a grid of (13, 5)
    if ((x[0][0].size == 0 or
         x[12][0].size == 0) and x.ndim == 2) or x.ndim == 3:
        x_flt = flatten_signal(x)
    else:
        x_flt = x
    
    # Discarding unwanted channels
    if np.all(discard) is not None:
        x_flt = np.delete(x_flt, discard, axis=0)


    # Applying band-pass filter
    if bandpass:
        bpf_num, bpf_den = butter(order, [lowcut, highcut], fs=fs, btype="band")
        x_flt = np.apply_along_axis(apply_filter, 
                                    axis=1, 
                                    arr=x_flt, 
                                    b=bpf_num, 
                                    a=bpf_den)
    
    # Applying notch filter
    if use_notch_filter:
        for notch_cutoff in np.array(notch_cutoffs):
            notch_num, notch_den = iirnotch(w0=notch_cutoff, Q=q_factor, fs=fs)
            x_flt = np.apply_along_axis(apply_filter, 
                                        axis=1, 
                                        arr=x_flt, 
                                        b=notch_num, 
                                        a=notch_den)

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
        w_i, s_i, mu_peak_indices, sil, pnr_score = refinement(
            w_i, z, i, l, sil_pnr, thresh, max_iter_ref, random_seed, verbose
        )
    
        B[:, i] = w_i # Update i-th column of separation matrix

        if mu_peak_indices.size > 0:  # Only save information for accepted vectors
            MUPulses.append(np.array(mu_peak_indices, dtype="int64"))
            sils.append(sil)
            pnrs.append(pnr_score)
            s.append(s_i)

        # Update initialization matrix for next iteration
        z_peaks = np.delete(z_peaks, z_highest_peak, axis=1)
        z_peak_heights = np.delete(z_peak_heights, z_highest_peak)
    
    

    decomp_results["B"] = B[:, B.any(0)] # Only save columns of B that have accepted vectors
    if len(MUPulses) > 1:
        length = len(MUPulses[0])
        if any(len(arr) != length for arr in MUPulses):
            decomp_results["MUPulses"] = np.array(MUPulses, dtype="object")
        else:
            decomp_results["MUPulses"] = np.array(MUPulses, dtype="int64")
    else:
        decomp_results["MUPulses"] = np.array(MUPulses, dtype="int64")
    decomp_results["SIL"] = np.array(sils, dtype="float")
    decomp_results["PNR"] = np.array(pnrs, dtype="float")
    decomp_results["s"] = np.array(s, dtype="float")
    if np.all(discard) is not None:
        decomp_results["discard_ch"] = np.array(discard, dtype="int")
    else:
        decomp_results["discard_ch"] = discard
    decomp_results["SIG"] = x
    decomp_results["fsamp"] = fs
    decomp_results["ext_factor"] = R
    decomp_results["min_distance"] = l
    decomp_results["n_iter"] = M
    decomp_results["use_bpf"] = bandpass
    decomp_results["lowcut_freq"] = lowcut
    decomp_results["highcut_freq"] = highcut
    decomp_results["order"] = order
    decomp_results["use_notch_filter"] = use_notch_filter 
    decomp_results["notch_cutoffs"] = notch_cutoffs
    decomp_results["use_sil"] = sil_pnr
    decomp_results["sil_threshold"] = thresh
    decomp_results["max_sep_iter"] = max_iter_sep
    decomp_results["x_tolerance"] = Tolx
    decomp_results["max_ref_iter"] = max_iter_ref
    
    # Checking for unique extracted motor units
    if decomp_results["MUPulses"].size != 0:
        unique_mus = unique_mu(ind_pt = decomp_results["MUPulses"], 
                              data = x, 
                              min_distance = l, 
                              thr=similarity_thd)
    else:
        unique_mus = []

    decomp_results["unique_mu"] = unique_mus
    
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

    # Center
    if center: 
        x_cent = center_matrix(x)
    else:
        x_cent = x

    # Extend
    x_ext = extend_all_channels(x_cent, R)

    # Whitening Matrix: wzca
    #   Calculating covariance matrix
    cov_mat = np.cov(x_ext, rowvar=True, bias=True)
    #   Eigenvalues and eigenvectors
    w, v = linalg.eig(cov_mat)
    #   Applying regularization factor, replacing eigenvalues smaller than it with the factor
    reg_factor = w[round(len(w) / 2):].mean()
    w = np.where(w < reg_factor, reg_factor, w)
    #   Diagonal matrix inverse square root of eigenvalues
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real
    #   Whitening using zero component analysis: v diagw v.T x
    wzca = np.dot(v, np.dot(diagw, v.T))

    # 1. Realtime separation matrix: 
    #   B_realtime = wzca . B
    B_realtime = np.dot(wzca, B)
    #   Normalized separation matrix
    for i in range(B_realtime.shape[0]):
        B_realtime[i] = normalize(B_realtime[i])

    # 2. Mean of training data
    x_ext_tm = extend_all_channels(x, R=R)
    mean_tm = x_ext_tm.mean(axis=1)

    return B_realtime, mean_tm