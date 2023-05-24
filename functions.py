from scipy.signal import butter, lfilter
from scipy import interpolate
import numpy as np
from emgdecompy.preprocessing import flatten_signal
import matplotlib.pyplot as plt
import copy 


def interpolate_1d(data, data_ref):
    """
    Interpolate 1D array data to the size of 1d-array data_ref
    
    Args:
    	data	        : numpy.ndarray
            1D array containing data to be interpolated
        data_ref        : numpy.ndarray
            1D array being the size reference of the resulting interpolation
        
    Returns:
        data_result   : numpy.ndarray
            1D array containing data after interpolated to fit the size of data_ref
    """
    data_interp = interpolate.interp1d(np.arange(data.size), data)
    data_result = data_interp(np.linspace(0, data.size-1, data_ref.size))

    return data_result


def calc_snr(data, noise_start=0.0, noise_end=3.0, fs=2048):
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    n_x = data.shape[1]
    n_ch = data.shape[0]
    snr = np.zeros(n_ch, dtype="float")
    
    for i in range(n_ch):
        noise = data[i][int(noise_start*fs):int(noise_end*fs)]
        snr[i] = np.square(data[i]).mean() / np.square(noise).mean()
    return snr  


def bad_channels(data, signal="signal", thd_snr=2.0, noise_start=0.0, noise_end=3.0, fs=2048):
    snr = calc_snr(data, noise_start=noise_start, noise_end=noise_end, fs=fs)
    bad_ch = np.argwhere(snr < thd_snr)
    bad_ch = bad_ch.flatten()
    print(f"Bad channels in {signal}: {bad_ch}")
    return bad_ch


def discard_ch_flat(discard_ch_vec, data):
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = data.flatten()
    empty_ch = [channel for channel, signal in enumerate(data) if 0 in signal.shape]
    if discard_ch_vec.ndim == 2 :
        discard_ch_vec = discard_ch_vec.flatten()
    disc_ch_flat = np.delete(discard_ch_vec, empty_ch, axis=0)
    disc_ch_flat = np.argwhere(disc_ch_flat == 1).squeeze()
    return disc_ch_flat


def plot_signal(data, fs=2048, title="EMG signal"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    n_x = data.shape[1]
    n_ch = data.shape[0]
    
    # Creating subplot
    n_rows = n_ch
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 2.5*n_rows]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting data
    x_axis = np.arange(0, n_x, dtype="float")
    time_axis = x_axis / float(fs)
    plt.rc('xtick', labelsize=font_medium)
    plt.rc('ytick', labelsize=font_medium)
    ax[0].set_title(title, fontsize=font_large)

    for i in range(0, n_rows):
        ax[i].plot(time_axis, data[i])
        ax[i].set_ylabel(f"Ch {i}", fontsize=font_medium)
    plt.show()


def plot_sil(sil_scores, thd_sil=None, fs=2048, title="SIL scores"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    x = sil_scores.shape[1]
    n_mu = sil_scores.shape[0]
    
    # Creating subplot
    n_rows = n_mu
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 2.5*n_rows]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting data
    x_axis = np.arange(0, x, dtype="float")
    time_axis = x_axis / float(fs)
    plt.rc('xtick', labelsize=font_large)
    plt.rc('ytick', labelsize=font_large)
    ax[0].set_title(title, fontsize=font_large)

    if thd_sil != None:
        thd_line = np.repeat(thd_sil, x)
        for i in range(0, n_rows):
            ax[i].plot(time_axis, sil_scores[i])
            ax[i].plot(time_axis, thd_line)
            ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        plt.show()
    else:
        for i in range(0, n_rows):
            ax[i].plot(time_axis, sil_scores[i])
            ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        plt.show()


def env_data(data,
               fs=2048, 
               order=4, 
               l_bpf=40, 
               h_bpf=900, 
               lpf_cut=1):
    """
    Data is band-pass filtered, rectified, and low-pass filtered,
    resulting in a rectified envelope of the signal.
    
    Args:
    	data	        : numpy.ndarray
            Array containing EMG data to be processed
        fs		        : float
            sampling frequency (Hz)
        order	        : int
            order of filter
        l_bpf	        : float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	        : float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	        : float
            cutoff frequency of the low-pass filter (Hz)
    
    Returns:
        envelope_data   : numpy.ndarray
            1D array containing rectified envelope of the EMG data, averaged
    
    Example:
        env_gl_10 = env_data(gl_10)
    """
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
        n_ch = flatten_signal(data).shape[0]
        data_flt = flatten_signal(data)
    else:
        n_x = data.shape[1]
        n_ch = data.shape[0]
        data_flt = data
    
    envelope_data = np.zeros((n_ch, n_x), dtype="float")    
    
    for i in range(n_ch):
        # Bandpass filter
        b0, a0 = butter(order, [l_bpf, h_bpf], fs=fs, btype="band")
        bpfiltered_data = lfilter(b0, a0, data_flt[i])
        # Rectifying signal
        rectified_data = abs(bpfiltered_data)
        # Lowpass filter
        b1, a1 = butter(order, lpf_cut, fs=fs, btype="low")
        envelope_data[i] = lfilter(b1, a1, rectified_data)
    envelope_data_avg = envelope_data.mean(axis=0)

    return envelope_data_avg



def plot_env_data(data, 
                    fs=2048, 
                    order=4, 
                    l_bpf=40, 
                    h_bpf=900, 
                    lpf_cut=.2):
    """
    Plots envelope of data.
    
    Args:
    	data	: numpy.ndarray
            Array containing EMG data to be processed
        fs		: float
            sampling frequency (Hz)
        order	: int
            order of filter
        l_bpf	: float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	: float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	: float
            cutoff frequency of the low-pass filter (Hz)
    
    Example:
        plot_lpf_signal(gl_10_flatten[0])
    """
    
    envelope_data = env_data(data=data, 
                             fs=fs, 
                             order=order, 
                             l_bpf=l_bpf, 
                             h_bpf=h_bpf, 
                             lpf_cut=lpf_cut)
    
    # Plotting results
    x = np.arange(0, len(envelope_data))
    time = x / fs
    plt.rcParams['figure.figsize'] = [15,5]
    plt.plot(time, envelope_data)
    plt.show()
    


def acquire_remove_ind(data,
                       fs=2048,
                       order=4, 
                       l_bpf=40, 
                       h_bpf=900, 
                       lpf_cut=.2, 
                       tol=5e-6):
    """
    Retrieves indices of ramps in-between flat EMG signal that are to be removed.
    data is band-pass filtered, rectified, and low-pass filtered; 
    then starting and end points of ramps are marked.
    
    Args:
    	data	: numpy.ndarray
            Array containing EMG data to be processed
        fs		: float
            sampling frequency (Hz)
        order	: int
            order of filter
        l_bpf	: float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	: float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	: float
            cutoff frequency of the low-pass filter (Hz)
        tol		: float
            tolerated value of gradient ~ 0 for marking the start/end of a ramp
    
    Returns:
    	remove_ind	: numpy.ndarray
            contains the indices of the parts to be removed:
            - between data[remove_ind[0]] and data[remove_ind[1]]
            - between data[remove_ind[2]] and data[remove_ind[3]]
    
    Example:
        acquire_remove_ind(gl_10_flatten[0])
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
    # Remove duplicate points
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
            (13, 5) array containing 64 channels of EMG data to be modified
            
    Returns:
        data_mod    : numpy.ndarray
            (13, 5) array containing modified EMG data
        
    Example:
        modify_signal(gl_10, remove_ind_gl10)
    """
    
    data_mod = copy.deepcopy(data)
    
    # Acquiring indices to remove from data
    rm_ind_flt = np.ndarray.flatten(remove_ind)
    indices_to_remove = np.asarray([], dtype="int64")
    for i in range(len(rm_ind_flt) // 2):
        indices_to_remove = np.append( indices_to_remove, 
                                        [ np.arange(rm_ind_flt[2*i], rm_ind_flt[2*i + 1]) ] )
    
    # Removing indices from data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if len(data[i][j]) != 0:
                tmp = np.delete(data[i][j][0], indices_to_remove)
                data_mod[i][j] = np.asarray([tmp]) 
    return data_mod


def crop_data(data, start=0.0, end=40.0, fs=2048):
    """
    Returns the cropped data between start and end (in seconds).
    
    Args:
    	data	    : numpy.ndarray
            (13, 5) array containing 64 channels of EMG data to be modified
        start       : float
            starting time (in seconds) of the cropped data
        end         : float
            end time (in seconds) of the cropped data
        fs          : int
            sampling frequency (Hz)
            
    Returns:
        data_crop    : numpy.ndarray
            (13, 5) array containing EMG data in the specified window
        
    Example:
        gl_10_crop = crop_data(gl_10, end=70)
    """
    if (((data[0][0].size == 0 or
          data[12][0].size == 0) and data.ndim == 2) or 
         data.ndim == 3):  # .mat format 
        n_x = data[0][1].shape[1]
        if end > (n_x / fs):
            end = n_x / fs
        data_crop = copy.deepcopy(data)
        for i in range(0, data_crop.shape[0]):
            for j in range(0, data_crop.shape[1]):
                if np.size(data_crop[i][j]) != 0:
                    data_crop[i][j] = np.asarray([data[i][j][0][int(start*fs): int(end*fs)]])
    else:                   # .csv format
        n_x = data.shape[1]
        n_ch = data.shape[0]
        if end > (n_x / fs):
            end = n_x / fs
        nx_start = int(start*fs)
        nx_end = int(end*fs)
        data_crop = np.zeros((n_ch, nx_end), dtype="float")
        for i in range(0, data_crop.shape[0]):
            if np.size(data_crop[i]) != 0:
                data_crop[i] = np.asarray([data[i][nx_start : nx_end]])
    
    return data_crop

    

def crop_ind_pt(ind_pt, data, start=0.0, end=40.0, fs=2048):
    """
    Returns the cropped pt and data between start and end (in seconds).
    
    Args:
    	ind_pt  	: numpy.ndarray
            indices of motor units' pulse trains
        start       : float
            starting time (in seconds) of the cropped data
        end         : float
            end time (in seconds) of the cropped data
        fs          : int
            sampling frequency (Hz)
            
    Returns:
        ind_pt_crop : numpy.ndarray
            indices of motor units' pulse trains in the specified window
        data_crop   : numpy.ndarray
            (13, 5) array containing EMG data in the specified window
        
    Example:
        ind_pt_gl10_crop, gl_10_crop = crop_pt(gl_10, end=20.0)
    """
    data_crop = crop_data(data, start=start, end=end, fs=fs)
    n_mu = ind_pt.shape[0]
    if ((data_crop[0][0].size == 0 or
         data_crop[12][0].size == 0) and data_crop.ndim == 2) or data_crop.ndim == 3:
        x = data_crop[0][1].shape[1]
    else:
        x = data_crop.shape[1]

    start_idx = int(start * fs)
    end_idx = start_idx + x
    
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


def ind_to_pt(ind_pt, data,):
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


    
def visualize_pt(ind_pt, data, ref_signal=None, fs=2048, title="decomposition"):
    """
    Plots envelope of data and pulse trains of motor units from decomp.
    
    Args:
    	ind_pt  : numpy.ndarray
            indices of motor units' pulse trains
        data	: numpy.ndarray
            (13, 5) array containing 64 channels of EMG data
        fs		: float
            sampling frequency (Hz)
    
    Example:
        visualize_pt(decomp_gl_10_mod, gl_10_mod)
    """
    font_large = 24
    font_medium = 20
    font_small = 16
    
    if ind_pt.shape[0] != 1:
        ind_pt = ind_pt.squeeze()
    n_mu = ind_pt.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Pulse train
    pt = np.zeros((n_mu, n_x), dtype="int64")
    if n_mu > 1:
        for i in range(n_mu):
            pt[i][ind_pt[i]] = 1
    else:
        pt[0][ind_pt[0]]=1

    # Creating subplot
    n_rows = n_mu + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    x = np.arange(0, n_x, dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=font_medium)
    plt.rc('ytick', labelsize=font_medium)
    if ref_signal is None:
        # Plotting envelope of emg_data
        envelope_data = env_data(data)
        ax[0].plot(time, envelope_data)
    else:
        # Plotting ref_signal
        ref_signal = ref_signal.squeeze()
        ax[0].plot(time, ref_signal)
    ax[0].set_title(title, fontsize=font_large)

    for i in range(1, n_rows):
        ax[i].plot(time,pt[i-1])
        ax[i].set_ylabel(f"MU {i-1}", fontsize=font_medium)
    plt.show()


def visualize_pt_sort(ind_pt, data, ref_signal=None, asc=True , fs=2048, title="decomposition"):
    """
    Plots envelope of data and pulse trains of motor units from decomp.
    
    Args:
    	ind_pt  : numpy.ndarray
            indices of motor units' pulse trains
        data	: numpy.ndarray
            (13, 5) array containing 64 channels of EMG data
        fs		: float
            sampling frequency (Hz)
    
    Example:
        visualize_pt(decomp_gl_10_mod, gl_10_mod)
    """
    font_large = 24
    font_medium = 20
    font_small = 16

    ind_pt = ind_pt.squeeze()
    n_mu = ind_pt.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
        n_ch = flatten_signal(data).shape[0]
    else:
        n_x = data.shape[1]
        n_ch = data.shape[0]
    
    # Pulse train, sorted
    tmp = []
    pt_sort = np.zeros((n_mu, n_x), dtype="int64")
    for i in range(n_mu):
        ind_pt[i] = ind_pt[i].squeeze()
        tmp.append(ind_pt[i][0])
    tmp = np.array(tmp)
    mu_sort = np.argsort(tmp)
    if asc == False:
        mu_sort = np.flip(mu_sort)
    for i in range(n_mu):
        pt_sort[i][ind_pt[mu_sort[i]]] = 1
    
    # Creating subplot
    n_rows = n_mu + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    x = np.arange(0, n_x, dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=font_medium)
    plt.rc('ytick', labelsize=font_medium)
    if ref_signal is None:
        # Plotting envelope of emg_data
        envelope_data = env_data(data)
        ax[0].plot(time, envelope_data)
    else:
        # Plotting ref_signal
        ref_signal = ref_signal.squeeze()
        ax[0].plot(time, ref_signal)
    ax[0].set_title(title, fontsize=font_large)

    for i in range(1, n_rows):
        ax[i].plot(time,pt_sort[i-1])
        ax[i].set_ylabel(f"MU {mu_sort[i-1]}", fontsize=font_medium)
    plt.show()


def visualize_pt_window(ind_pt, data, ref_signal=None, start=0.0, end=40.0, fs=2048, title="decomposition"):
    """
    Plots a window of the envelope of data and pulse trains of motor units from decomp.
    
    Args:
    	ind_pt  : numpy.ndarray
            indices of motor units' pulse trains
        data	: numpy.ndarray
            (13, 5) array containing 64 channels of EMG data
        start       : float
            starting time (in seconds) of the window
        end         : float
            end time (in seconds) of the window
        fs		: float
            sampling frequency (Hz)
    
    Example:
        visualize_pt_window(decomp_gl_10_mod, gl_10_mod)
    """
    font_large = 24
    font_medium = 20
    font_small = 16
    
    # Windowed data
    data_crop = crop_data(data, start = start, end = end)
    n_mu = ind_pt.shape[0]
    if data_crop[0][0].size == 0 or data_crop.ndim == 3:
        n_x = data_crop[0][1].shape[1]
        n_ch = flatten_signal(data_crop).shape[0]
    else:
        n_x = data_crop.shape[1]
        n_ch = data_crop.shape[0]


    # Pulse train in the range of the window
    pt = np.zeros((n_mu, n_x), dtype="int64")
    for i in range(n_mu):
        for j in range(ind_pt[i].shape[0]):
            if ind_pt[i][j] < n_x:
                pt[i][ind_pt[i][j]] = 1
    
    # Creating subplot
    n_rows = ind_pt.shape[0] + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting reference signal
    if ref_signal is None:
        ref_signal = env_data(data_crop)
    else:
        ref_signal = ref_signal.squeeze()
        ref_signal = ref_signal[int(start*fs):int(end*fs)]
    x = np.arange(0, n_x, dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=font_medium)
    plt.rc('ytick', labelsize=font_medium)
    ax[0].plot(time, ref_signal)
    ax[0].set_title(title, fontsize=font_large)    
    
    # Plotting pulse trains
    for i in range(1, n_rows):
        y = pt[i-1]
        ax[i].plot(time,y)
        ax[i].set_ylabel(f"MU {i-1}", fontsize=font_medium)
    plt.show()


def compare_pt(ind_pt1, ind_pt2, data, fs=2048, title="decomposition", label1="offline", label2="realtime"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    n_mu = ind_pt1.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Pulse trains
    pt1 = np.zeros((n_mu, n_x), dtype="int64")
    pt2 = np.zeros((n_mu, n_x), dtype="int64")
    for i in range(n_mu):
        if ind_pt1[i].size != 0:
            pt1[i][ind_pt1[i]] = 1
        if ind_pt2[i].size != 0:
            pt2[i][ind_pt2[i]] = 1
    
    # Creating subplot
    n_rows = n_mu + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting envelope of emg_data
    envelope_data = env_data(data)
    x = np.arange(0, n_x, dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=font_medium)
    plt.rc('ytick', labelsize=font_medium)
    ax[0].plot(time, envelope_data)
    ax[0].set_title(title, fontsize=font_large)

    for i in range(1, n_rows):
        ax[i].plot(time, pt1[i-1], color="tab:blue", label=label1)
        ax[i].plot(time, pt2[i-1], color="tab:orange", label=label2)
        ax[i].set_ylabel(f"MU {i-1}", fontsize=font_medium)
        if i == 1:
            ax[i].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()



def calc_roa(ind_pt1, ind_pt2, data, decomp="realtime_decomp"):
    n_mu = ind_pt1.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    roa = np.zeros((n_mu), dtype="float")
    # Pulse trains
    pt1 = np.zeros((n_mu, n_x), dtype="int64")
    pt2 = np.zeros((n_mu, n_x), dtype="int64")
    for i in range(n_mu):
        if ind_pt1[i].size != 0:
            pt1[i][ind_pt1[i]] = 1
        if ind_pt2[i].size != 0:
            pt2[i][ind_pt2[i]] = 1
        # Number of discharges of the i-th motor unit identified by both decompositions
        c_i = np.dot(pt1[i], pt2[i])
        # Number of discharges of the i-th motor unit identified by ind_pt1
        a_i = pt1[i].sum() - c_i
        # Number of discharges of the i-th motor unit identified by ind_pt2
        b_i = pt2[i].sum() - c_i
        roa[i] = (c_i * 100) / (c_i + a_i + b_i)
    print(f"RoA between offline decomposition and {decomp} (%):")
    for i in range(n_mu):
        print(f"Motor unit {i}: {roa[i]}")
    print(f"mean: {roa.mean()}")
    return roa
    


def cross_corr(muap_dict_1, muap_dict_2):
    """
    Calculates the cross correlation between MUAPs of 2 decompositions, 
    muap_dict_1 and muap_dict_2
    
    Args:
        muap_dict_1 : dictionary of MUAP shapes for each motor unit
        muap_dict_2 : dictionary of MUAP shapes for each motor unit
    
    Returns:
        cc_values   : numpy array of cross correlation values
    
    Example:
        cc_gl10_gl30 = cross_corr(muap_gl10, muap_gl30)
    """
    
    # number of channels
    n_ch = np.max(muap_dict_1["mu_0"]["channel"]) + 1
    
    # number of motor units 
    n_mu_1 = len(muap_dict_1)
    n_mu_2 = len(muap_dict_2)

    # length of MUAP shape
    muap_size = int(muap_dict_1["mu_0"]["signal"].shape[0] / n_ch)
    
    # Initializing array to store cross correlation values
    cc_values = np.zeros((n_mu_1, n_mu_2, n_ch), dtype="float")
    
    # Comparing each MUAP (k=0-63) for every Motor Unit in muap_1 (i) against 
    #           each MUAP (k=0-63) for every Motor Unit in muap_2 (j) 
    for i in range(n_mu_1):
        for j in range(n_mu_2):
            for k in range(n_ch):
                # Normalized MUAP signal from muap_dict_1
                muap_1 = muap_dict_1[f"mu_{i}"]["signal"][muap_size*k : muap_size*(k+1)]
                muap_1_norm = (muap_1 - np.mean(muap_1)) / (np.std(muap_1) * len(muap_1))
                
                # Normalized MUAP signal from muap_dict_2
                muap_2 = muap_dict_2[f"mu_{j}"]["signal"][muap_size*k : muap_size*(k+1)]
                muap_2_norm = (muap_2 - np.mean(muap_2)) / np.std(muap_2)
                
                # Cross correlation
                cc_muap = np.correlate(muap_1_norm, muap_2_norm)
                
                # Store value in array
                cc_values[i][j][k] = cc_muap
                
    return cc_values
    


def calc_cc(signal1, signal2):
    # Normalized signal1
    signal1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) * len(signal1))
    
    # Normalized signal2
    signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # Cross correlation
    cc_signals = np.correlate(signal1_norm, signal2_norm)
    
    return cc_signals

    
def mean_cc(cc_values):
    """
    Calculates mean value of cross correlation between MUAPs across channels (64)
    from a known array of cross correlation values (cc_values)
        
    Args:
        cc_values       : numpy.ndarray
            (n_mu_1, n_mu_2, n_ch) array containing cross correlation values between
            MUAPs from MU1 and MU2
            
    Returns:
        mean_cc_values   : numpy.ndarray
            (n_mu_1, n_mu_2) array containing mean values of cross correlation between
            MUAPs from MU1 and MU2
   
    """
    
    mean_cc_values = np.mean(cc_values, axis=2, dtype="float64")
    
    return mean_cc_values
    


def find_high_cc(mean_cc_values, decomp1="decomposition_1", decomp2="decomposition_2", thr=.75):
    """
    Retrieves indices of where mean_cc_values > thr;
    and prints these values and indices
    
    Args:
        mean_cc_values  : numpy.ndarray
            (n_mu_1, n_mu_2) array containing mean values of cross correlation between
            MUAPs from MU1 and MU2
        thr             : float
            threshold value
    
    Returns:
        high_cc_values  : numpy.ndarray
            array containing indices where mean_cc_values > thr
    """
    high_cc_values = np.argwhere(mean_cc_values > thr)

    index_max = np.unravel_index(np.argmax(mean_cc_values), mean_cc_values.shape)
    
    print("mean cc_values for", decomp1,"and", decomp2, "higher than", thr, ":",)
    print(np.max(mean_cc_values), "at", index_max)
    for i in range(high_cc_values.shape[0]):
        tmp1 = high_cc_values[i][0]
        tmp2 = high_cc_values[i][1]
        print(mean_cc_values[tmp1][tmp2], "at", high_cc_values[i])
    return high_cc_values



def plot_meancc(mean_cc_values, y_axis="decomposition_1", x_axis="decomposition_2"):
    """
    Plots mean cross correlation between motor units from 2 decompositions.
    
    Args:
    	mean_cc_values	: numpy.ndarray
            2D array containing mean cross correlation values
        y_axis  		: char
            name of y axis
        x_axis      	: char
            name of x axis
            
    Example:
        plot_meancc(mean_cc_gl10_gl10_mod)
    """
    font_large = 16
    font_medium = 12
    font_small = 10
    
    # make plot
    fig, ax = plt.subplots()
    ratio = mean_cc_values.shape[1] / mean_cc_values.shape[0]
    plt.rcParams["figure.figsize"] = [10*ratio, 10]

    # show image
    shw = ax.imshow(mean_cc_values)

    # make bar
    bar = plt.colorbar(shw, fraction=0.046, pad=0.04)

    # show plot with labels
    plt.xlabel(x_axis, fontsize=font_medium)
    plt.ylabel(y_axis, fontsize=font_medium)
    bar.set_label('Mean cross-correlation', fontsize=font_medium)
    plt.rc('xtick', labelsize=font_small)
    plt.rc('ytick', labelsize=font_small)
    plt.show()


def calc_isi(ind_pt, fs=2048):
    n_mu = ind_pt.shape[0]
    isi = []
    
    # Inter-spike interval for each MU
    for i in range(n_mu):
        isi.append(np.diff(ind_pt[i])*1000/fs)
    isi = np.array(isi, dtype="object")
    return isi


def hist_isi(ind_pt, fs=2048, title="ISI histogram"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    n_mu = ind_pt.shape[0]
    isi = calc_isi(ind_pt=ind_pt, fs=fs)
    
    # Creating subplot
    n_rows = n_mu
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [15, 2*n_rows]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    ax[0].set_title(title, fontsize=font_large)
    
    for i in range(n_rows):
        ax[i].hist(isi[i], 100)
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        ax[i].set_xlabel("interspike interval (ms)", fontsize=font_small)
        ax[i].tick_params(axis='both', which='major', labelsize=font_small)
    plt.show()


def calc_cov(isi):
    n_mu = isi.shape[0]
    cov = np.zeros(n_mu)

    for i in range(n_mu):
        cov[i] = isi[i].std() / isi[i].mean()

    return cov



def plot_firing_rate(ind_pt, data, time_bin=.4, filt=True, filt_size=.2, fs=2048, title="Firing rate"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    n_mu = ind_pt.shape[0]
    isi = []
    n_bin = int(time_bin * fs)
    
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Firing rate for each MU 
    firing_rates = np.zeros((n_mu, n_x), dtype="float")
    filtered = np.zeros((n_mu, n_x), dtype="float")
    
    for mu in range(n_mu):
        # Pulse train
        pt_i = np.zeros(n_x, dtype="int64")
        if ind_pt[mu].size != 0:
            pt_i[ind_pt[mu]] = 1
        # Spikes every n_bin
        for i in range(int(n_bin/2), int(n_x-(n_bin/2))):
            spikeCount = pt_i[i-int(n_bin/2) : i+int(n_bin/2)].sum()
            firing_rates[mu][i] = fs * spikeCount/n_bin

        if filt:    
            # filtered firing rate
            windowSize = fs * filt_size
            window = np.hanning(windowSize)
            window = window / window.sum()
            filtered[mu] = np.convolve(window, firing_rates[mu], mode='same')
        else:
            filtered[mu] = firing_rates[mu]

    # Creating subplot
    n_rows = n_mu
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [15, 2.5*n_rows]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    ax[0].set_title(title, fontsize=font_large)
    
    time_axis = np.arange(n_x, dtype="float")/fs
    for i in range(n_rows):
        ax[i].plot(time_axis, filtered[i])
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        ax[i].tick_params(axis='both', which='major', labelsize=font_small)
    plt.show()



def compare_firing_rate(ind_pt1, ind_pt2, data, 
                        time_bin=4.0, filt=False, filt_size=.2, fs=2048,
                        title="Firing rate", label1="realtime", label2="offline"):
    font_large = 16
    font_medium = 12
    font_small = 10
    
    n_mu = ind_pt1.shape[0]
    n_bin = int(time_bin * fs)
    
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Firing rate for each MU 
    firing_rates_1 = np.zeros((n_mu, n_x), dtype="float")
    firing_rates_2 = np.zeros((n_mu, n_x), dtype="float")
    filtered_1 = np.zeros((n_mu, n_x), dtype="float")
    filtered_2 = np.zeros((n_mu, n_x), dtype="float")
    
    for mu in range(n_mu):
        # Pulse train
        pt1_i = np.zeros(n_x, dtype="int64")
        pt2_i = np.zeros(n_x, dtype="int64")
        if ind_pt1[mu].size != 0:
            pt1_i[ind_pt1[mu]] = 1
        if ind_pt2[mu].size != 0:
            pt2_i[ind_pt2[mu]] = 1
        # Spikes every second
        for i in range(int(n_bin/2), int(n_x-(n_bin/2))):
            spikeCount1 = pt1_i[i-int(n_bin/2) : i+int(n_bin/2)].sum()
            firing_rates_1[mu][i] = fs * spikeCount1/n_bin
            spikeCount2 = pt2_i[i-int(n_bin/2) : i+int(n_bin/2)].sum()
            firing_rates_2[mu][i] = fs * spikeCount2/n_bin

        if filt:    
            # filtered firing rate
            windowSize = fs * filt_size
            window = np.hanning(windowSize)
            window = window / window.sum()
            filtered_1[mu] = np.convolve(window, firing_rates_1[mu], mode='same')
            filtered_2[mu] = np.convolve(window, firing_rates_2[mu], mode='same')
        else:
            filtered_1[mu] = firing_rates_1[mu]
            filtered_2[mu] = firing_rates_2[mu]

    # Creating subplot
    n_rows = n_mu
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [15, 2.5*n_rows]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    ax[0].set_title(title, fontsize=font_large)
    
    time_axis = np.arange(n_x, dtype="float")/fs
    for i in range(n_rows):
        ax[i].plot(time_axis, filtered_1[i], color="tab:blue", label=label1)
        ax[i].plot(time_axis, filtered_2[i], color="tab:orange", label=label2)
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        ax[i].tick_params(axis='both', which='major', labelsize=font_small)
        if i==0:
            ax[i].legend(loc='upper right', shadow=False, fontsize=font_small)
    plt.show()
    return filtered_1, filtered_2



def compare_firing_rate2(ind_pt1, ind_pt2, data, 
                        lpf_hanning=False, filt_size=2.0, fs=2048,
                        order=4, lpf_cut=2,
                        title="Firing rate", label1="realtime", label2="offline"):
    font_large = 16
    font_medium = 12
    font_small = 10
    
    n_mu = ind_pt1.shape[0]
    
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
    # Firing rate for each MU 
    firing_rates_1 = np.zeros((n_mu, n_x), dtype="float")
    firing_rates_2 = np.zeros((n_mu, n_x), dtype="float")
    fr_similarity = np.zeros(n_mu, dtype="float")
    
    if lpf_hanning:
        for mu in range(n_mu):
            # Pulse train
            pt1_i = np.zeros(n_x, dtype="int64")
            pt2_i = np.zeros(n_x, dtype="int64")
            if ind_pt1[mu].size != 0:
                pt1_i[ind_pt1[mu]] = 1
            if ind_pt2[mu].size != 0:
                pt2_i[ind_pt2[mu]] = 1
            
            # Lowpass filter
            b1, a1 = butter(N=order, Wn=lpf_cut, fs=fs, btype="low")
            firing_rates_1[mu] = lfilter(b1, a1, pt1_i)
            firing_rates_2[mu] = lfilter(b1, a1, pt2_i)
            
            fr_similarity[mu] = calc_cc(signal1=firing_rates_1[mu],
                                        signal2=firing_rates_2[mu])
            
    else:
        for mu in range(n_mu):
            # Pulse train
            pt1_i = np.zeros(n_x, dtype="int64")
            pt2_i = np.zeros(n_x, dtype="int64")
            if ind_pt1[mu].size != 0:
                pt1_i[ind_pt1[mu]] = 1
            if ind_pt2[mu].size != 0:
                pt2_i[ind_pt2[mu]] = 1
               
            # Hanning window
            windowSize = fs * filt_size
            window = np.hanning(windowSize)
            window = window * fs / window.sum()
            firing_rates_1[mu] = np.convolve(window, pt1_i, mode='same')
            firing_rates_2[mu] = np.convolve(window, pt2_i, mode='same')

            fr_similarity[mu] = calc_cc(signal1=firing_rates_1[mu],
                                        signal2=firing_rates_2[mu])            

    # Creating subplot
    n_rows = n_mu
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [15, 2.5*n_rows]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    ax[0].set_title(title, fontsize=font_large)
    
    time_axis = np.arange(n_x, dtype="float")/fs
    for i in range(n_rows):
        ax[i].plot(time_axis, firing_rates_1[i], color="tab:blue", label=label1)
        ax[i].plot(time_axis, firing_rates_2[i], color="tab:orange", label=label2)
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        ax[i].text(.02,.9, f"similarity={fr_similarity[i]:.4f}", 
                   fontsize=font_medium, transform=ax[i].transAxes)
        ax[i].tick_params(axis='both', which='major', labelsize=font_small)
        if i==0:
            ax[i].legend(loc='upper right', shadow=False, fontsize=font_small)
    plt.show()
    return firing_rates_1, firing_rates_2, fr_similarity



def muap_dict_mod(data, ind_pt, l=31):
    # ind_pt, data = crop_ind_pt(decomp_gl_10["MUPulses"], gl_10, start=10.0, end=50.0)
    # muap_dict_mod(data, ind_pt)
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        data = flatten_signal(data)
    channels = data.shape[0]
    shape_dict = {}

    for i in range(ind_pt.shape[0]):
        ind_pt[i] = ind_pt[i].squeeze()

        # Create array to contain indices of peak shapes
        ptl = np.zeros((ind_pt[i].shape[0], l * 2 + 1), dtype="int")

        for j in range(ind_pt[i].size):
            if ind_pt[i].size == 1:
                k = ind_pt[i]
            else:
                k = ind_pt[i][j]
            ptl[j] = np.arange(k - l, k + l + 1)

            # This is to ensure that if early peak happens before half of AP discharge time.
            # that there is no negative indices
            if k < l:
                ptl[j] = np.arange(k - l, k + l + 1)
                neg_idx = abs(k - l)
                ptl[j][:neg_idx] = np.repeat(0, neg_idx)

        ptl = ptl.flatten()

        # Create channel index of each peak
        channel_index = np.repeat(np.arange(channels), l * 2 + 1)

        # Get sample number of each position along each peak
        sample = np.arange(l * 2 + 1)
        sample = np.tile(sample, channels)

        # Get average signals from each channel
        signal = (
            data[:, ptl]
            .reshape(channels, ptl.shape[0] // (l * 2 + 1), l * 2 + 1)
            .mean(axis=1)
            .flatten()
        )

        shape_dict[f"mu_{i}"] = {
            "sample": sample,
            "signal": signal,
            "channel": channel_index,
        }
"""
data = gl_10
ind_pt = decomp_gl_10["MUPulses"]

muap_data = muap_dict(data, ind_pt)

n_mu = ind_pt.shape[0]
if ((data[0][0].size == 0 or
     data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
    data = flatten_signal(data)
    n_ch = data.shape[0]
    x = data.shape[1]
else:
    n_ch = data.shape[0]
    n_ch = data.shape[1]
muap_len = l*2 + 1

muap_shapes = np.zeros((n_mu, n_ch, muap_len), dtype="float")

firing_rates = np.zeros((n_mu, n_ch, x), dtype="float")
for i in range(n_mu):    
    for channel in range(n_ch):
        muap_shape = muap_data[f"mu_{i}"]["signal"][channel*muap_len : (channel+1)*muap_len]
        firing_rates[i][channel] = np.convolve(muap_shape, data[channel], "same")

mean_firing_rates = np.mean(firing_rates, axis=1, dtype="float")
"""
