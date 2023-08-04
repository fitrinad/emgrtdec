import numpy as np
import matplotlib.pyplot as plt
from functions.preprocessing import (env_data, crop_data)
from emgdecompy.preprocessing import flatten_signal

def plot_signal(data, fs=2048, title="EMG signal"):
    """
    Plots all channels of an array containing EMG signal, data.  
        
    Args:
        data	        : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        fs              : float
            Sampling frequency (Hz)
        title           : string
            Title of resulting figure
    """
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
    """
    Plots SIL scores (sil_scores) of each motor unit pulse train from a simulated realtime decomposition.
        
    Args:
        sil_scores      : numpy.ndarray
            Array containing SIL scores of each motor unit from a simulated realtime decomposition
        thd_sil         : float
            SIL score threshold, a line is plotted when specified
        fs              : float
            Sampling frequency (Hz)
        title           : string
            Title of resulting figure
    """
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
    ## Plotting SIL threshold line and SIL scores
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


def plot_env_data(data, fs=2048, order=4, l_bpf=40, h_bpf=900, lpf_cut=.2):
    """
    Plots envelope of data.
    
    Args:
    	data	: numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        fs		: float
            Sampling frequency (Hz)
        order	: int
            Order of filter
        l_bpf	: float
            Lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	: float
            Higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	: float
            Cutoff frequency of the low-pass filter (Hz)
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
    

def visualize_pt(ind_pt, data, ref_signal=None, fs=2048, 
                 use_filt=True, filt_size=2.0, title="decomposition",
                 export_files=False, 
                 file_name_pdf="decomposition.pdf", 
                 file_name_png="decomposition.png"):
    """
    Plots reference signal of data and pulse trains of motor units from decomp.
    If ref_signal = None, plots the envelope of data instead of a reference signal.

    
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
    
    if ind_pt.shape[0] == 1 and ind_pt.ndim > 1:
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
        if use_filt:
            # Hanning window
            windowSize = fs * filt_size
            window = np.hanning(windowSize)
            window = window / window.sum()
            ref_signal = np.convolve(window, ref_signal.squeeze(), mode='same')
        else:
            # Plotting ref_signal
            ref_signal = ref_signal.squeeze()
        ax[0].plot(time, ref_signal)
    ax[0].set_title(title, fontsize=font_large)

    for i in range(1, n_rows):
        ax[i].plot(time,pt[i-1])
        ax[i].set_ylabel(f"MU {i-1}", fontsize=font_medium)

    if export_files == True:
        plt.savefig(file_name_pdf)
        plt.savefig(file_name_png)
        
    plt.show()

    
    
def visualize_pt_tmod(ind_pt, data, ref_signal=None, fs=2048, title="decomposition"):
    """
    Plots reference signal of data and pulse trains of motor units from decomp.
    If ref_signal = None, plots the envelope of data instead of a reference signal.
    
    Args:
    	ind_pt      : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        ref_signal  : numpy.ndarray
            Array containing measured force
        fs		    : float
            Sampling frequency (Hz)
        title       : string
            Title of resulting figure
    """
    font_large = 14 # 24
    font_medium = 10 # 20
    font_small = 6 # 16
    
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
    # plt.rc('xtick', labelsize=font_medium)
    # plt.rc('ytick', labelsize=font_small)
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
        ax[i].set_ylabel(f"MU {i-1}", fontsize=font_small)
        ax[i].tick_params(axis='x', which='major', labelsize=font_medium)
        # ax[i].tick_params(axis='x', which='minor', labelsize=font_small)
        ax[i].tick_params(axis='y', which='major', labelsize=font_small)
        # ax[i].tick_params(axis='y', which='minor', labelsize=font_small)
    
    fig.subplots_adjust(top=0.975,
                        bottom=0.075,
                        left=0.250,
                        right=0.750,
                        hspace=0.1,
                        wspace=0.1)
    plt.show()


def visualize_pt_sort(ind_pt, data, ref_signal=None, asc=True , fs=2048, title="decomposition"):
    """
    Plots reference signal of data and pulse trains of motor units from decomp by the order of their first firing.
    If ref_signal = None, plots the envelope of data instead of a reference signal.
    Motor unit pulse trains are plotted in order of the first firing: 
    - motor unit with the earliest firing first if asc == True, 
    - motor unit with the earliest firing last if asc == False. 
    
    Args:
    	ind_pt      : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        ref_signal  : numpy.ndarray
            Array containing measured force
        asc         : bool
            Motor unit with the earliest first firing will be plotted: 
            - first if asc == True, 
            - last if asc == False
        fs		    : float
            sampling frequency (Hz)
        title       : string
            Title of resulting figure
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
    	ind_pt      : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        ref_signal  : numpy.ndarray
            Array containing measured force
        start       : float
            Starting time (in seconds) of the window
        end         : float
            End time (in seconds) of the window
        fs		: float
            Sampling frequency (Hz)
        title       : string
            Title of resulting figure
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
    """
    Plots the envelope of data and pulse trains of motor units from 2 different decompositions.
    
    Args:
    	ind_pt1  : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 1;
            extracted from data
        ind_pt2  : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 1;
            extracted from data
        data	: numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        fs		: float
            Sampling frequency (Hz)
        title   : string
            Title of resulting figure
        label1  : string
            Label of ind_pt1
        label2  : string
            Label of ind_pt2
    """
    
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
    
    # Plotting envelope of data
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

