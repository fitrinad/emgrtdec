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
    """
    font_large = 24
    font_medium = 20
    font_small = 16
    
    # if ind_pt.shape[0] == 1 and ind_pt.ndim > 1:
    #     ind_pt = ind_pt.squeeze()
    n_mu = ind_pt.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        ind_pt = ind_pt.squeeze()
        n_mu = ind_pt.shape[0]
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
    Used after the training module window finished its decomposition.
    
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
    
    ind_pt = ind_pt.squeeze()
    n_mu = ind_pt.shape[0]
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        n_x = data[0][1].shape[1]
    else:
        n_x = data.shape[1]
    
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


def visualize_muap(muap_dict, mu_index, electrode_type = "ext", title = "MUAP shapes"):
    """
    Visualizes MUAP shapes at 64 channels of a selected MU from muap_dict (dictionary of MUAP shapes from a decomposition result).

    Args:
    	muap_dict       : dict
            Dictionary of MUAP shapes
        mu_index        : int
            Index of selected MU to be visualized
        electrode_type  : string
            "int"   : Plots the MUAP shapes in a configuration of 13 x 5
            "ext"   : Plots the MUAP shapes in a configuration of 8 x 8
        title   : string
            Title of resulting figure
        
    """
    
    font_large = 20
    font_medium = 16
    
    # Number of channels
    n_ch = np.max(muap_dict["mu_0"]["channel"]) + 1
    # Length of MUAP shape
    muap_size = int(muap_dict["mu_0"]["signal"].shape[0] / n_ch)

    muap_max = np.max(muap_dict[f"mu_{mu_index}"]["signal"])
    muap_min = np.min(muap_dict[f"mu_{mu_index}"]["signal"])
    muap_mean = np.mean(muap_dict[f"mu_{mu_index}"]["signal"])
    muap_min_norm = (muap_min - muap_mean) / muap_size
    muap_max_norm = (muap_max - muap_mean) / muap_size

    if electrode_type == "ext":
        # Creating subplots of size (8, n_ch/8)
        n_cols = 8
        n_rows = int(n_ch / n_cols)
        plt.rcParams['figure.figsize'] = [32, 4*n_rows]
        fig, ax = plt.subplots(n_rows, n_cols)

        # Plotting muap shapes in 
        for k in range(0, n_ch, 1):
            row = divmod(k, n_rows)[1]
            col = divmod(k, n_rows)[0]
            muap = muap_dict[f"mu_{mu_index}"]["signal"][muap_size*k : muap_size*(k+1)]
            muap_norm = np.nan_to_num((muap - muap_mean) / len(muap))

            ax[row, col].plot(muap_norm)
            ax[row, col].set_ylabel(f"ch {k}", fontsize=font_medium)
            ax[row, col].set_ylim([muap_min_norm, muap_max_norm])
            ax[row, col].set_xticks([])
            ax[row, col]. set_yticks([])
    if electrode_type == "int":
        # Creating subplots of size (8, n_ch/8)
        n_cols = 5
        n_rows = int(np.ceil(n_ch / n_cols))
        plt.rcParams['figure.figsize'] = [20, 4*n_rows]
        fig, ax = plt.subplots(n_rows, n_cols)

        # Plotting muap shapes in 
        for k in range(0, n_ch, 1):
            row = divmod(k+1, n_rows)[1]
            col = n_cols - divmod(k+1, n_rows)[0] - 1
            muap = muap_dict[f"mu_{mu_index}"]["signal"][muap_size*k : muap_size*(k+1)]
            muap_norm = np.nan_to_num((muap - muap_mean) / len(muap))

            ax[row, col].plot(muap_norm)
            ax[row, col].set_ylabel(f"ch {k}", fontsize=font_medium)
            ax[row, col].set_ylim([muap_min_norm, muap_max_norm])
            ax[row, col].set_xticks([])
            ax[row, col]. set_yticks([])
        ax[0, n_cols-1].set_visible(False)
    fig.suptitle(title, size=font_large, y = 0.89)
    plt.subplots_adjust(hspace = 0.15, wspace = 0.15)
    plt.show()



