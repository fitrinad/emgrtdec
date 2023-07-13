import numpy as np
import matplotlib.pyplot as plt



def calc_roa(ind_pt1, ind_pt2, data, decomp="realtime_decomp"):
    """
    Calculates the rate of agreement (RoA) between 2 pulse trains (ind_pt1 and ind_pt2) 
    from 2 different decompositions of data (realtime decomposition and offline decomposition).
    
    Args:
    	ind_pt1 : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 1;
            extracted from data
        ind_pt2 : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 2;
            extracted from data
        data    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        decomp  : string
            String to name the decomposition being compared to the offline decomposition
    Returns:
        roa     : numpy.ndarray
            Array containing the RoA between ind_pt1 and ind_pt2
    """

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
    Calculates the cross correlation between motor unit action potentials (MUAPs) of 2 decompositions, 
    muap_dict_1 and muap_dict_2.
    
    Args:
        muap_dict_1 : dict 
            Dictionary of MUAP shapes for each motor unit from decomposition 1
        muap_dict_2 : dict 
            Dictionary of MUAP shapes for each motor unit from decomposition 2
    
    Returns:
        cc_values   : numpy.ndarray 
            Array containing cross correlation values between MUAP shapes of decomposition 1
            (muap_dict_1) and MUAP shapes of decomposition 2 (muap_dict_2)
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
    """
    Calculates the cross correlation between 2 signals (signal1 and signal2).
    
    Args:
        signal1     : numpy.ndarray 
            1D array containing signal
        signal2     : numpy.ndarray 
            1D array containing signal
    
    Returns:
        cc_signals  : float 
            Cross correlation value between signal1 and signal2
    """
    
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
            Threshold value
    
    Returns:
        high_cc_values  : numpy.ndarray
            Array containing indices where mean_cc_values > thr
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
            Name of y axis
        x_axis      	: char
            Name of x axis
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
    """
    Calculates the inter-spike interval (ISI) of all motor units' pulse trains in ind_pt.

    Args:
    	ind_pt  : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing
        fs      : float
            Sampling frequency (Hz)

    Returns:
        isi     : numpy.ndarray
            Array containing the ISI for each MU in ind_pt
    """
    n_mu = ind_pt.shape[0]
    isi = []
    
    # Inter-spike interval for each MU
    for i in range(n_mu):
        isi.append(np.diff(ind_pt[i])*1000/fs)
    isi = np.array(isi, dtype="object")
    return isi


def hist_isi(ind_pt, fs=2048, title="ISI histogram"):
    """
    Plots a histogram of the inter-spike interval (ISI) of all motor units' pulse trains in ind_pt.

    Args:
    	ind_pt  : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing
        fs      : float
            Sampling frequency (Hz)
        title   : string
            Title of resulting figure    
    """
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
    """
    Calculates the coefficient of variation (CoV) for the inter-spike interval (ISI) of 
    each motor unit in isi.

    Args:
    	isi     : numpy.ndarray
            Array containing the ISI for each MU
        
    Returns:
        cov     : numpy.ndarray
            Array containing the CoV for each MU
    """
    n_mu = isi.shape[0]
    cov = np.zeros(n_mu)

    for i in range(n_mu):
        cov[i] = isi[i].std() / isi[i].mean()

    return cov


def plot_firing_rate(ind_pt, data, time_bin=.4, filt=True, filt_size=.2, fs=2048, title="Firing rate"):
    """
    Plots the firing rate of all motor units in ind_pt, which were extracted from data.
    
    Args:
    	ind_pt      : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing,
            extracted from data
        data	    : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        time_bin    : float
            The width of the time window used to calculate the number of spikes
        filt        : bool
            If filt == True, applies a Hanning filter to the array of calculated spike count
        filt_size   : float
            Size of the applied Hanning filter (in seconds)
        fs		: float
            Sampling frequency (Hz)
        title   : string
            Title of resulting figure
    """
    
    font_large = 24
    font_medium = 20
    font_small = 16
    
    n_mu = ind_pt.shape[0]
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


def compare_firing_rate2(ind_pt1, ind_pt2, data,
                         lpf_hanning=False, filt_size=2.0, fs=2048,
                         order=4, lpf_cut=2,
                         title="Firing rate", label1="realtime", label2="offline"):
    """
    Plots the firing rate of all motor units in ind_pt1 and ind_pt2, which were extracted from data; and
    also shows the similarity between 2 firing rate curves. 
    
    Args:
    	ind_pt1         : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 1;
            extracted from data
        ind_pt2         : numpy.ndarray
            Array containing indices where the motor units' pulse trains are firing, from decomposition 2;
            extracted from data
        data            : numpy.ndarray
            Array containing EMG data, could contain an empty channel
            (if the channels are in a grid of (13,5))
        lpf_hanning     : bool
            If lpf_hanning == True, applies a low-pass filter to the pulse train.
            If lpf_hanning == False, applies a Hanning filter to the pulse train.
        filt_size       : float
            Size of the applied Hanning filter (in seconds)
        fs		        : float
            Sampling frequency (Hz)
        order           : int
            Order of the low-pass filter
        lpf_cut         : float
            Cutoff frequency of the low-pass filter
        title           : string
            Title of resulting figure
        label1          : string
            Label of the firing rate curve for ind_pt1
        label2          : string
            Label of the firing rate curve for ind_pt2
    
    Returns:
        firing_rates_1  : numpy.ndarray
            Array containing the firing rate curve of each motor unit in ind_pt1
        firing_rates_2  : numpy.ndarray
            Array containing the firing rate curve of each motor unit in ind_pt2
        fr_similarity   : numpy.ndarray
            Array containing the similarity value between each motor unit's firing curve
            in ind_pt1 and ind_pt2
    """
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
    
    if lpf_hanning: # Using a lowpass filter if True
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
            
    else:           # Using a Hanning filter
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

