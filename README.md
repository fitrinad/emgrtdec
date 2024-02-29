# emgrtdec
An implementation of real-time high-density EMG signals decomposition based on the algorithm presented in [`D. Y. Barsakcioglu, D. Farina (2018)`](https://ieeexplore.ieee.org/document/8584659) and [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

The packages needed for the user interface are included in `requirements.txt` and can be installed to a virtual environment.

## User Interface
To start the user interface, run `python tmod_window.py`. This window runs the training module on previously recorded EMG data, calculating a separation matrix and the motor unit spike trains. After the decomposition is complete, a window showing the resulting motor unit spike trains will appear and the results are saved in the folder `tmod_data`.

Clicking `Realtime Decomposition` will open `decmod_window.py`, where the real-time decomposition module can be simulated on recorded EMG data or applied to live EMG data streamed from the SyncStation+. The user can choose which motor units extracted by the training module will be used in the real-time decomposition. `Calculate Separation Matrix` will compute the real-time separation matrix used in the decomposition. If `Simulated data` is unselected, the UI will connect to the SyncStation. When `Plot` is clicked, the window will start the decomposition and visualize the resulting motor unit spike trains. `Decompose` only applies the decomposition, but does not visualize the results. After the real-time decomposition is stopped, the results are saved in the folder `dec_data`.

## `data`
Contains an open source EMG data file used for the real-time decomposition evaluation in `Experimental_data_Raw` from [`Hug et al. (2021)`](https://doi.org/10.6084/m9.figshare.13695937).

## `functions`
Contains functions for preprocessing, visualization, and analysis of EMG data. 

## `user_interface`
Contains the .ui files of the training module window and the decomposition module window. These can be edited with Qt Designer.

## Offline decomposition
`tmod.py` contains the function used for offline decomposition: `decomposition_tmod`. This can be used separately and saves the results as a dictionary.

