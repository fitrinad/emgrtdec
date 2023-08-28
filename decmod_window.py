from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, QThreadPool
from PyQt5 import uic, QtWidgets
import pyqtgraph as pg
from scipy.io import (loadmat, savemat)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from Worker import Worker
import collections
from MuoviHandler import MuoviApp
import sys
import signal
import warnings
import serial
import serial.tools.list_ports
from scipy.signal import butter

import os
import time
from functions.preprocessing import (crop_data, apply_filter)
from tmod import *
from decmod import (rt_decomp, sim_decomp_plot)
from rt_decmod_cy.rt_decomp_live_cy import rt_decomp_live_cy
from rt_decmod_cy.rt_decomp_plotpg_cy import rt_decomp_plotpg_cy


abs_path = os.path.dirname(__file__)

class Window_decmod(QMainWindow):   # Decomposition Module window
    def __init__(self):
        super(Window_decmod, self).__init__()
        uic.loadUi("./user_interface/gui_decmod.ui", self)
        self.setWindowTitle("Decomposition Module")
        self.show()

        # Default trained module
        self.tmod_file = "./tmod_data/dec_gm10_x62_2tmod.obj"
        #   Default sampling frequency
        self.fs = 2048
        #   Default discard channels
        self.discard_ch = 62
        #   Default extension factor
        self.ext_factor = 16

        # Default simulated data
        self.use_simulated_data = True
        self.emg_file = "./data/Experimental_data_Raw/GM_10.mat"
        self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.start_time = 68.0
        self.end_time = 102.0
        self.start_time_plot = 0.0
        self.end_time_plot = self.end_time - self.start_time
        self.time_diff = 0.0
        self.emgdata_rt = flatten_signal(crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time,
                                 end=self.end_time,
                                 fs=self.fs))

        # Default parameter values
        self.save_foldername = "decomp_rt"
        self.save_filename_pt = self.save_foldername + "_pt.mat"
        self.save_filename_emg = self.save_foldername + "_emg.csv"
        self.save_filename_param = self.save_foldername + "_param.mat"
        self.batch_size = 4.0
        self.overlap = 3.5

        self.min_distance = 31
        self.use_bpf = True
        self.lowcut_freq = 20.0
        self.highcut_freq = 500.0
        self.order = 4
        self.use_sil = True
        self.sil_threshold = 0.85
        self.use_pps = False
        self.pps_threshold = 5
        
        self.display_interval = 10.0
        self.update_interval = self.batch_size - self.overlap  
        self.max_nsamples = int(self.display_interval * self.fs)

        # Showing trained module parameters
        self.lineEdit_tmodFile.setText(self.tmod_file)
        self.lineEdit_SamplingFreq.setText(str(self.fs))
        if self.lineEdit_DiscardChannels.text() == "":
            self.lineEdit_DiscardChannels.setText(str(self.discard_ch))
        else:
            self.lineEdit_DiscardChannels.setText(self.lineEdit_DiscardChannels.text())
        
        # Showing realtime decomposition parameters
        self.lineEdit_ExtFactor.setText(str(self.ext_factor))
        self.checkBox_SimulatedData.setChecked(self.use_simulated_data)
        self.get_simulated_data()
        self.lineEdit_EMGFile.setText(self.emg_file)
        self.lineEdit_StartTime.setText(str(self.start_time))
        self.lineEdit_EndTime.setText(str(self.end_time))
        self.lineEdit_BatchSize.setText(str(self.batch_size))
        self.lineEdit_Overlap.setText(str(self.overlap))
        self.checkBox_SIL.setChecked(self.use_sil)
        self.lineEdit_SILThreshold.setText(str(self.sil_threshold))
        self.get_sil_threshold()
        self.checkBox_pps.setChecked(self.use_pps)
        self.lineEdit_ppsThreshold.setText(str(self.pps_threshold))
        self.get_pps_threshold()
        self.lineEdit_MinDistance.setText(str(self.min_distance))
        self.checkBox_BandpassFilter.setChecked(self.use_bpf)
        self.lineEdit_lowcutFreq.setText(str(self.lowcut_freq))
        self.lineEdit_highcutFreq.setText(str(self.highcut_freq))
        self.lineEdit_order.setText(str(self.order))
        self.get_bandpass_filter()
        self.lineEdit_Name.setText(self.save_foldername)

        
        # Initialization
        # self.n_count = 0
        self.sel_array = 1          # Selected electrode array: 1 or 2
        self.grid_size = (8, 8)     # Grid size: (8, 8) or (16, 8)
        self.use_arduino = False
        self.is_recording = False
        # self.is_recording_sim = False
        # self.is_recording_live = False
        self.is_pulling_emg = False
        
        self.data_live = collections.deque(maxlen=8)
        self.data_live.append(np.zeros((16, 8)))
        self.data_live.append(np.zeros((16, 8)))
        self.data_live.append(np.zeros((16, 8)))
        self.data_live.append(np.zeros((16, 8)))
        self.data_live.append(np.zeros((16, 8)))
        #self.data_live.append(np.zeros((16, 8)))
        data_live = np.uint8(self.data_live[-1])
        
        self.current_batch = []
        
        self.visTimer = QTimer()
        self.threadpool = QThreadPool()
        
        self.checkBox_SimulatedData.toggled.connect(self.get_simulated_data)
        self.checkBox_BandpassFilter.toggled.connect(self.get_bandpass_filter)
        self.checkBox_SIL.toggled.connect(self.get_sil_threshold)
        self.checkBox_pps.toggled.connect(self.get_pps_threshold)

        self.pushButton_tmodBrowse.clicked.connect(self.tmodbrowse_clicked)
        self.pushButton_ShowMU.clicked.connect(self.showmu_clicked)
        self.pushButton_SelectAll.clicked.connect(self.selectall_clicked)
        self.pushButton_CalcSepMatrix.clicked.connect(self.calcsepmatrix_clicked)
        self.pushButton_EMGBrowse.clicked.connect(self.emgbrowse_clicked)
        self.pushButton_Decompose.clicked.connect(self.decompose_clicked)
        self.pushButton_Plot.clicked.connect(self.plot_clicked)
        self.pushButton_Stop.clicked.connect(self.stop_clicked)

        self.actionExit.triggered.connect(exit)


    def get_simulated_data(self):           # self checkBox_SimulatedData.toggled
        if self.checkBox_SimulatedData.isChecked():
            self.lineEdit_EMGFile.setEnabled(True)
            self.pushButton_EMGBrowse.setEnabled(True)
            self.lineEdit_StartTime.setEnabled(True)
            self.lineEdit_EndTime.setEnabled(True)
            # self.radioButton_FirstArray.setCheckable(False)
            # self.radioButton_SecondArray.setCheckable(False)
            # self.radioButton_AllArrays.setCheckable(False)
        else:
            self.lineEdit_EMGFile.setEnabled(False)
            self.lineEdit_EMGFile.setText("")
            self.pushButton_EMGBrowse.setEnabled(False)
            self.lineEdit_StartTime.setEnabled(False)
            self.lineEdit_StartTime.setText("")
            self.lineEdit_EndTime.setEnabled(False)
            self.lineEdit_EndTime.setText("")
            # self.radioButton_FirstArray.setCheckable(True)
            # self.radioButton_SecondArray.setCheckable(True)
            # self.radioButton_AllArrays.setCheckable(True)

    def get_bandpass_filter(self):          # self.checkBox_BandpassFilter.toggled
        if self.checkBox_BandpassFilter.isChecked():
            self.lineEdit_lowcutFreq.setEnabled(True)
            self.lineEdit_highcutFreq.setEnabled(True)
            self.lineEdit_order.setEnabled(True)
        else:
            self.lineEdit_lowcutFreq.setEnabled(False)
            self.lineEdit_lowcutFreq.setText("")
            self.lineEdit_highcutFreq.setEnabled(False)
            self.lineEdit_highcutFreq.setText("")
            self.lineEdit_order.setEnabled(False)
            self.lineEdit_order.setText("")

    def get_sil_threshold(self):            # self.checkBox_SIL.toggled
        if self.checkBox_SIL.isChecked():
            self.lineEdit_SILThreshold.setEnabled(True)
        else:
            self.lineEdit_SILThreshold.setEnabled(False)

    def get_pps_threshold(self):            # self.checkBox_pps.toggled
        if self.checkBox_pps.isChecked():
            self.lineEdit_ppsThreshold.setEnabled(True)
        else:
            self.lineEdit_ppsThreshold.setEnabled(False)

    def tmodbrowse_clicked(self):           # self.pushButton_tmodBrowse.clicked
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.lineEdit_tmodFile.setText(fname[0]) 


    def showmu_clicked(self):               # self.pushButton_ShowMU.clicked
        self.tableWidget_MotorUnits.setColumnCount(0)
        self.tableWidget_MotorUnits.setRowCount(0)
        # Training module
        ##  Trained module selected
        if self.lineEdit_tmodFile.text() != "":
            self.tmod_file = self.lineEdit_tmodFile.text()
        if self.tmod_file.endswith(".obj"):
            with open(self.tmod_file, 'rb') as f: self.tmod = pickle.load(f)
            """
            data_tmp = loadmat(self.emg_file)['SIG']
            # EMG channels saved in grid of shape (13, 5) or (8, 8, n_samples)
            if ((data_tmp[0][0].size == 0 or 
                 data_tmp[12][0].size == 0) and data_tmp.ndim == 2) or data_tmp.ndim == 3:
                data_tmp = flatten_signal(data_tmp)
            """
        # TODO Load .mat file
        elif self.tmod_file.endswith(".mat"):
            self.tmod = loadmat(self.tmod_file)
        
        ## Sampling frequency
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        else:
            self.fs = self.tmod['fsamp']
        ##  Discarded channels
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            discard_ch = self.lineEdit_DiscardChannels.text().removeprefix("[").removesuffix("]")
            if discard_ch != "":
                self.discard_ch = np.array(list(map(int, discard_ch.split(" "))))
            else:
                self.discard_ch = None
        else:
            if "discard_ch" in self.tmod:
                self.discard_ch = self.tmod["discard_ch"]
            else:
                self.discard_ch = self.tmod["discarded_channels"]
        ##  Extension factor
        if self.lineEdit_ExtFactor.text() != "":
            self.ext_factor = int(self.lineEdit_ExtFactor.text())
        else:
            self.ext_factor = self.tmod["ext_factor"]
        
        # Showing trained module parameters
        self.lineEdit_tmodFile.setText(self.tmod_file)
        self.lineEdit_SamplingFreq.setText(str(self.fs))
        if self.lineEdit_DiscardChannels.text() == "":
            self.lineEdit_DiscardChannels.setText(str(self.discard_ch))
        else:
            self.lineEdit_DiscardChannels.setText(self.lineEdit_DiscardChannels.text())
        self.lineEdit_ExtFactor.setText(str(self.ext_factor))
        
        # Showing MUs extracted by the trained module
        self.B_tmod = self.tmod["B"]
        self.data_tmod = self.tmod["SIG"]
        self.n_MU = self.B_tmod.shape[1]
        self.n_rows_max = 10
        self.n_rows = self.n_MU
        if self.n_MU > self.n_rows_max:
            self.n_rows = self.n_rows_max
        self.n_cols = divmod(self.n_MU, self.n_rows)[0]
        if divmod(self.n_MU, self.n_rows)[1] > 0:
            self.n_cols = divmod(self.n_MU, self.n_rows)[0] + 1
        self.tableWidget_MotorUnits.setColumnCount(self.n_cols)
        self.tableWidget_MotorUnits.setRowCount(self.n_rows)
        for MU in range(self.n_MU):
            item = QTableWidgetItem(f"MU{MU}")
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked)
            row = divmod(MU, self.n_rows)[1]
            col = divmod(MU, self.n_rows)[0]
            self.tableWidget_MotorUnits.setItem(row, col, item)
        self.pushButton_SelectAll.setEnabled(True)
        self.pushButton_CalcSepMatrix.setEnabled(True)


    def selectall_clicked(self):            # self.pushButton_SelectAll.clicked
        self.retrieve_unchecked_mu()
        if np.size(self.unchecked_MUs) == 0:    # Deselecting all MUs
            for MU in range(self.n_MU):
                item = QTableWidgetItem(f"MU{MU}")
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked)
                row = divmod(MU, self.n_rows)[1]
                col = divmod(MU, self.n_rows)[0]
                self.tableWidget_MotorUnits.setItem(row, col, item)
        else:                                   # Selecting all MUs
            for MU in range(self.n_MU):
                item = QTableWidgetItem(f"MU{MU}")
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Checked)
                row = divmod(MU, self.n_rows)[1]
                col = divmod(MU, self.n_rows)[0]
                self.tableWidget_MotorUnits.setItem(row, col, item)

    def retrieve_unchecked_mu(self):
        self.unchecked_MUs = []
        for MU in range(self.n_MU):
            row = divmod(MU, self.n_rows)[1]
            col = divmod(MU, self.n_rows)[0]
            if self.tableWidget_MotorUnits.item(row,col).checkState() == Qt.CheckState.Unchecked:
                self.unchecked_MUs.append(MU)
        self.unchecked_MUs = np.array(self.unchecked_MUs)


    def calcsepmatrix_clicked(self):        # self.pushButton_CalcSepMatrix.clicked
        self.retrieve_checked_mu()
        # Calculating separation matrix for realtime decomposition 
        if np.size(self.checked_MUs) != 0:
            self.B_tmod_selected = self.B_tmod[:, self.checked_MUs]
        else:
            self.B_tmod_selected = np.array([])
        print("Calculating...")
        print(f"ext_factor: {self.ext_factor}")
        self.B_rt_selected, self.mean_tmod = sep_realtime(x=self.data_tmod, 
                                                        B=self.B_tmod_selected,
                                                        discard=self.discard_ch,
                                                        R=self.ext_factor)
        # Length of training data, in samples
        if ((self.data_tmod[0][0].size == 0 or 
             self.data_tmod[12][0].size == 0) and self.data_tmod.ndim == 2) or self.data_tmod.ndim == 3:
            self.tmod_duration = flatten_signal(self.data_tmod).shape[1] / self.fs
        else:

            self.tmod_duration = self.data_tmod.shape[1] / self.fs
        print(f"Training data duration: {self.tmod_duration}")
        
        # Calculating signal and noise centroids
        _, self.sc_tmod, self.nc_tmod  = rt_decomp(data=self.data_tmod, B_realtime=self.B_rt_selected, 
                                                   mean_tm=self.mean_tmod, discard = self.discard_ch, R=self.ext_factor, 
                                                   bandpass=self.use_bpf, 
                                                   lowcut=self.lowcut_freq, highcut=self.highcut_freq, order=self.order,
                                                   l=self.min_distance, 
                                                   classify_mu=self.use_sil, thd_sil=0.85, 
                                                   use_pps=False, thd_pps=self.pps_threshold, 
                                                   sc_tm=None, nc_tm=None, fs = self.fs)
        
        print(f"Separation matrix: {self.B_rt_selected.shape}")
        print(f"Signal centroid: {self.sc_tmod.shape}")
        print(f"Noise centroid: {self.nc_tmod.shape}")
        print(f"Selected MUs: {len(self.checked_MUs)}")
        
    def retrieve_checked_mu(self):
        self.checked_MUs = []
        for MU in range(self.n_MU):
            row = divmod(MU, self.n_rows)[1]
            col = divmod(MU, self.n_rows)[0]
            if self.tableWidget_MotorUnits.item(row,col).checkState() == Qt.CheckState.Checked:
                self.checked_MUs.append(MU)
        self.checked_MUs = np.array(self.checked_MUs)


    def emgbrowse_clicked(self):            # self.pushButton_EMGBrowse.clicked
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.lineEdit_EMGFile.setText(fname[0])

#############################################################################################################################
### Decomposition module  ###################################################################################################
#############################################################################################################################
    def decompose_clicked(self):            # self.pushButton_Decompose.clicked
        self.use_plots = False
        self.pushButton_Plot.setEnabled(False)
        self.pushButton_Decompose.setEnabled(False)
        self.pushButton_Stop.setEnabled(True)

        self.acquire_parameters()
        path_pt = "dec_data\\" + self.save_foldername
        abs_path_pt = os.path.join(abs_path, path_pt)
        # Checking whether the specified path exists or not
        if not os.path.exists(abs_path_pt):
            # Creating a new directory if it does not exist
            os.makedirs(abs_path_pt)

        self.show_parameters()
        self.init_decomp()
        if self.use_arduino:
            self.init_arduino()

        # 1. Using simulated data:
        if self.use_simulated_data == True:    
            # Running decomposition
            self.visTimer.timeout.connect(self.decomp_sim)
            self.visTimer.start(int((self.update_interval) * 1000))
            self.is_recording = True
            # self.is_recording_sim = True
            
            # self.record_emgsim()
        else:
        # 2. Using realtime HD sEMG data:
            # Initialization
            self.rec_data_live = []
            # Connecting to device:
            TCP_PORT = 54320
            HOSTNAME = '192.168.76.1'
            self.muovi = MuoviApp(TCP_PORT, HOSTNAME)
            try:
                self.muovi.connect()

                def signal_handler(signal, frame):
                    print('You pressed Ctrl+C!')
                    self.muovi.close()
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)  # interrupt from keyboard (CRTL+C)
                signal.signal(signal.SIGTERM, signal_handler)  # termination signal
                signal.signal(signal.SIGILL, signal_handler)  # illegal instruction
                signal.signal(signal.SIGABRT, signal_handler)  # Abort signal from abort(3)
            except:
                pass
            
            # Running decomposition
            self.visTimer.timeout.connect(self.decomp_live)
            self.is_recording = True
            # self.is_recording_live = True
            self.record_emglive()

        if self.use_arduino:
            self.record_force()

        self.n_updates = 0
        self.start_time_plot = time.time()
    
    def acquire_parameters(self):
        ## Simulated data
        self.use_simulated_data = self.checkBox_SimulatedData.isChecked()
        if self.lineEdit_EMGFile.text() != "":
            self.emg_file = self.lineEdit_EMGFile.text()
        ### Start and End time
        if self.lineEdit_StartTime.text() != "":
            self.start_time = float(self.lineEdit_StartTime.text())
        if self.lineEdit_EndTime.text() != "":
            self.end_time = float(self.lineEdit_EndTime.text())
        ### Selected simulated data
        if self.emg_file.endswith(".mat"):
            data_tmp = loadmat(self.emg_file)['SIG']
            # EMG channels saved in grid of shape (13, 5) or (8, 8, n_samples)
            if ((data_tmp[0][0].size == 0 or 
                 data_tmp[12][0].size == 0) and data_tmp.ndim == 2) or data_tmp.ndim == 3:
                data_tmp = flatten_signal(data_tmp)
        elif self.emg_file.endswith(".csv"):
            data_tmp = np.loadtxt(self.emg_file, delimiter=',')
            data_tmp = data_tmp[1:, :].T
        # Changing self.end_time if the input value is > than the length of recorded data
        if (self.end_time > data_tmp.shape[1]/self.fs):
            self.end_time = data_tmp.shape[1]/self.fs        
            self.emgdata_rt = crop_data(data=data_tmp, 
                                        start=self.start_time, end=self.end_time, 
                                        fs=self.fs)
        """
        if self.emg_file.endswith(".mat"):
            if (self.end_time > flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs):
                self.end_time = flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs        
            self.emgdata_rt = flatten_signal(crop_data(data=loadmat(self.emg_file)['SIG'], 
                                                       start=self.start_time, end=self.end_time,
                                                       fs=self.fs))
        elif self.emg_file.endswith(".csv"):
            data_tmp = np.loadtxt(self.emg_file, delimiter=',')
            data_tmp = data_tmp[1:, :].T
            if (self.end_time > data_tmp.shape[1]/self.fs):
                self.end_time = data_tmp.shape[1]/self.fs        
            self.emgdata_rt = crop_data(data=data_tmp, 
                                        start=self.start_time, end=self.end_time, 
                                        fs=self.fs)
        """

        ## Selected array for live EMG data
        if self.radioButton_FirstArray.isChecked():
            self.sel_array = 1
            self.grid_size = (8, 8)
        elif self.radioButton_SecondArray.isChecked():
            self.sel_array = 2
            self.grid_size = (8, 8)
        elif self.radioButton_AllArrays.isChecked():
            self.sel_array = 1
            self.grid_size = (16, 8)

        ## Using Arduino board
        if self.checkBox_UseArduino.isChecked():
            self.use_arduino = True
        else:
            self.use_arduino = False        

        ## Batch size
        if self.lineEdit_BatchSize.text() != "":
            self.batch_size = float(self.lineEdit_BatchSize.text())
        ## Overlap
        if self.lineEdit_Overlap.text() != "":
            self.overlap = float(self.lineEdit_Overlap.text())
        
        ##  SIL
        if self.checkBox_SIL.isChecked():
            self.use_sil = True
            if self.lineEdit_SILThreshold.text() != "":
                self.sil_threshold = float(self.lineEdit_SILThreshold.text())
        else:
            self.use_sil = False
        
        ##  Pulse per second
        if self.checkBox_pps.isChecked():
            self.use_pps = True
            if self.lineEdit_ppsThreshold.text() != "":
                self.pps_threshold = float(self.lineEdit_ppsThreshold.text())
        else:
            self.use_pps = False
        
        ## Display interval
        self.display_interval = float(self.comboBox_DisplayInterval.currentText())
        self.update_interval = self.batch_size - self.overlap  
        self.max_nsamples = int(self.display_interval * self.fs) 

        ##  Minimum distance between peaks
        if self.lineEdit_MinDistance.text() != "":
            self.min_distance = int(self.lineEdit_MinDistance.text())
        
        ##  Bandpass filter
        if self.checkBox_BandpassFilter.isChecked():
            self.use_bpf = True
            if self.lineEdit_lowcutFreq.text() != "":
                self.lowcut_freq = float(self.lineEdit_lowcutFreq.text())
            if self.lineEdit_highcutFreq.text() != "":
                self.highcut_freq = float(self.lineEdit_highcutFreq.text())
            if self.lineEdit_order.text() != "":
                self.order = int(self.lineEdit_order.text())
        else:
            self.use_bpf = False
        if self.use_bpf:
            self.bpf_num, self.bpf_den = butter(N=self.order, 
                                                Wn=[self.lowcut_freq, self.highcut_freq], 
                                                fs=self.fs, btype="band")
        

        ##  Save file name
        if self.lineEdit_Name.text() != "":
            self.save_foldername = self.lineEdit_Name.text()
            self.save_filename_pt = self.save_foldername + "_pt.mat"
            self.save_filename_emg = self.save_foldername + "_emg.csv"
            self.save_filename_param = self.save_foldername + "_param.mat"

        # Acquiring parameters for the training module and realtime decomposition
        discard_ch = self.discard_ch
        if self.discard_ch is None:
            discard_ch = []
        self.parameters = {"tmod_file"          : self.tmod_file,
                           "fs"                 : self.fs,
                           "discard_ch"         : discard_ch,
                           "ext_factor"         : self.ext_factor,
                           "use_simulated_data" : self.use_simulated_data,
                           "emg_file"           : self.emg_file,
                           "start_time"         : self.start_time,
                           "end_time"           : self.end_time, 
                           "sel_array"          : self.sel_array,
                           "grid_size"          : self.grid_size,
                           "batch_size"         : self.batch_size,
                           "overlap"            : self.overlap,
                           "use_sil"            : self.use_sil,
                           "sil_threshold"      : self.sil_threshold,
                           "use_pps"            : self.use_pps,
                           "pps_threshold"      : self.pps_threshold,
                           "min_distance"       : self.min_distance,
                           "use_bpf"            : self.use_bpf,
                           "lowcut_freq"        : self.lowcut_freq,
                           "highcut_freq"       : self.highcut_freq,
                           "order"              : self.order,
                           "save_foldername"    : self.save_foldername,
                           "checked_MUs"        : self.checked_MUs }    
        
    def show_parameters(self):
        # Showing parameters        
        self.lineEdit_tmodFile.setText(str(self.parameters["tmod_file"]))
        self.lineEdit_SamplingFreq.setText(str(self.parameters["fs"]))
        self.lineEdit_DiscardChannels.setText(str(self.parameters["discard_ch"]))
        self.lineEdit_ExtFactor.setText(str(self.parameters["ext_factor"]))
        if self.use_simulated_data:
            self.lineEdit_EMGFile.setText(str(self.parameters["emg_file"]))
            self.lineEdit_StartTime.setText(str(self.parameters["start_time"]))
            self.lineEdit_EndTime.setText(str(self.parameters["end_time"]))
        self.lineEdit_BatchSize.setText(str(self.parameters["batch_size"]))
        self.lineEdit_Overlap.setText(str(self.parameters["overlap"]))
        if self.use_sil:
            self.lineEdit_SILThreshold.setText(str(self.parameters["sil_threshold"]))
        if self.use_pps:
            self.lineEdit_ppsThreshold.setText(str(self.parameters["pps_threshold"]))
        self.lineEdit_MinDistance.setText(str(self.parameters["min_distance"]))
        if self.use_bpf:
            self.lineEdit_lowcutFreq.setText(str(self.parameters["lowcut_freq"]))
            self.lineEdit_highcutFreq.setText(str(self.parameters["highcut_freq"]))
            self.lineEdit_order.setText(str(self.parameters["order"]))
        self.lineEdit_Name.setText(str(self.parameters["save_foldername"]))

    def init_decomp(self):
        self.MUPulses = []
        self.time_diff = 0.0

    def init_arduino(self):
        # Detecting arduino board
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description
        ]
        if not arduino_ports:
            raise IOError("No Arduino found")
        if len(arduino_ports) > 1:
            warnings.warn('Multiple Arduinos found - using the first')
        port = arduino_ports[0]

        self.serial_arduino = serial.Serial(port, 9600)

        # Initializing force reading
        self.ref_force = []
    
    def plot_clicked(self):                 # self.pushButton_Plot.clicked
        self.use_plots = True
        self.pushButton_Plot.setEnabled(False)
        self.pushButton_Decompose.setEnabled(False)
        self.pushButton_Stop.setEnabled(True)

        self.acquire_parameters()
        path_pt = "dec_data\\" + self.save_foldername
        abs_path_pt = os.path.join(abs_path, path_pt)
        # Checking whether the specified path exists or not
        if not os.path.exists(abs_path_pt):
            # Creating a new directory if it does not exist
            os.makedirs(abs_path_pt)

        self.show_parameters()
        self.init_plot()
        if self.use_arduino:
            self.init_arduino()

        # 1. Using simulated data:
        if self.use_simulated_data == True:    
            # Updating next plots:
            if self.plotter == "pg":
                self.visTimer.timeout.connect(self.update_plots_sim_pg)
                # self.visTimer.timeout.connect(self.update_plots_sim_pg_multiprocess)
            else:
                self.visTimer.timeout.connect(self.update_plots_sim_plt)
            self.visTimer.start(int((self.update_interval) * 1000))
            self.is_recording = True
            # self.is_recording_sim = True
            
            # self.record_emgsim()
        else:
        # 2. Using realtime HD sEMG data:
            # Initialization
            self.rec_data_live = []
            # Connecting to device:
            TCP_PORT = 54320
            HOSTNAME = '192.168.76.1'
            self.muovi = MuoviApp(TCP_PORT, HOSTNAME)
            try:
                self.muovi.connect()

                def signal_handler(signal, frame):
                    print('You pressed Ctrl+C!')
                    self.muovi.close()
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)  # interrupt from keyboard (CRTL+C)
                signal.signal(signal.SIGTERM, signal_handler)  # termination signal
                signal.signal(signal.SIGILL, signal_handler)  # illegal instruction
                signal.signal(signal.SIGABRT, signal_handler)  # Abort signal from abort(3)
            except:
                pass
            
            # Updating next plots:
            self.visTimer.timeout.connect(self.update_plots_live)
            self.is_recording = True
            # self.is_recording_live = True
            self.record_emglive()
        if self.use_arduino:
            self.record_force()    
        self.n_updates = 0
        self.start_time_plot = time.time()

    def init_plot(self):
        self.plotter = "pg"
        self.n_rows = len(self.checked_MUs)
        self.MUPulses = []
        self.plot_line = []
        self.current_pt = np.zeros((self.n_rows, self.max_nsamples), dtype="int")
        self.x_axis = np.arange(0, self.max_nsamples, dtype="float")
        self.time_axis = self.x_axis/self.fs
        self.time_diff = 0.0
        
        # Clear plots if widget_Decomposition has been used
        if self.widget_Decomposition.layout() is None:
            self.lay = QtWidgets.QVBoxLayout(self.widget_Decomposition)
        else:
            self.lay.removeWidget(self.plotWidget)
            self.plotWidget.setParent(None)
            # self.lay.addWidget(self.plotWidget)
            
            # self.lay = QtWidgets.QVBoxLayout(self.widget_Decomposition)
            

        # Initializing plot
        if self.plotter == "pg":        
            # self.line_handler = collections.deque(maxlen = self.max_nsamples)
            self.line_handler = [None for i in range(self.n_rows)]
            
            self.plotWidget = pg.GraphicsLayoutWidget()
            self.lay.addWidget(self.plotWidget)
            for i in range(self.n_rows):
                self.plot_handler = self.plotWidget.addPlot()
                self.plot_handler.setYRange(0, 1)
                self.plot_handler.setLabel('left', f"MU {self.checked_MUs[i]}")
                self.plotWidget.nextRow()
                self.line_handler[i] = self.plot_handler.plot(pen=pg.mkPen(i))
                self.line_handler[i].setData(self.time_axis, self.current_pt[i])
            
        else:
            self.height_ratio = np.ones(self.n_rows)
            self.plot_fig, self.plot_ax = plt.subplots(self.n_rows, 1,
                                                    gridspec_kw={'height_ratios': self.height_ratio})
            
            for i in range(self.n_rows):
                line, = self.plot_ax[i].plot(self.time_axis, self.current_pt[i])
                self.plot_line.append(line)
                self.plot_ax[i].set_ylim([0, 1])
                self.plot_ax[i].set_ylabel(f"MU {self.checked_MUs[i]}", fontsize=6)
                self.plot_ax[i].tick_params(axis='x', labelsize=6)
                self.plot_ax[i].tick_params(axis='y', labelsize=6)
            # plt.show(block=False)
            plt.subplots_adjust(left=0.05, right=1, top=0.99, bottom=0.02, 
                                hspace=0.2, wspace=0)
            
            self.plotWidget = FigureCanvasQTAgg(self.plot_fig)
            self.lay.setContentsMargins(0, 0, 0, 0)      
            self.lay.addWidget(self.plotWidget)
   
    #############################################################################################################################
    ### Using simulated data  ###################################################################################################
    #############################################################################################################################
    def decomp_sim(self):
        self.prev_timediff = self.time_diff
        self.time_diff = time.time() - self.start_time_plot
        self.pullEMG_sim()
        print(self.time_diff)
        # if self.is_recording_sim == True:
        if self.is_recording == True:
            if (self.time_diff >= self.batch_size):
                if self.use_bpf:
                    filtered_data = np.apply_along_axis(apply_filter,
                                                        axis=1,
                                                        arr=self.current_batch,
                                                        b=self.bpf_num,
                                                        a=self.bpf_den)
                else:
                    filtered_data = self.current_batch
                self.MUPulses, self.n_updates = rt_decomp_live_cy(data = filtered_data, B_realtime = self.B_rt_selected, batch_size = self.batch_size, 
                                                               n_updates = self.n_updates, prev_timediff = self.prev_timediff, time_diff = self.time_diff, 
                                                               prev_MUPulses = self.MUPulses, 
                                                               mean_tm = self.mean_tmod, discard = self.discard_ch, R = self.ext_factor, 
                                                               l = self.min_distance, 
                                                               classify_mu = self.use_sil, thd_sil = self.sil_threshold, 
                                                               use_pps = self.use_pps, thd_pps = self.pps_threshold, 
                                                               sc_tm = self.sc_tmod, nc_tm = self.nc_tmod, fs = self.fs)
        else:
            self.stop_clicked()
        return

    
    def update_plots_sim_pg(self) -> None:
        self.prev_timediff = self.time_diff
        self.time_diff = time.time() - self.start_time_plot
        self.pullEMG_sim()
        print(self.time_diff)
        # if self.is_recording_sim == True:
        if self.is_recording == True:
            if (self.time_diff >= self.batch_size):
                if self.use_bpf:
                    filtered_data = np.apply_along_axis(apply_filter,
                                                        axis=1,
                                                        arr=self.current_batch,
                                                        b=self.bpf_num,
                                                        a=self.bpf_den)
                else:
                    filtered_data = self.current_batch
                (self.MUPulses, 
                self.line_handler, 
                self.n_updates, 
                self.current_pt) = rt_decomp_plotpg_cy(data = filtered_data, B_realtime = self.B_rt_selected, 
                                                batch_size = self.batch_size, n_updates = self.n_updates, 
                                                prev_timediff = self.prev_timediff, time_diff = self.time_diff, max_nsamples = self.max_nsamples, 
                                                current_pt = self.current_pt,
                                                line_handler = self.line_handler, time_axis = self.time_axis,
                                                prev_MUPulses = self.MUPulses, 
                                                mean_tm = self.mean_tmod, discard = self.discard_ch, R = self.ext_factor, 
                                                l = self.min_distance, 
                                                classify_mu = self.use_sil, thd_sil = self.sil_threshold, 
                                                use_pps = self.use_pps, thd_pps = self.pps_threshold, 
                                                sc_tm = self.sc_tmod, nc_tm = self.nc_tmod, fs = self.fs)
        else:
            self.stop_clicked()
        return
    

    def update_plots_sim_plt(self) -> None:
        self.prev_timediff = self.time_diff
        self.time_diff = time.time() - self.start_time_plot
        self.pullEMG_sim()
        print(self.time_diff)
        # if self.is_recording_sim == True:
        if self.is_recording == True:
            if (self.time_diff >= self.batch_size):
                (self.MUPulses, 
                self.plot_line, 
                self.plot_ax, 
                self.n_updates, 
                self.current_pt) = sim_decomp_plot(data = self.current_batch, B_realtime = self.B_rt_selected, 
                                                batch_size = self.batch_size, n_updates = self.n_updates, 
                                                prev_timediff = self.prev_timediff, overlap = self.overlap, 
                                                current_pt = self.current_pt, 
                                                plot_ax = self.plot_ax, plot_line = self.plot_line, 
                                                prev_MUPulses = self.MUPulses, 
                                                mean_tm = self.mean_tmod, discard = self.discard_ch, R = self.ext_factor, 
                                                bandpass = self.use_bpf, 
                                                lowcut = self.lowcut_freq, highcut = self.highcut_freq, order = self.order, 
                                                l = self.min_distance, 
                                                classify_mu = self.use_sil, thd_sil = self.sil_threshold, 
                                                use_pps = self.use_pps, thd_pps = self.pps_threshold, 
                                                sc_tm = self.sc_tmod, nc_tm = self.nc_tmod, fs = self.fs)
                self.plotWidget.update()
                self.plotWidget.flush_events()
        else:
            self.stop_clicked()
        return
    
    

    
    """
    def record_emgsim(self):
        emgsim_pulling = Worker(self.pullEMG_sim)
        emgsim_pulling.signals.result.connect(emgsim_pulling.print_output)
        emgsim_pulling.signals.progress.connect(emgsim_pulling.progress_fn)
        self.threadpool.start(self.pullEMG_sim)
    """    
    def pullEMG_sim(self):
        # while self.is_recording_sim:
        if ((self.prev_timediff - self.update_interval) > (self.end_time - self.start_time)):
            self.is_recording = False
            # self.is_recording_sim = False
        else:
            if self.n_updates > 0:
                start_crop = self.time_diff - self.batch_size
                end_crop = self.time_diff
            else:
                start_crop = 0.0
                end_crop = self.time_diff
            if start_crop < 0.0:
                start_crop = 0.0
            """
            start_crop = self.n_count / self.fs
            end_crop = (self.n_count / self.fs) + self.batch_size
            """
            rec_data = crop_data(data=self.emgdata_rt, 
                                 start=start_crop, end=end_crop,
                                 fs=self.fs)
            
            if self.n_updates == 0: 
                self.current_batch = (rec_data[(self.grid_size[0]*self.grid_size[1])*(self.sel_array-1) : 
                                               (self.grid_size[0]*self.grid_size[1])*self.sel_array, : ])
                self.rec_data = self.current_batch
                # print(f"n={self.n_updates}")
                # time.sleep(self.batch_size)
            else:
                self.current_batch = rec_data
                if ((self.current_batch[0][0].size == 0 or
                     self.current_batch[12][0].size == 0) and self.current_batch.ndim == 2) or self.current_batch.ndim == 3:
                    current_batch_size = flatten_signal(self.current_batch).shape[1]
                else:
                    current_batch_size = self.current_batch.shape[1]
                self.rec_data = np.column_stack([self.rec_data, rec_data[(self.grid_size[0]*self.grid_size[1])*(self.sel_array-1) : 
                                                                         (self.grid_size[0]*self.grid_size[1])*self.sel_array, 
                                                                         int(current_batch_size - 
                                                                             int((self.time_diff - self.prev_timediff)*self.fs)):]])
                
                # self.rec_data = np.column_stack([self.rec_data, rec_data[:,int(self.overlap*self.fs):]])
                # time.sleep(self.update_interval)
                # print(f"n={self.n_updates}")
                # print(current_batch_size)
                # print(int((self.time_diff - self.prev_timediff)*self.fs))
                # print(self.rec_data.shape)
            # self.n_count = self.rec_data.shape[1] - int(self.overlap * self.fs)
            # print(f"n_count:{self.n_count}")
            # print(time.time() - self.start_time_plot)

    def record_force(self):
        force_pulling = Worker(self.pullforce_live)
        force_pulling.signals.result.connect(force_pulling.print_output)
        force_pulling.signals.progress.connect(force_pulling.progress_fn)
        # self.threadpool.start(self.pullforce_live)
        self.threadpool.start(force_pulling)
        
    def pullforce_live(self, progress_callback=None):
        # while self.is_recording_live:
        while self.is_recording:
            b = self.serial_arduino.readline()      # read a byte string
            string_n = b.decode()                   # decode byte string into Unicode  
            string = string_n.rstrip()              # remove \n and \r
            if string != "":
                flt = float(string)                 # convert string to float
                # print(flt)
                self.ref_force.append(flt)          # add to the end of data list


    #############################################################################################################################
    ### Using live EMG data  ####################################################################################################
    #############################################################################################################################
    def decomp_live(self):
        self.prev_timediff = self.time_diff
        self.time_diff = time.time() - self.start_time_plot
        print(self.time_diff)
        if (self.time_diff >= self.batch_size):
            self.rec_dataArray_live = np.array(self.rec_data_live)
            self.current_batch = (self.rec_dataArray_live[-int(self.batch_size*self.fs): , 
                                                        (self.grid_size[0]*self.grid_size[1])*(self.sel_array-1) : 
                                                        (self.grid_size[0]*self.grid_size[1])*self.sel_array].T)
            
            if self.use_bpf:
                filtered_data = np.apply_along_axis(apply_filter,
                                                        axis=1,
                                                        arr=self.current_batch,
                                                        b=self.bpf_num,
                                                        a=self.bpf_den)
            else:
                filtered_data = self.current_batch
            self.MUPulses, self.n_updates = rt_decomp_live_cy(data = filtered_data, B_realtime = self.B_rt_selected, batch_size = self.batch_size, 
                                                            n_updates = self.n_updates, prev_timediff = self.prev_timediff, time_diff = self.time_diff, 
                                                            prev_MUPulses = self.MUPulses, 
                                                            mean_tm = self.mean_tmod, discard = self.discard_ch, R = self.ext_factor, 
                                                            l = self.min_distance, 
                                                            classify_mu = self.use_sil, thd_sil = self.sil_threshold, 
                                                            use_pps = self.use_pps, thd_pps = self.pps_threshold, 
                                                            sc_tm = self.sc_tmod, nc_tm = self.nc_tmod, fs = self.fs)
        return

    
    def update_plots_live(self) -> None:
        self.prev_timediff = self.time_diff
        self.time_diff = time.time() - self.start_time_plot
        print(self.time_diff)
        if (self.time_diff >= self.batch_size):
            self.rec_dataArray_live = np.array(self.rec_data_live)
            self.current_batch = (self.rec_dataArray_live[-int(self.batch_size*self.fs): , 
                                                          (self.grid_size[0]*self.grid_size[1])*(self.sel_array-1) : 
                                                          (self.grid_size[0]*self.grid_size[1])*self.sel_array].T)
            
            if self.use_bpf:
                filtered_data = np.apply_along_axis(apply_filter,
                                                        axis=1,
                                                        arr=self.current_batch,
                                                        b=self.bpf_num,
                                                        a=self.bpf_den)
            else:
                filtered_data = self.current_batch
            (self.MUPulses, 
            self.line_handler, 
            self.n_updates, 
            self.current_pt) = rt_decomp_plotpg_cy(data = filtered_data, B_realtime = self.B_rt_selected, 
                                                batch_size = self.batch_size, n_updates = self.n_updates, 
                                                prev_timediff = self.prev_timediff, time_diff = self.time_diff, max_nsamples = self.max_nsamples,
                                                current_pt = self.current_pt, 
                                                line_handler = self.line_handler, time_axis = self.time_axis,
                                                prev_MUPulses = self.MUPulses, 
                                                mean_tm = self.mean_tmod, discard = self.discard_ch, R = self.ext_factor, 
                                                l = self.min_distance, 
                                                classify_mu = self.use_sil, thd_sil = self.sil_threshold, 
                                                use_pps = self.use_pps, thd_pps = self.pps_threshold, 
                                                sc_tm = self.sc_tmod, nc_tm = self.nc_tmod, fs = self.fs)
        return

    def record_emglive(self):
        self.muovi.startEMGDataSending()
        self.visTimer.start(int((self.update_interval) * 1000))
        
        emglive_pulling = Worker(self.pullEMG_live)
        emglive_pulling.signals.result.connect(emglive_pulling.print_output)
        emglive_pulling.signals.progress.connect(emglive_pulling.progress_fn)
        # self.threadpool.start(self.pullEMG_live)
        self.threadpool.start(emglive_pulling)
        
    def pullEMG_live(self, progress_callback=None):
        # while self.is_recording_live:
        while self.is_recording:
            data_live = self.muovi.receive_signal(5)
            #print(data)
            if(data_live != None):
                self.data_live.append(np.reshape(data_live[0], (16, 8))) # append both to one [(16,8)] array
                self.data_live.append(np.reshape(data_live[1], (16, 8)))
                self.data_live.append(np.reshape(data_live[2], (16, 8)))
                self.data_live.append(np.reshape(data_live[3], (16, 8)))
                self.data_live.append(np.reshape(data_live[4], (16, 8)))
                
                self.rec_data_live.append(np.reshape(self.data_live[-5], (128)))
                self.rec_data_live.append(np.reshape(self.data_live[-4], (128)))
                self.rec_data_live.append(np.reshape(self.data_live[-3], (128)))
                self.rec_data_live.append(np.reshape(self.data_live[-2], (128)))
                self.rec_data_live.append(np.reshape(self.data_live[-1], (128)))


    def stop_clicked(self):                 # self.pushButton_Stop.clicked
        print(self.time_diff)
        # self.save_parameters()
        self.save_results()
        self.pushButton_Plot.setEnabled(True)
        self.pushButton_Decompose.setEnabled(True)
        self.pushButton_Stop.setEnabled(False)
        self.visTimer.stop()
        self.is_recording = False
        if self.use_plots:
            if self.use_simulated_data:
                # self.is_recording_sim = False
                if self.plotter == "pg":
                    self.visTimer.timeout.disconnect(self.update_plots_sim_pg)
                else:
                    self.visTimer.timeout.disconnect(self.update_plots_sim_plt)
            else:
                self.visTimer.timeout.disconnect(self.update_plots_live)
                # self.is_recording_live = False
                self.muovi.stopEMGDataSending()
                self.muovi.close()
                print(f"self.is_recording_live:{self.is_recording}")
        else:
            if self.use_simulated_data:
                self.visTimer.timeout.disconnect(self.decomp_sim)
                # self.is_recording_sim = False
            else:
                self.visTimer.timeout.disconnect(self.decomp_live)
                # self.is_recording_live = False
                self.muovi.stopEMGDataSending()
                self.muovi.close()
                print(f"self.is_recording_live:{self.is_recording}")
        if self.use_arduino:
            self.serial_arduino.close()
        
    """
    def save_parameters(self):
        # TODO combine with save_results
        self.filename_param =  "dec_data/" + self.save_foldername + "/" + self.save_filename_param
        savemat(self.filename_param, self.parameters)
        print(f"Parameters saved in: {self.filename_param}")
    """

    def save_results(self):
        self.filename_pt = "dec_data/" + self.save_foldername + "/" + self.save_filename_pt
        self.filename_emg = "dec_data/" + self.save_foldername + "/" + self.save_filename_emg
        results = {}
        if self.use_simulated_data:    
            emg_df = pd.DataFrame(self.rec_data)    
            emg_df.to_csv(self.filename_emg, index=False)
            results["SIG"] = self.rec_data            
        else:
            emg_df = pd.DataFrame(self.rec_dataArray_live)    
            emg_df.to_csv(self.filename_emg, index=False)
            results["SIG"] = self.rec_dataArray_live.T
        results["MUPulses"] = self.MUPulses
        if self.use_arduino:
            results["ref_force"] = np.array(self.ref_force)
        
        for key in self.parameters.keys():
            results[key] = self.parameters[key]
        savemat(self.filename_pt, results)
        print(f"Results saved in: {self.filename_pt}")

            