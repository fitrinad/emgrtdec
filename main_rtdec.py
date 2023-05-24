from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout,
                             QLabel, QMessageBox, QTableWidgetItem, QApplication)
from PyQt5.QtCore import Qt, QTimer, QThreadPool
from PyQt5 import uic, QtWidgets
from scipy.io import loadmat
import pickle
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Worker import Worker
import collections

import time
from functions import crop_data
from tmod import *
from decmod import (batch_decomp, batch_decomp_window)


class Window_tmod(QMainWindow):     # Training Module window
    def __init__(self):
        super(Window_tmod, self).__init__()
        uic.loadUi("user_interface/gui_tmod.ui", self)
        self.setWindowTitle("Training Module")
        self.show()
        
        # Default parameter values
        self.emg_file = "./data/experimental_data_raw/gm_10.mat"
        self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.start_time = 0.0
        self.end_time = 68.0
        self.data_tm = crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time, end=self.end_time,
                                 fs=self.fs)
        self.save_filename = "dec_gm10_x62_2tmod.obj"
        self.discard_ch = 62
        self.ext_factor = 16
        self.n_iter = 64
        self.min_distance = 31
        self.use_bpf = True
        self.lowcut_freq = 10.0
        self.highcut_freq = 900.0
        self.order = 6
        self.use_sil = True
        self.sil_threshold = 0.9
        self.max_sep_iter = 10
        self.x_tolerance = 10e-4
        self.max_ref_iter = 10

        self.checkBox_BandpassFilter.toggled.connect(self.get_bandpass_filter)
        self.checkBox_SIL.toggled.connect(self.get_sil_threshold)
        
        self.pushButton_Browse.clicked.connect(self.browse_clicked)
        self.pushButton_Decompose.clicked.connect(self.decompose_clicked)
        self.pushButton_RealtimeDecomposition.clicked.connect(self.decmod_window)

        self.actionExit.triggered.connect(exit)


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

    def browse_clicked(self):           # self.pushButton_Browse.clicked
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.lineEdit_EMGFile.setText(fname[0])

    def decompose_clicked(self):        # self.pushButton_Decompose.clicked
        self.acquire_parameters()
        self.confirm_parameters()


    def acquire_parameters(self):
        self.parameters = []
        
        ##  EMG data
        if self.lineEdit_EMGFile.text() != "":
            self.emg_file = self.lineEdit_EMGFile.text()
        
        self.parameters.append(self.emg_file)

        ## Start and End time
        if self.lineEdit_StartTime.text() != "":
            self.start_time = float(self.lineEdit_StartTime.text())
        self.parameters.append(self.start_time)
        if self.lineEdit_EndTime.text() != "":
            self.end_time = float(self.lineEdit_EndTime.text())
            if (self.end_time > flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs or
                self.emg_file != "./data/experimental_data_raw/gm_10.mat"):
                self.end_time = flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs
        self.data_tm = crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time, end=self.end_time,
                                 fs=self.fs)
        self.parameters.append(self.end_time)

        ##  Save file name
        if self.lineEdit_SaveAs.text() != "":
            self.save_filename = self.lineEdit_SaveAs.text()
            self.save_filename += ".obj"
        self.parameters.append(self.save_filename.removesuffix(".obj"))
        
        ##  Sampling frequency
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        else:
            self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.parameters.append(self.fs)
        
        ##  Discarded channels
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            self.discard_ch = np.array(list(map(int, 
                                           self.lineEdit_DiscardChannels.text().split(","))))
            self.parameters.append(self.lineEdit_DiscardChannels.text())
        else:
            self.parameters.append(self.discard_ch)
        
        ##  Extension factor
        if self.lineEdit_ExtFactor.text() != "":
            self.ext_factor = int(self.lineEdit_ExtFactor.text())
        self.parameters.append(self.ext_factor)
        
        ##  Number of iterations
        if self.lineEdit_Iterations.text() != "":
            self.n_iter = int(self.lineEdit_Iterations.text())
        self.parameters.append(self.n_iter)
        
        ##  Minimum distance between peaks
        if self.lineEdit_MinDistance.text() != "":
            self.min_distance = int(self.lineEdit_MinDistance.text())
        self.parameters.append(self.min_distance)
        
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
        self.parameters.append(self.use_bpf)
        self.parameters.append(self.lowcut_freq)
        self.parameters.append(self.highcut_freq)
        self.parameters.append(self.order)
        
        ##  SIL
        if self.checkBox_SIL.isChecked():
            self.use_sil = True
            if self.lineEdit_SILThreshold.text() != "":
                self.sil_threshold = float(self.lineEdit_SILThreshold.text())
        else:
            self.use_sil = False
        self.parameters.append(self.use_sil)
        self.parameters.append(self.sil_threshold) 
        
        ##  Maximum separation iteration
        if self.lineEdit_MaxSepIter.text() != "":
            self.max_sep_iter = int(self.lineEdit_MaxSepIter.text())
        self.parameters.append(self.max_sep_iter)
        
        ##  Tolerance value for element-wise comparison in separation
        if self.lineEdit_xTolerance.text() != "":
            self.x_tolerance = float(self.lineEdit_xTolerance.text())
        self.parameters.append(self.x_tolerance)
        
        ##  Maximum refinement iteration
        if self.lineEdit_MaxRefIter.text() != "":
            self.max_ref_iter = int(self.lineEdit_MaxRefIter.text())
        self.parameters.append(self.max_ref_iter)

        self.parameters = np.array(self.parameters, dtype="object")    
        
    def confirm_parameters(self):
        # Showing parameters        
        self.lineEdit_EMGFile.setText(str(self.parameters[0]))
        self.lineEdit_StartTime.setText(str(self.parameters[1]))
        self.lineEdit_EndTime.setText(str(self.parameters[2]))
        self.lineEdit_SaveAs.setText(str(self.parameters[3]))
        self.lineEdit_SamplingFreq.setText(str(self.parameters[4]))
        self.lineEdit_DiscardChannels.setText(str(self.parameters[5]))
        self.lineEdit_ExtFactor.setText(str(self.parameters[6]))
        self.lineEdit_Iterations.setText(str(self.parameters[7]))
        self.lineEdit_MinDistance.setText(str(self.parameters[8]))
        self.lineEdit_lowcutFreq.setText(str(self.parameters[10]))
        self.lineEdit_highcutFreq.setText(str(self.parameters[11]))
        self.lineEdit_order.setText(str(self.parameters[12]))
        self.lineEdit_SILThreshold.setText(str(self.parameters[14]))
        self.lineEdit_MaxSepIter.setText(str(self.parameters[15]))
        self.lineEdit_xTolerance.setText(str(self.parameters[16]))
        self.lineEdit_MaxRefIter.setText(str(self.parameters[17]))

        # Showing dialog
        self.prm = ( "Parameters:\n\t" + 
                "EMG file: " + str(self.emg_file) + "\n\t" +
                "\tStart (s): " + str(self.start_time) + "\n\t" +
                "\tEnd (s): " + str(self.end_time) + "\n\t" +
                "Save filename: " + str(self.save_filename) + "\n\t" + 
                "Sampling frequency: " + str(self.fs) + "\n\t" +
                "Discarded channels: " + str(self.discard_ch) + "\n\t" +
                "Extension factor: " + str(self.ext_factor) + "\n\t" + 
                "Number of iterations: " + str(self.n_iter) + "\n\t" + 
                "Minimum distance between peaks: " + str(self.min_distance) + "\n\t" + 
                "Bandpass filter: " + str(self.use_bpf) + "\n\t" + 
                "\tLowcut freq: " + str(self.lowcut_freq) + "\n\t" + 
                "\tHighcut freq: " + str(self.highcut_freq) + "\n\t" + 
                "\tOrder: " + str(self.order) + "\n\t" + 
                "SIL: " + str(self.use_sil) + "\n\t" + 
                "\tSIL threshold: " + str(self.sil_threshold) + "\n\t" + 
                "Maximum separation iteration: " + str(self.max_sep_iter) + "\n\t" + 
                "\tTolerance value: " + str(self.x_tolerance) + "\n\t" + 
                "Maximum refinement iteration: " + str(self.max_ref_iter)
                )

        self.dialog_params = QDialog()
        self.dialog_params.setWindowTitle("Parameters")
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.dialog_params.buttonBox = QDialogButtonBox(QBtn)
        self.dialog_params.buttonBox.accepted.connect(self.decompose_emg)
        self.dialog_params.buttonBox.rejected.connect(self.dialog_params.reject)

        self.dialog_params.layout = QVBoxLayout()
        dlg = QLabel(self.prm)
        self.dialog_params.layout.addWidget(dlg)
        self.dialog_params.layout.addWidget(self.dialog_params.buttonBox)
        self.dialog_params.setLayout(self.dialog_params.layout)

        self.dialog_params.exec_()

    def decompose_emg(self):
        self.dialog_params.close()
        # Decomposing EMG data
        self.decomp_tmod = decomposition_tmod(x=self.data_tm,
                                         discard=self.discard_ch,
                                         R=self.ext_factor,
                                         M=self.n_iter,
                                         bandpass=self.use_bpf,
                                         lowcut=self.lowcut_freq,
                                         highcut=self.highcut_freq,
                                         fs=self.fs,
                                         order=self.order,
                                         Tolx=self.x_tolerance,
                                         contrast_fun=skew,
                                         ortho_fun=gram_schmidt,
                                         max_iter_sep=self.max_sep_iter,
                                         l=self.min_distance,
                                         sil_pnr=self.use_sil,
                                         thresh=self.sil_threshold,
                                         max_iter_ref=self.max_ref_iter,
                                         random_seed=None,
                                         verbose=False)
        # Saving results and parameters
        self.save_results()
        self.save_parameters()        
        self.show_results()

    def save_results(self):
        decomp_sample_pkl = open(self.save_filename, 'wb') 
        pickle.dump(self.decomp_tmod, decomp_sample_pkl)
        decomp_sample_pkl.close()

    def save_parameters(self):
        self.parameter_file = self.lineEdit_SaveAs.text() + "_param" + ".csv"
        np.savetxt(self.parameter_file, self.parameters, delimiter=",", fmt="%s")

    def show_results(self):
        # Showing parameters and number of extracted MUs
        self.message = QMessageBox()
        self.message.setWindowTitle("Decomposition result")
        self.msg =  (self.prm + "\n" + 
                 "\nExtracted MUs: " + str(self.decomp_tmod["B"].shape[1]) )
        self.message.setText(self.msg)
        self.message.exec_()
     

    def decmod_window(self):            # self.pushButton_RealtimeDecomposition.clicked
        # Opens Decomposition Module window
        self.ext_window = Window_decmod()
        self.ext_window.show() 



class Window_decmod(QMainWindow):   # Decomposition Module window
    def __init__(self):
        super(Window_decmod, self).__init__()
        uic.loadUi("user_interface/gui_decmod.ui", self)
        self.setWindowTitle("Decomposition Module")
        self.show()

        # Default trained module
        self.tmod_file = "./dec_gm10_x62_2tmod.obj"
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
        self.stop_time_plot = self.end_time - self.start_time
        self.emgdata_rt = flatten_signal(crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time,
                                 end=self.end_time,
                                 fs=self.fs))

        # Default parameter values
        self.save_foldername = "decomp_rt"
        self.save_filename_pt = self.save_foldername + "_pt.csv"
        self.save_filename_emg = self.save_foldername + "_emg.csv"
        self.batch_size = 4.0
        self.overlap = 3.0

        self.min_distance = 31
        self.use_bpf = True
        self.lowcut_freq = 10.0
        self.highcut_freq = 900.0
        self.order = 6
        self.use_sil = True
        self.sil_threshold = 0.85
        self.use_pps = False
        self.pps_threshold = 5
        
        self.display_interval = 5.0
        self.update_interval = self.batch_size - self.overlap  
        self.max_nsamples = int(self.display_interval * self.fs)
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
        self.pushButton_Plot.clicked.connect(self.plot_clicked)
        self.pushButton_Stop.clicked.connect(self.stop_clicked)

        self.actionExit.triggered.connect(exit)


    def get_simulated_data(self):           # self checkBox_SimulatedData.toggled
        if self.checkBox_SimulatedData.isChecked():
            self.lineEdit_EMGFile.setEnabled(True)
            self.pushButton_EMGBrowse.setEnabled(True)
            self.lineEdit_StartTime.setEnabled(True)
            self.lineEdit_EndTime.setEnabled(True)
        else:
            self.lineEdit_EMGFile.setEnabled(False)
            self.lineEdit_EMGFile.setText("")
            self.pushButton_EMGBrowse.setEnabled(False)
            self.lineEdit_StartTime.setEnabled(False)
            self.lineEdit_StartTime.setText("")
            self.lineEdit_EndTime.setEnabled(False)
            self.lineEdit_EndTime.setText("")

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
        self.parameters = []
        ## Training module
        #   Trained module selected
        if self.lineEdit_tmodFile.text() != "":
            self.tmod_file = self.lineEdit_tmodFile.text()
        with open(self.tmod_file, 'rb') as f: self.tmod = pickle.load(f)

        ## Sampling frequency, Discarded channels, Extension factor
        if self.tmod_file != "./dec_gm10_x62_2tmod.obj":
            self.fs = self.tmod["fsamp"]
            self.discard_ch = self.tmod["discarded_channels"]
            self.ext_factor = self.tmod["ext_factor"]
        # Sampling frequency selected
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        # Discard channels selected
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            self.discard_ch = np.array(list(map(int, 
                                            self.lineEdit_DiscardChannels.text().split(","))))
        # Extension factor selected
        if self.lineEdit_ExtFactor.text() != "":
            self.ext_factor = int(self.lineEdit_ExtFactor.text())
        
        # Showing parameters
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
        self.B_rt_selected, self.mean_tmod = sep_realtime(x=self.data_tmod, 
                                                        B=self.B_tmod_selected,
                                                        discard=self.discard_ch,
                                                        R=self.ext_factor)
        if ((np.size(self.tmod["SIG"][0][0]) == 0 or
             np.size(self.tmod["SIG"][12][0] == 0)) and self.tmod["SIG"].ndim == 2) or self.tmod["SIG"].ndim == 3:
            self.tmod_duration = flatten_signal(self.tmod["SIG"]).shape[1]
        else:
            self.tmod_duration = self.tmod["SIG"].shape[1]
        # Calculating signal and noise centroids
        self.init_decmod = batch_decomp(data=self.data_tmod, 
                                        B_realtime=self.B_rt_selected, 
                                        mean_tm=self.mean_tmod, 
                                        discard = self.discard_ch, 
                                        use_kmeans=True, 
                                        classify_mu=True, 
                                        sil_dist=True, 
                                        thd_sil=0.85, 
                                        sc_tm=None, nc_tm=None, 
                                        batch_size=self.tmod_duration, overlap=0.0)
        self.sc_tmod = self.init_decmod["signal_centroids"][:,0]
        self.nc_tmod = self.init_decmod["noise_centroids"][:,0]
        print(f"Separation matrix: {self.B_rt_selected.shape}")
        print(f"Signal centroid: {self.sc_tmod.shape}")
        print(f"Noise centroid: {self.nc_tmod.shape}")
        print(len(self.checked_MUs))
        
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
    def plot_clicked(self):                 # self.pushButton_Plot.clicked
        self.pushButton_Plot.setEnabled(False)
        self.pushButton_Stop.setEnabled(True)

        self.acquire_parameters()
        self.show_parameters()

        # Initializing plot
        self.visTimer.start((self.batch_size - self.overlap) * 1000)
        self.plotWidget = FigureCanvasQTAgg()
        self.start_time_plot = time.time()
        
        if self.use_simulated_data == True:
        # 1. Using simulated data:
            # First batch:
            #   Creating subplot
            self.n_rows = len(self.checked_MUs)
            self.current_pt = np.zeros((self.n_rows, self.max_nsamples), dtype="int")
            self.height_ratio = np.ones(self.n_rows)        
            fig, ax = plt.subplots(self.n_rows, 1, 
                                   gridspec_kw={'height_ratios': self.height_ratio})
            self.x_axis = np.arange(0, self.max_nsamples, dtype="float")
            self.time_axis = self.x_axis/self.fs
            
            for i in range(self.n_rows):
                ax[i].plot(self.time_axis, self.current_pt[i])
                ax[i].set_ylabel(f"MU {self.checked_MUs[i]}")
            
            print(self.current_pt.shape)
            fig.tight_layout()
            
            self.plotWidget = FigureCanvasQTAgg(fig)
            lay = QtWidgets.QVBoxLayout(self.widget_Decomposition)  
            lay.setContentsMargins(0, 0, 0, 0)      
            lay.addWidget(self.plotWidget)
            
            # Updating next plots:
            self.visTimer.timeout.connect(self.update_plots_sim)
            self.update_plots_sim()
        else:
        # 2. Using realtime HD sEMG data:
            # First batch

            # Updating next plots
            self.visTimer.timeout.connect(self.update_plots_live)
            self.update_plots_live()
            
    
    def acquire_parameters(self):
        self.parameters = []
        
        ## Training module
        if self.lineEdit_tmodFile.text() != "":
            self.tmod_file = self.lineEdit_tmodFile.text()
        self.parameters.append(self.tmod_file)

        ## Sampling frequency
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        else:
            self.fs = loadmat(self.tmod_file)['fsamp'].squeeze()
        self.parameters.append(self.fs)

        ## Discarded channels
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            self.discard_ch = np.array(list(map(int, 
                                           self.lineEdit_DiscardChannels.text().split(","))))
            self.parameters.append(self.lineEdit_DiscardChannels.text())
        else:
            self.parameters.append(self.discard_ch)
        self.n_ch = int(self.B_rt_selected.shape[0] / (self.ext_factor+1))

        # Acquiring parameters for realtime decomposition
        # Simulated data
        self.use_simulated_data = self.checkBox_SimulatedData.isChecked()
        if self.lineEdit_EMGFile.text() != "":
            self.emg_file = self.lineEdit_EMGFile.text()
        if self.lineEdit_StartTime.text() != "":
            self.start_time = float(self.lineEdit_StartTime.text())
        if self.lineEdit_EndTime.text() != "":
            self.end_time = float(self.lineEdit_EndTime.text())
        self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.start_time_plot = 0.0
        self.stop_time_plot = self.end_time - self.start_time
        self.emgdata_rt = flatten_signal(crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time,
                                 end=self.end_time,
                                 fs=self.fs))

        ##  EMG data
        if self.lineEdit_EMGFile.text() != "":
            self.emg_file = self.lineEdit_EMGFile.text()
            self.data_tm = loadmat(self.emg_file)['SIG']
        self.parameters.append(self.emg_file)
        
        ##  Save file name
        if self.lineEdit_Name.text() != "":
            self.save_foldername = self.lineEdit_Name.text()
            self.save_filename_pt = self.save_foldername + "_pt.csv"
            self.save_filename_emg = self.save_foldername + "_emg.csv"
        self.parameters.append(self.save_foldername)
        
        ##  Sampling frequency
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        else:
            self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.parameters.append(self.fs)
        
        ##  Discarded channels
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            self.discard_ch = np.array(list(map(int, 
                                           self.lineEdit_DiscardChannels.text().split(","))))
        self.parameters.append(self.discard_ch)
        
        ##  Extension factor
        if self.lineEdit_ExtFactor.text() != "":
            self.ext_factor = int(self.lineEdit_ExtFactor.text())
        self.parameters.append(self.ext_factor)
        
        ##  Minimum distance between peaks
        if self.lineEdit_MinDistance.text() != "":
            self.min_distance = int(self.lineEdit_MinDistance.text())
        self.parameters.append(self.min_distance)
        
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
        self.parameters.append(self.use_bpf)
        self.parameters.append(self.lowcut_freq)
        self.parameters.append(self.highcut_freq)
        self.parameters.append(self.order)
        
        ##  SIL
        if self.checkBox_SIL.isChecked():
            self.use_sil = True
            if self.lineEdit_SILThreshold.text() != "":
                self.sil_threshold = float(self.lineEdit_SILThreshold.text())
        else:
            self.use_sil = False
        self.parameters.append(self.use_sil)
        self.parameters.append(self.sil_threshold) 

        ## Display interval
        self.display_interval = float(self.comboBox_DisplayInterval.currentText())
        self.update_interval = self.batch_size - self.overlap  
        self.max_nsamples = int(self.display_interval * self.fs)

        
        self.parameters = np.array(self.parameters, dtype="object")    
        
    def show_parameters(self):
        pass

    def update_plots_sim(self):
        self.start_time_plot = time.time()
          
        

    def rtdecomp_emgsim(self):
        # getting simulated data
        self.current_time = time.time()
        self.time_diff = self.current_time - self.start_time_plot
        
        # decomposing current batch


        # Decomposing EMG data
        

    def get_current_simbatch(self):
        self.current_time = time.time()




    def update_plots_live(self):
        pass


    def stop_clicked(self):                 # self.pushButton_Stop.clicked
        self.plotWidget.flush_events()
        # self.save_results()
        self.pushButton_Plot.setEnabled(True)
        self.pushButton_Stop.setEnabled(False)
        


def main():
    app = QApplication([])
    window = Window_tmod()
    app.exec_()

if __name__ == "__main__":
    main()