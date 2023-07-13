from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout,
                             QLabel, QMessageBox, QApplication)
from PyQt5 import QtCore, uic
from scipy.io import (loadmat, savemat)
import pickle
import numpy as np
import pandas as pd
import sys

from functions.preprocessing import (crop_data, bad_channels, notch_filter) 
from functions.visualization import visualize_pt_tmod
from tmod import *
from decmod_window import Window_decmod

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)



class Window_tmod(QMainWindow):     # Training Module window
    def __init__(self):
        super(Window_tmod, self).__init__()
        uic.loadUi("./user_interface/gui_tmod.ui", self)
        self.setWindowTitle("Training Module")
        self.show()
        
        # Default parameter values
        self.emg_file = "./data/experimental_data_raw/gm_10.mat"
        self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        self.sel_array = 1           # Selected electrode array: 1 or 2
        self.grid_size = (8, 8)         # Grid size: (8, 8) or (16, 8)
        self.start_time = 0.0
        self.end_time = 68.0
        self.data_tm = crop_data(data=loadmat(self.emg_file)['SIG'], 
                                 start=self.start_time, end=self.end_time,
                                 fs=self.fs)
        self.save_filename = "dec_gm10_x62_2tmod"
        self.noise_start = 0.0
        self.noise_end = 3.0
        self.snr_threshold = 2.0
        
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

        # Showing default parameters
        self.lineEdit_EMGFile.setText(self.emg_file)
        self.lineEdit_SamplingFreq.setText(str(self.fs))
        self.lineEdit_StartTime.setText(str(self.start_time))
        self.lineEdit_EndTime.setText(str(self.end_time))
        self.lineEdit_SaveAs.setText(self.save_filename)
        self.lineEdit_NoiseStart.setText(str(self.noise_start))
        self.lineEdit_NoiseEnd.setText(str(self.noise_end))
        self.lineEdit_SNRThreshold.setText(str(self.snr_threshold))
        self.lineEdit_DiscardChannels.setText(str(self.discard_ch))
        self.lineEdit_ExtFactor.setText(str(self.ext_factor))
        self.lineEdit_Iterations.setText(str(self.n_iter))
        self.lineEdit_MinDistance.setText(str(self.min_distance))
        self.lineEdit_lowcutFreq.setText(str(self.lowcut_freq))
        self.lineEdit_highcutFreq.setText(str(self.highcut_freq))
        self.lineEdit_order.setText(str(self.order))
        self.lineEdit_SILThreshold.setText(str(self.sil_threshold))
        self.lineEdit_MaxSepIter.setText(str(self.max_sep_iter))
        self.lineEdit_xTolerance.setText(str(self.x_tolerance))
        self.lineEdit_MaxRefIter.setText(str(self.max_ref_iter))

        # Connecting signals
        self.checkBox_BandpassFilter.toggled.connect(self.get_bandpass_filter)
        self.checkBox_SIL.toggled.connect(self.get_sil_threshold)
        
        self.pushButton_Browse.clicked.connect(self.browse_clicked)
        self.pushButton_ShowBadChannels.clicked.connect(self.showbadch_clicked)
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

    def browse_clicked(self):               # self.pushButton_Browse.clicked
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.lineEdit_EMGFile.setText(fname[0])

    def showbadch_clicked(self):            # self.pushButton_ShowBadChannels.clicked
        self.acquire_emgdata()

        # Flattening data, excluding the empty channel, if the channels are in a grid of (13, 5)
        if ((self.data_tm[0][0].size == 0 or
             self.data_tm[12][0].size == 0) and self.data_tm.ndim == 2) or self.data_tm.ndim == 3:
            self.data_tm = flatten_signal(self.data_tm)

        # Denoising
        print("Denoising...")
        self.data_tm = np.apply_along_axis(butter_bandpass_filter,
                                           axis = 1,
                                           arr = self.data_tm,
                                           lowcut = 10,
                                           highcut = 500,
                                           fs = self.fs,
                                           order = 2)
        """
        self.data_tm = np.apply_along_axis(notch_filter,
                                           axis = 1,
                                           arr = self.data_tm,
                                           freq_cutoff = 50,
                                           fs = self.fs,
                                           q_factor = 30)
        """
        # Checking for and showing bad channels
        print("Checking for bad channels...")
        self.discard_ch = bad_channels(data = self.data_tm, signal=self.emg_file, 
                                       thd_snr = self.snr_threshold, 
                                       noise_start = self.noise_start, noise_end = self.noise_end, 
                                       fs=self.fs)
        if len(self.discard_ch) == 0:
            self.discard_ch = None
        self.lineEdit_DiscardChannels.setText(str(self.discard_ch))

        # Calculating and showing extension factor
        if np.all(self.discard_ch) is not None:
            self.ext_factor = int(np.ceil(1000/(self.data_tm.shape[0] - self.discard_ch.size)))
        else:
            self.ext_factor = int(np.ceil(1000/(self.data_tm.shape[0])))
        print(f"Extension factor: {self.ext_factor}")
        self.lineEdit_ExtFactor.setText(str(self.ext_factor))
        
    def acquire_emgdata(self):
        ##  EMG data
        if self.lineEdit_EMGFile.text() != "":
            self.emg_file = self.lineEdit_EMGFile.text()
        
        ##  Sampling frequency
        if self.lineEdit_SamplingFreq.text() != "":
            self.fs = float(self.lineEdit_SamplingFreq.text())
        elif self.emg_file.endswith(".mat"):
            self.fs = loadmat(self.emg_file)['fsamp'].squeeze()
        elif self.emg_file.endswith(".csv"):
            self.fs = 2000
        
        ## Start and End time
        if self.lineEdit_StartTime.text() != "":
            self.start_time = float(self.lineEdit_StartTime.text())
        if self.lineEdit_EndTime.text() != "":
            self.end_time = float(self.lineEdit_EndTime.text())
        
        ## Selected electrode array
        if self.radioButton_FirstArray.isChecked():
            self.sel_array = 1
            self.grid_size = (8, 8)
        elif self.radioButton_SecondArray.isChecked():
            self.sel_array = 2
            self.grid_size = (8, 8)
        elif self.radioButton_AllArrays.isChecked():
            self.sel_array = 1
            self.grid_size = (16, 8)

        if self.emg_file.endswith(".mat"):
            if (self.end_time > flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs or
                self.emg_file != "./data/experimental_data_raw/gm_10.mat"):
                self.end_time = flatten_signal(loadmat(self.emg_file)['SIG']).shape[1]/self.fs        
            self.data_tm = crop_data(data=loadmat(self.emg_file)['SIG'], 
                                     start=self.start_time, end=self.end_time,
                                     fs=self.fs)
        elif self.emg_file.endswith(".csv"):
            """
            data_tmp = np.loadtxt(self.emg_file, delimiter=',')
            # if self.sel_array is not None:
            data_tmp = (data_tmp[1:,  (self.grid_size[0]*self.grid_size[1]) * (self.sel_array-1) :
                                          (self.grid_size[0]*self.grid_size[1]) * self.sel_array].T  )
            """
            data_tmp = np.array( pd.read_csv(self.emg_file) )
            data_tmp = (data_tmp[:,  (self.grid_size[0]*self.grid_size[1]) * (self.sel_array-1) :
                                      (self.grid_size[0]*self.grid_size[1]) * self.sel_array].T  )
            # else:
            #     data_tmp = data_tmp[1:,  :].T
            if (self.end_time > data_tmp.shape[1]/self.fs):
                self.end_time = data_tmp.shape[1]/self.fs        
            self.data_tm = crop_data(data=data_tmp, 
                                     start=self.start_time, end=self.end_time, 
                                     fs=self.fs)

        ## Selected noise and SNR threshold
        if self.lineEdit_NoiseStart.text() != "":
            self.noise_start = float(self.lineEdit_NoiseStart.text())
        if self.lineEdit_NoiseEnd.text() != "":
            self.noise_end = float(self.lineEdit_NoiseEnd.text())
        if self.lineEdit_SNRThreshold.text() != "":
            self.snr_threshold = float(self.lineEdit_SNRThreshold.text())   


    def decompose_clicked(self):            # self.pushButton_Decompose.clicked
        self.acquire_parameters()
        self.confirm_parameters()

    def acquire_parameters(self):
        self.acquire_emgdata()

        ##  Save file name
        if self.lineEdit_SaveAs.text() != "":
            self.save_filename = self.lineEdit_SaveAs.text()

        ##  Discarded channels
        if (self.lineEdit_DiscardChannels.text() != "" and 
            self.lineEdit_DiscardChannels.text() != "None"):
            discard_ch = self.lineEdit_DiscardChannels.text().removeprefix("[").removesuffix("]")
            if discard_ch != "":
                self.discard_ch = np.array(list(map(int, discard_ch.split(" "))))
            else:
                self.discard_ch = None
        elif (self.emg_file != "./data/experimental_data_raw/gm_10.mat"):
            self.discard_ch = None
        
        ##  Extension factor
        if self.lineEdit_ExtFactor.text() != "":
            self.ext_factor = int(self.lineEdit_ExtFactor.text())
        
        ##  Number of iterations
        if self.lineEdit_Iterations.text() != "":
            self.n_iter = int(self.lineEdit_Iterations.text())
        
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
        
        ##  SIL
        if self.checkBox_SIL.isChecked():
            self.use_sil = True
            if self.lineEdit_SILThreshold.text() != "":
                self.sil_threshold = float(self.lineEdit_SILThreshold.text())
        else:
            self.use_sil = False
        
        ##  Maximum separation iteration
        if self.lineEdit_MaxSepIter.text() != "":
            self.max_sep_iter = int(self.lineEdit_MaxSepIter.text())
        
        ##  Tolerance value for element-wise comparison in separation
        if self.lineEdit_xTolerance.text() != "":
            self.x_tolerance = float(self.lineEdit_xTolerance.text())
        
        ##  Maximum refinement iteration
        if self.lineEdit_MaxRefIter.text() != "":
            self.max_ref_iter = int(self.lineEdit_MaxRefIter.text())
        
        discard_ch = self.discard_ch
        if self.discard_ch is None:
            discard_ch = []
        self.parameters = {"emg_file": self.emg_file, 
                           "fs": self.fs,
                           "start_time": self.start_time, 
                           "end_time": self.end_time, 
                           "sel_array": self.sel_array, 
                           "save_filename": self.save_filename, 
                           "noise_start": self.noise_start, 
                           "noise_end": self.noise_end, 
                           "snr_threshold": self.snr_threshold, 
                           "discard_ch": discard_ch, 
                           "ext_factor": self.ext_factor, 
                           "n_iter": self.n_iter, 
                           "min_distance": self.min_distance, 
                           "use_bpf": self.use_bpf, 
                           "lowcut_freq": self.lowcut_freq, 
                           "highcut_freq": self.highcut_freq, 
                           "order": self.order, 
                           "use_sil": self.use_sil, 
                           "sil_threshold": self.sil_threshold, 
                           "max_sep_iter": self.max_sep_iter, 
                           "x_tolerance": self.x_tolerance,
                           "max_ref_iter": self.max_ref_iter
                           }
        
    def confirm_parameters(self):
        # Showing parameters        
        self.lineEdit_EMGFile.setText(str(self.parameters["emg_file"]))
        self.lineEdit_SamplingFreq.setText(str(self.parameters["fs"]))
        self.lineEdit_StartTime.setText(str(self.parameters["start_time"]))
        self.lineEdit_EndTime.setText(str(self.parameters["end_time"]))
        self.lineEdit_SaveAs.setText(str(self.parameters["save_filename"]))
        self.lineEdit_DiscardChannels.setText(str(self.parameters["discard_ch"]))
        self.lineEdit_ExtFactor.setText(str(self.parameters["ext_factor"]))
        self.lineEdit_Iterations.setText(str(self.parameters["n_iter"]))
        self.lineEdit_MinDistance.setText(str(self.parameters["min_distance"]))
        self.lineEdit_lowcutFreq.setText(str(self.parameters["lowcut_freq"]))
        self.lineEdit_highcutFreq.setText(str(self.parameters["highcut_freq"]))
        self.lineEdit_order.setText(str(self.parameters["order"]))
        self.lineEdit_SILThreshold.setText(str(self.parameters["sil_threshold"]))
        self.lineEdit_MaxSepIter.setText(str(self.parameters["max_sep_iter"]))
        self.lineEdit_xTolerance.setText(str(self.parameters["x_tolerance"]))
        self.lineEdit_MaxRefIter.setText(str(self.parameters["max_ref_iter"]))

        # Showing dialog
        self.prm = ( "Parameters:\n\t" + 
                "EMG file: " + str(self.emg_file) + "\n\t" +
                "\tSampling frequency: " + str(self.fs) + "\n\t" +
                "\tStart (s): " + str(self.start_time) + "\n\t" +
                "\tEnd (s): " + str(self.end_time) + "\n\t" +
                "\tSelected array: " + str(self.sel_array) + "\n\t" +
                "\tGrid size: " + str(self.grid_size) + "\n\t" +
                "Save filename: " + str(self.save_filename) + "\n\t" + 
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
        saveresults_path = "./tmod_data/" + self.save_filename + ".obj"
        decomp_sample_pkl = open(saveresults_path, 'wb') 
        pickle.dump(self.decomp_tmod, decomp_sample_pkl)
        decomp_sample_pkl.close()
        print(f"Results saved in: {saveresults_path}")

    def save_parameters(self):
        saveparams_path = "./tmod_data/" + self.save_filename + "_param" + ".mat"
        savemat(saveparams_path, self.parameters)
        # np.savetxt(saveparams_path, self.parameters, delimiter=",", fmt="%s")
        print(f"Parameters saved in: {saveparams_path}")

    def show_results(self):
        # Showing parameters and number of extracted MUs
        self.message = QMessageBox()
        self.message.setWindowTitle("Decomposition result")
        self.msg =  (self.prm + "\n" + 
                 "\nExtracted MUs: " + str(self.decomp_tmod["B"].shape[1]) )
        self.message.setText(self.msg)
        # self.message.exec_()

        # Showing extracted MU pulse trains
        visualize_pt_tmod(ind_pt = self.decomp_tmod["MUPulses"],
                          data = self.data_tm, 
                          fs = self.fs, 
                          title = self.save_filename)
        
     

    def decmod_window(self):                # self.pushButton_RealtimeDecomposition.clicked
        # Opens Decomposition Module window
        self.ext_window = Window_decmod()
        self.ext_window.show() 



def main():
    app = QApplication(sys.argv)
    window = Window_tmod()
    app.exec_()

if __name__ == "__main__":
    main()