import socket
from socket import SHUT_RDWR
import numpy as np
import time
from crc.crc import CrcCalculator, Crc8
from Worker import Worker


class MuoviApp():
    def __init__(self, tcp_port: int, hostname: str):
        self.client = 0

        self.hostname = hostname
        self.tcp_port = tcp_port
        self.START_SIGNAL = [int('00000101', 2), int('01011011', 2), int('01001011', 2)]
        self.STOP_SIGNAL = [int('00000100', 2), int('01011010', 2), int('01001010', 2)]
        self.STOP_TCP = [int('00000000', 2)]
        self.ConvFact = 0.000286  ## conversion to mV
        #self.data = np.ones(128).astype(int)

    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client.settimeout(5)
        ip = socket.gethostbyname(self.hostname)
        address = (ip, self.tcp_port)
        self.client.connect(address)

    def startEMGDataSending(self):
        self.send_signal(self.START_SIGNAL)

    def stopEMGDataSending(self):
        self.send_signal(self.STOP_SIGNAL)

    def send_signal(self, signal):
        packet = bytearray(signal)
        crc_calculator = CrcCalculator(Crc8.MAXIM_DOW)
        packet.append(crc_calculator.calculate_checksum(packet))  # .to_bytes(2, 'big'))  # Add CRC-8
        self.client.send(packet)
        time.sleep(0.3)

    def receive_signal(self, n):
        packet = self.client.recv(292*n)
        int_data = np.array([int.from_bytes(packet[i:(i + 2)], byteorder="big") for i in range(0, len(packet), 2)])
        int_data[np.where(int_data > 32767)] -= 65536
        #int_data = abs(int_data)
        #print("Shape of data: ", int_data.shape)
        if(int_data.shape[0] >= n*(292//2)):
            out_data = []
            for dp in range(n):
                out_data.append(np.concatenate((int_data[list(range(2*dp*64 + 2*dp*9, (2*dp+1)*64 + 2*dp*9))] , int_data[list(range((2*dp+1)*64 + (2*dp+1)*9 - 3, 2*(dp+1)*64 + (2*dp+1)*9 - 3))])))
            return out_data
            # return np.concatenate((int_data[list(range(0, 64))] , int_data[list(range(64 + 6, 2*64 + 6))])),\
            #        np.concatenate((int_data[list(range(2*64 + 2*9, 3*64 + 2*9))] , int_data[list(range(3*64 + 3*9 - 3, 4*64 + 3*9 - 3))])),\
            #        np.concatenate((int_data[list(range(4*64 + 4*9, 5*64 + 4*9))] , int_data[list(range(5*64 + 5*9 - 3, 6*64 + 5*9 - 3))])),\
            #        np.concatenate((int_data[list(range(6*64 + 6*9, 7*64 + 6*9))] , int_data[list(range(7*64 + 7*9 - 3, 8*64 + 7*9 - 3))]))  # *self.ConvFact
        else:
            return None #np.zeros((128)),np.zeros((128))
        # filtered = self.butter_bandpass_filter(data = data, lowcut = 10, highcut = 600, fs = 2000, order = 2)
        # return data#filtered

    def close(self):  #
        # self.send_signal(self.STOP_SIGNAL)
        self.send_signal(self.STOP_TCP)
        self.client.shutdown(SHUT_RDWR)
        self.client.close()
        self.client = 0




