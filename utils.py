import h5py
import torch
import numpy as np
import random
#torch.manual_seed(1234)
import csv

def load_mat(ShapeSpace):
    with h5py.File(ShapeSpace, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

#%%  set device CPU/GPU
def setDevice(overrideGPU = True):
    if(torch.cuda.is_available() and (overrideGPU == False) ):
        device = torch.device("cuda:0")
        print("GPU enabled")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device

#%% Seeding
def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)

