import sys
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from LatticeTO import TopologyOptimizer
import matplotlib.pyplot as plt
from utils import setDevice
import torch
import time

## Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./struct.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

overrideGPU = False
device = setDevice(overrideGPU) 
torch.autograd.set_detect_anomaly(True)

plt.close('all') 
start = time.perf_counter()
data_type = torch.float32
topOpt = TopologyOptimizer(config,data_type)
desiredVolumeFraction = 0.46 #can change
desiredQ = torch.tensor([[0.267, 0.021, 0.0], [0.021, 0.111, 0.0], [0.0, 0.0, 0.005]], dtype = data_type) #can change
topOpt.optimizeDesign(config,desiredVolumeFraction, desiredQ) 
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
print(topOpt.exper_name)
topOpt.plotConvergence() 
