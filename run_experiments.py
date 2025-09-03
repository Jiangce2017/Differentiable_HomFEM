import sys
#import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from LatticeTO import TopologyOptimizer
import matplotlib.pyplot as plt
from utils import setDevice
import torch
import time
import numpy as np

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

#desiredVolumeFraction = 0.46 #can change
#desiredQ = torch.tensor([[0.267, 0.021, 0.0], [0.021, 0.111, 0.0], [0.0, 0.0, 0.005]], dtype = data_type)

# config.desiredVolumeFraction = 0.3 #can change
# config.desiredC11 = 1 ## max 1.0989011
# config.desiredC12 = 0.3 ## max 0.3296703
# config.desiredC22 = 1  ## max 1.0989011
# config.desiredC66 = 0.3 ## max 0.38461538

#   Property 1:  min = 0.0002,  max = 1.3160
#   Property 2:  min = -0.0440,  max = 0.6448
#   Property 3:  min = 0.0003,  max = 1.3160
#   Property 4:  min = 0.0000,  max = 0.3356
#   Property 5:  min = 0.1360,  max = 1.0000

#np.random.seed(42)
for _ in range(10000):  # Generate 5 different random designs 
    config.desiredVolumeFraction = np.round(np.random.uniform(low=0.3, high=0.7),decimals=2)
    config.desiredC11 = np.round(np.random.uniform(low=0.05, high=1),decimals=2)
    config.desiredC12 = np.round(np.random.uniform(low=-0.05, high=0.33),decimals=2)
    config.desiredC22 = np.round(np.random.uniform(low=0.05, high=1),decimals=2)
    config.desiredC66 = np.round(np.random.uniform(low=0.01, high=0.38),decimals=2)

    config.desiredQ = torch.tensor([[config.desiredC11, config.desiredC12, 0.0], [config.desiredC12, config.desiredC22, 0.0], [0.0, 0.0, config.desiredC66]], dtype = data_type)

    topOpt = TopologyOptimizer(config,data_type,device)
    print(topOpt.exper_name)
    topOpt.optimizeDesign(config) 
    print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
    print(topOpt.exper_name)
    #topOpt.plotConvergence() 
