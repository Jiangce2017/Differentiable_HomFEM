import numpy as np
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
from os import path
from hom_FE import StructuralFE
import matplotlib.pyplot as plt
from matplotlib import colors
from pytictoc import TicToc
import os.path as osp
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from TO_models import TopNet
from utils import setDevice,set_seed,Logger

from matplotlib import rc
rc('text', usetex=False)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 150
timer = TicToc()

overrideGPU = False
device = setDevice(overrideGPU) 
torch.autograd.set_detect_anomaly(True)
    
class TopologyOptimizer:
    def __init__(self,config,data_type):
        self.nelx = config.nelx
        self.nely = config.nely
        self.len_x = config.len_x
        self.len_y = config.len_y
        self.symXAxis = True 
        self.symYAxis = True
        self.max_grad = config.max_grad
        self.simplexDim = config.simplexDim
        self.cell_width = config.cell_width
        self.cell_type = config.cell_type
        self.results_dir = config.results_dir
        self.interactive = config.interactive
        self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular' : None } 

        self.exper_name = config.nn_type+ "_"+config.cell_type + "_" + str(config.desiredVolumeFraction) 
        self.initializeFE(config,data_type)

    def initializeFE(self,config,data_type):
        self.FE = StructuralFE() 
        self.FE.initializeSolver(data_type,config.nelx, config.nely, config.penal, config.Emin, config.Emax, config.nu) 
        self.xy, self.nonDesignIdx = self.generatePoints(config.nelx, config.nely, 1, self.nonDesignRegion) 
    
    def generatePoints(self, nelx, nely, resolution = 1, nonDesignRegion = None): # generate points in elements
        ctr = 0 
        xy = np.zeros((resolution*nelx*resolution*nely,2)) 
        nonDesignIdx = torch.zeros((resolution*nelx*resolution*nely), requires_grad = False).to(device) 
        for i in range(resolution*nelx):
            for j in range(resolution*nely):
                xy[ctr,0] = (i + 0.5)/resolution 
                xy[ctr,1] = (j + 0.5)/resolution 
                if(nonDesignRegion['Rect'] is not None):
                    if( (xy[ctr,0] < nonDesignRegion['Rect']['x<']) and (xy[ctr,0] > nonDesignRegion['Rect']['x>']) and (xy[ctr,1] < nonDesignRegion['Rect']['y<']) and (xy[ctr,1] > nonDesignRegion['Rect']['y>'])):
                        nonDesignIdx[ctr] = 1 
                if(nonDesignRegion['Circ'] is not None):
                    if( ( (xy[ctr,0]-nonDesignRegion['Circ']['center'][0])**2 + (xy[ctr,1]-nonDesignRegion['Circ']['center'][1])**2 ) <= nonDesignRegion['Circ']['rad']**2):
                        nonDesignIdx[ctr] = 1 
                if(nonDesignRegion['Annular'] is not None):
                     locn =  (xy[ctr,0]-nonDesignRegion['Annular']['center'][0])**2 + (xy[ctr,1]-nonDesignRegion['Annular']['center'][1])**2 
                     if ((locn <= nonDesignRegion['Annular']['rad_out']**2) and (locn > nonDesignRegion['Annular']['rad_in']**2) ):
                         nonDesignIdx[ctr] = 1 
                ctr += 1 
        xy = torch.tensor(xy, requires_grad = True).float().view(-1,2).to(device) 
        return xy, nonDesignIdx 
    
    def density_filter(self, rho, xy, r_min):
        """
        Vectorized density filter using distance-weighted average.

        Parameters:
        - rho: (N,) tensor of densities
        - xy: (N,2) tensor of coordinates
        - r_min: filtering radius

        Returns:
        - filtered_rho: (N,) tensor
        """
        # Compute pairwise distances (N x N)
        dist = torch.cdist(xy, xy, p=2)  # Euclidean distance
        #weight = torch.clamp(r_min - dist, min=0.0)  # (N x N), clamp negative values to 0
        weight = r_min > dist
        weight = weight.float()  # Convert boolean to float
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)  # Normalize weights along rows to prevent division by zero
        #weight = torch.softmax(weight, dim=1)  # Normalize weights along rows
        filtered_rho = weight @ rho  # (N x N) @ (N,) → (N,)

        # weighted_rho = (weight * rho.unsqueeze(0))  # Broadcast rho: (1,N) → (N,N)


        # filtered_rho = weighted_rho.sum(dim=1) / (weight.sum(dim=1) + 1e-8)

        return filtered_rho
    
    def sensitivity_filter(self, grad_rho, xy, r_min):
        """
        Applies sensitivity filtering to the gradient of the design variables.
        
        Parameters:
        - grad_rho: (N,) raw gradient tensor of objective w.r.t. rho
        - xy: (N,2) tensor of element coordinates
        - r_min: minimum filter radius

        Returns:
        - filtered_grad: (N,) filtered gradient tensor
        """
        with torch.no_grad():
            dist = torch.cdist(xy, xy, p=2)  # Shape: (N, N)
            weight = torch.clamp(r_min - dist, min=0.0)  # Shape: (N, N)

            weighted_grad = weight @ grad_rho  # (N,N) @ (N,) → (N,)
            normalization = weight.sum(dim=1) + 1e-8  # Prevent divide by zero

        return weighted_grad / normalization


    def optimizeDesign(self,config, desiredVolumeFraction, desiredQ):
        manualSeed = 1234  # NN are seeded manually 
        set_seed(manualSeed)
        self.topNet = TopNet(config,self.symXAxis,self.symYAxis).to(device)
        self.objective = 0.
        self.convergenceHistory = []
        train_logger = Logger(
            osp.join(config.results_dir, self.exper_name+'train.log'),
            ['ep', 'elasticity_loss','volume_loss']
        )
        self.convergenceHistory = [] 
        savedNetFileName = osp.join(config.results_dir, str(self.nelx) + '_' + str(self.nely) +  '.nt')
        alphaMax = 100*desiredVolumeFraction 
        alphaIncrement= 0.08
        alpha = alphaIncrement  # start
        nrmThreshold = 0.1  # for gradient clipping
        if(config.useSavedNet):
            if (path.exists(savedNetFileName)):
                self.topNet = torch.load(savedNetFileName) 
            else:
                print("Network file not found")
        if config.nn_type == 'SIMP':
             self.optimizer = torch.optim.Adam([
                {'params': self.topNet.model.rho, 'lr': config.learningRate}
            ])  
        else:
            self.optimizer = optim.Adam(self.topNet.parameters(),lr=config.learningRate)
        w = self.cell_width
        self.topNet.train()
        batch_x =  self.xy.view(-1,2).float().to(device)
        for epoch in range(config.maxEpochs):
            self.optimizer.zero_grad()

            nn_rho = self.topNet(batch_x,1,self.nonDesignIdx)
            nn_rho = self.density_filter(nn_rho, self.xy.view(-1, 2).float().to(device), r_min=config.filterRadius)
            
            # # Apply minimum-length filtering
            # with torch.no_grad():  # Filtering itself doesn't require gradients
            #      nn_rho = self.density_filter(nn_rho, self.xy.view(-1, 2).float().to(device), r_min=config.filterRadius)
            # # Detach and re-enable grad for backprop
            # nn_rho = nn_rho_filtered.clone().detach().requires_grad_(True)

            Q = self.FE.solve(nn_rho)

            objective = torch.linalg.norm(Q-desiredQ) / torch.linalg.norm(desiredQ)
            volConstraint =((torch.mean(nn_rho)/desiredVolumeFraction) - 1.0) 
            currentVolumeFraction = torch.mean(nn_rho).item() 
            self.objective = objective
            loss = self.objective+ alpha*(pow(volConstraint,2))

            alpha = min(alphaMax, alpha + alphaIncrement) 

            loss.backward(retain_graph=True) 
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(),nrmThreshold)
            self.optimizer.step()

            # loss.backward(retain_graph=True)
            # # Apply sensitivity filtering only to grad of rho
            # # if config.nn_type == 'SIMP':
            # with torch.no_grad():
            #     if config.nn_type == 'SIMP':
            #         rho_param = self.topNet.model.rho
            #     else:
            #         #rho_param = self.topNet.parameters()
            #         rho_param = torch.cat([p.view(-1) for p in self.topNet.parameters()])
            #     if rho_param.grad is not None:
            #         grad = rho_param.grad.view(-1)
            #         xy_coords = self.xy.view(-1, 2).float().to(device)
            #         filtered_grad = self.sensitivity_filter(grad, xy_coords, config.filterRadius)
            #         rho_param.grad.copy_(filtered_grad.view_as(rho_param))

            # torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            # self.optimizer.step()


            if(volConstraint < 0.05): # Only check for gray when close to solving. Saves computational cost  
                greyElements = torch.sum((nn_rho > 0.05)*(nn_rho < 0.95)).item()  
                relGreyElements = greyElements/nn_rho.shape[0]
            else:
                relGreyElements = 1 
            self.convergenceHistory.append([ self.objective.item(), currentVolumeFraction,loss.item(),relGreyElements]) 
            train_logger.log({
                'ep': epoch,             
                'elasticity_loss': objective.item(),
                'volume_loss': volConstraint.item()
            })
            if(epoch % 10 == 0):
                print("{:3d} Elast_loss: {:.6F}; Volume_loss: {:.6F}; relGreyElems: {:.3F} "\
                    .format(epoch, self.objective.item(),volConstraint.item(),relGreyElements))
                if config.interactive:
                    self.plotTO(epoch, saveFig=False,saveFrame=config.saveFrame) 
            
            if ((epoch > config.minEpochs ) & (relGreyElements < 0.035) & (volConstraint< 0) ):
                break 
        self.plotTO(epoch,saveFig=True, saveFrame=config.saveFrame) 
        
        ### save data
        torch.save(self.topNet, savedNetFileName)
        
    def plotTO(self, iter,saveFig=True, saveFrame=False):
        w = self.cell_width
        batch_x = self.xy.view(-1,2).float().to(device)  
        nn_rho = self.topNet(batch_x,1,self.nonDesignIdx)
        nn_rho = nn_rho.to('cpu').detach().numpy()
        # nn_rho[nn_rho>0.5]=1
        # nn_rho[nn_rho<0.5]=0

        img = np.flip(nn_rho.reshape(self.FE.nely,self.FE.nelx).transpose(),axis=0)
        #true_rho = nn_rho
        if self.interactive:
            plt.ion() 
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        axes = plt.gca()
        cmap = plt.get_cmap('Greens')
        axes.imshow(img,cmap=cmap,vmin=0,vmax=1)
        if(saveFrame):
            frame_file_name = osp.join(self.results_dir, 'frames','f_'+str(iter)+'.png')
            plt.savefig(frame_file_name,transparent=True)
            print("frame plotted")   

        #plt.title('Iter = {:d}, E = {:.2F}, V_f = {:.2F}, V_des = {:.2F}'.format(iter, real_compliance, np.mean(true_rho),  self.desiredVolumeFraction),loc='left')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cbar = plt.colorbar(m, cax=cax, aspect=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Density", fontsize=10)
        plt.ticklabel_format(style="plain")
           
        if (saveFig):    
            fName = osp.join(self.results_dir, self.exper_name+'_topology.png')
            plt.savefig(fName,dpi = 450,transparent=True)
            np.save("results/lattice_output.npy", img.astype(np.uint8))  # save as 0-1 ints

        if self.interactive:  
            plt.pause(0.01)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory) 
        plt.figure()
        plt.semilogy(self.convergenceHistory[:,0], 'b:',label = 'Rel. Compliance')
        plt.semilogy(self.convergenceHistory[:,1], 'r--',label = 'Vol. Fraction')
        plt.title('Convergence Plots' ) 
        #plt.title('Convergence plots; V_des = {:.2F}'.format(self.desiredVolumeFraction))
        plt.xlabel('Iterations') 
        plt.grid('True')
        plt.legend(loc='lower left', shadow=True, fontsize='large')
        fName = osp.join(self.results_dir,self.exper_name+'_convergence.png')
        plt.savefig(fName,dpi = 450)

    