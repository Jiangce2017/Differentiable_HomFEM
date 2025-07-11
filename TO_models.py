import torch.nn as nn
import torch
from utils import set_seed
from fourier_2d import FNO2d,FNO2d_Lattice,CNN2d_Lattice

class TopNet(nn.Module):
    def __init__(self,config,symXAxis,symYAxis):
        super(TopNet, self).__init__()
        nn_type = config.nn_type
        if nn_type == 'FC':
            self.model = FC_Net(config,symXAxis,symYAxis)
        elif nn_type == 'FNO':
            self.model = FNO_Net(config,symXAxis,symYAxis)
        elif nn_type == 'CNN':
            self.model = CNN_Net(config,symXAxis,symYAxis)
        elif nn_type == 'SIMP':
            self.model = Simp(config,symXAxis,symYAxis)

    def forward(self, x,resolution, fixedIdx):
        return self.model(x,resolution, fixedIdx)
    
class Simp(nn.Module):
    def __init__(self, config,symXAxis,symYAxis):
        super(Simp,self).__init__()
        self.nelx = config.nelx # to impose symm, get size of domain
        self.nely = config.nely
        self.inputDim = 2
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis
        self.rho = torch.zeros((self.nelx*self.nely),requires_grad=True)
    def forward(self, x,resolution, fixedIdx = None):
        rho = torch.sigmoid(self.rho)
        return rho

class FC_Net(nn.Module):
    def __init__(self, config,symXAxis,symYAxis):
        super(FC_Net,self).__init__()
        self.nelx = config.nelx # to impose symm, get size of domain
        self.nely = config.nely
        self.inputDim = 2
        self.outputDim = 1

        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis
        
        self.layers = nn.ModuleList() 
        current_dim = self.inputDim 
        for lyr in range(config.numLayers): # define the layers
            l = nn.Linear(current_dim, config.numNeuronsPerLyr) 
            nn.init.xavier_normal_(l.weight) 
            nn.init.zeros_(l.bias) 
            self.layers.append(l) 
            current_dim = config.numNeuronsPerLyr 
        self.layers.append(nn.Linear(current_dim, self.outputDim)) 
        self.bnLayer = nn.ModuleList() 
        for lyr in range(config.numLayers): # batch norm 
            self.bnLayer.append(nn.BatchNorm1d(config.numNeuronsPerLyr)) 
    def forward(self, x,resolution, fixedIdx = None):
        # LeakyReLU ReLU6 ReLU
        m = nn.ReLU6()  # LeakyReLU 
        ctr = 0
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx) 
        else:
            xv = x[:,0] 
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely)  
        else:
            yv = x[:,1] 
        x = torch.transpose(torch.stack((xv,yv)),0,1)

        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        x = self.layers[-1](x)
        out = x.view(-1,self.outputDim)
        rho = torch.sigmoid(out)
        rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        return  rho
        
    def  getWeights(self): # stats about the NN
        modelWeights = [] 
        modelBiases = [] 
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.view(-1).cpu().numpy()) 
            modelBiases.extend(lyr.bias.data.view(-1).cpu().numpy()) 
        return modelWeights, modelBiases 

class FNO_Net(nn.Module):
    #inputDim = 2  # x and y coordn of the point
    #outputDim = 2  # if material/void at the point
    def __init__(self,config,symXAxis,symYAxis):
        super(FNO_Net,self).__init__()
        self.nelx = config.nelx  # to impose symm, get size of domain
        self.nely = config.nely 
        self.inputDim = 2
        self.outputDim = 1
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis 
        self.fno = FNO2d_Lattice(config.numLayers,config.numModex,config.numModey,config.numNeuronsPerLyr,config.searchMode,config.simplexDim,config.latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx) 
        else:
            xv = x[:,0] 
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)

        x = self.fno(x) 

        if (self.symYAxis):
            x_mid_idx = self.nelx//2
            x_back = torch.flip(x[:,:x_mid_idx,:,:],dims=(1,))
            if self.nelx % 2 == 0:
                x[:,x_mid_idx:,:,:] = x_back
            else:
                x[:,x_mid_idx+1:,:,:] = x_back
        if (self.symXAxis):
            y_mid_idx = self.nely//2
            y_back = torch.flip(x[:,:,:y_mid_idx,:],dims=(2,))
            if self.nely % 2 == 0:
                x[:,:,y_mid_idx:,:] = y_back
            else:
                x[:,:,y_mid_idx+1:,:] = y_back
        #out = x.view(-1,self.outputDim)
        out = x.flatten()
        rho = torch.sigmoid(out)
        rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        return  rho
    
    def  getWeights(self): # stats about the NN
        modelWeights = [] 
        modelBiases = [] 
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy()) 
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy()) 
        return modelWeights, modelBiases 

class CNN_Net(nn.Module):
    def __init__(self,config,symXAxis,symYAxis):
        super(CNN_Net,self).__init__()
        self.nelx = config.nelx  # to impose symm, get size of domain
        self.nely = config.nely 
        self.inputDim = 2
        self.outputDim = 1
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis 
        self.model = CNN2d_Lattice(config.numLayers,config.numModex,config.numModey,config.numNeuronsPerLyr,config.searchMode,config.simplexDim,config.latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx) 
        else:
            xv = x[:,0] 
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)
        x = self.model(x)  
        out = x.view(-1,self.outputDim)
        rho = torch.sigmoid(out)
        rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        return  rho
    def  getWeights(self): # stats about the NN
        modelWeights = [] 
        modelBiases = [] 
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy()) 
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy()) 
        return modelWeights, modelBiases 