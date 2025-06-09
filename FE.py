import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import cvxopt 
import cvxopt.cholmod
import sys
if sys.platform == 'linux':
    import torch_sparse_solve 
#-----------------------#

#%%  structural FE
class StructuralFE:
    #-----------------------#
    def initializeSolver(self, nelx, nely, forceBC, fixed, penal = 3,Emin = 1e-3, Emax = 1.0):
        self.Emin = Emin;
        self.Emax = Emax;
        self.penal = penal;
        self.nelx = nelx;
        self.nely = nely;
        self.ndof = 2*(nelx+1)*(nely+1)
        self.KE=self.getDMatrix();
        self.fixed = fixed;
        self.free = np.setdiff1d(np.arange(self.ndof),fixed);
        self.f = forceBC;
        self.edofMat=np.zeros((nelx*nely,8),dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

        self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()

    def getDMatrix(self):
        E=1
        nu=0.3
        k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
        return (KE)

    def solve(self, density):

        self.densityField = density
        self.u=np.zeros((self.ndof,1))
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = K[self.free,:][:,self.free]

        self.u[self.free,0]=spsolve(K,self.f[self.free,0])
        self.Jelem = (np.dot(self.u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)
        return self.u, self.Jelem


    def plotFE(self,figure_name):
         #plot FE results
         fig= plt.figure() # figsize=(10,10)
         plt.subplot(1,2,1);
         im = plt.imshow(self.u[1::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         J = ( (self.Emin+self.densityField**self.penal*(self.Emax-self.Emin))*self.Jelem).sum()
         plt.title('U_x , J = {:.2E}'.format(J))
         fig.colorbar(im)
         plt.subplot(1,2,2);
         im = plt.imshow(self.u[0::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         fig.colorbar(im)
         plt.title('U_y')
         plt.savefig(figure_name)
         #fig.show()

