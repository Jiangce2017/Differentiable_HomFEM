import numpy as np
from FE import StructuralFE

nelx = 60
nely = 30
example = 1

if(example == 1): #  MBBBeam
    exampleName = 'MBBBeam'
    ndof = 2*(nelx+1)*(nely+1) 
    dofs=np.arange(ndof) 
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1)
    forceBC = np.zeros((ndof,1)) 
    forceBC[2*(nely+1)+1 ,0]=-1 
elif(example == 2): #  TipCantilever, cantilever with a loading at bottom-right corner
    exampleName = 'TipCantilever'
    ndof = 2*(nelx+1)*(nely+1) 
    dofs=np.arange(ndof) 
    ## specify the fixed boundary condition, fixed and force boundary condition, forceBC for MBBeam

FE_solver = StructuralFE()
FE_solver.initializeSolver(nelx, nely, forceBC, fixed)
density = np.ones(nelx*nely)
u,J = FE_solver.solve(density)
figure_name = exampleName+".jpg"
FE_solver.plotFE(figure_name)