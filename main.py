import numpy as np
from FE import StructuralFE
#test
nelx = 60
nely = 30
example = 1

if(example == 1): #  MBBBeam
    exampleName = 'MBBBeam'
    ndof = 2*(nelx+1)*(nely+1) 
    dofs=np.arange(ndof) 
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1) #fixing left edge horizontally and bottom right node vertically
    forceBC = np.zeros((ndof,1)) #force vector with 0's
    forceBC[2*(nely+1)+1 ,0]=-1 #Apply a downward force 1 unit on second node (from the left) along the bottom edge in y.

elif(example == 2): #  TipCantilever, cantilever with a loading at bottom-right corner
    exampleName = 'TipCantilever'
    ndof = 2*(nelx+1)*(nely+1) 
    dofs=np.arange(ndof)

    fixedNodes = np.arange(nely+1)  #gets nodes at left edge
    fixedNodesX = 2*fixedNodes  # fixes left edge nodes in X direction
    fixedNodesY = 2*fixedNodes + 1 #fixes left edge nodes in Y direction
    fixed = np.union1d(fixedNodesX, fixedNodesY)  #creates array of fixed nodes

    forceBC = np.zeros((ndof,1)) # creates a 1D column vector of 0's
    forceNode = nelx*(nely+1)+0 # locates the node where force should be applied
    force_dof_y = 2*forceNode+1 # makes it so force only acts in y direction
    forceBC[force_dof_y,0] = -1 # assigns force value of -1 to the node in y direction

    ## specify the fixed boundary condition, fixed and force boundary condition, forceBC for MBBeam

FE_solver = StructuralFE()
FE_solver.initializeSolver(nelx, nely, forceBC, fixed)
density = np.ones(nelx*nely)
u,J = FE_solver.solve(density)
figure_name = exampleName+".jpg"
FE_solver.plotFE(figure_name)