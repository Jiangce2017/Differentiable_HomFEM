import numpy as np
from elasticity_tensor_2d import elasticity

if __name__ == '__main__':
    xPhys = np.ones((50,50))
    Q = elasticity(xPhys)
    print(Q)