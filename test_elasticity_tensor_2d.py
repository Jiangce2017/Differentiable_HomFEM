import numpy as np
import h5py
from elasticity_tensor_2d import elasticity

def load_shapespace_example(index, shape_path='ShapeSpace.mat', prop_path='PropertySpace.mat' ):
    with h5py.File(shape_path,'r') as f:
        shape_key = list(f.keys())[0]
        dset = f[shape_key]
        xPhys = dset[:, :, index]
        xPhys = xPhys.T  # Transpose to (nely, nelx) expected by elasticity()


    with h5py.File(prop_path,'r') as f:
        prop_key = list(f.keys())[0]
        props = f[prop_key]
        C11 = props[0, index]
        C12 = props[1, index]
        C22 = props[2, index]
        C66 = props[3, index]

    Q_example = np.array([[C11, C12, 0], [C12, C22, 0], [0, 0, C66]])

    return xPhys, Q_example


if __name__ == '__main__':
    sample_index = 0 # can change this value
    xPhys, Q_example = load_shapespace_example(sample_index)
    Q_code = elasticity(xPhys)

    print("Q_code (from your FEM):")
    print(np.round(Q_code, 4))

    print("\nQ_example (from dataset):")
    print(np.round(Q_example, 4))

