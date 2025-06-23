import numpy as np
import h5py
from elasticity_tensor_2d import elasticity

def load_shapespace_example(index, shape_path='ShapeSpace.mat', prop_path='PropertySpace.mat' ):
    with h5py.File(shape_path,'r') as f:
        shape_key = list(f.keys())[0]
        dset = f[shape_key]
        xPhys = np.empty((50,50), dtype=np.float32)
        dset.read_direct(xPhys, source_sel=np.s_[:, :, index])  # Load one slice only
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
    sample_index = 0
    print(f"\nTesting microstructure at index {sample_index}")
    xPhys, Q_example = load_shapespace_example(sample_index)

    Q_code = elasticity(xPhys)
    print("\n Input Microstructure Shape (xPhys): 50 x 50 binary grid")
    print("(omitted from print to save space)\n")

    print("Elasticity Tensor from Your Code (Q_code):")
    print(np.round(Q_code, 4))

    print("\n Reference Elasticity Tensor from Dataset (Q_example):")
    print(np.round(Q_example, 4))