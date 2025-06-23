import numpy as np
import h5py
from elasticity_tensor_2d import elasticity

def load_shapespace_example(index, shape_path='ShapeSpace.mat', prop_path='PropertySpace.mat' ):
    with h5py.File(shape_path,'r') as f:
        key = list(f.keys())[0]
        dataset = f[key]
        xPhys = dataset[:, :, index]
        xPhys = xPhys.T

    with h5py.File(prop_path,'r') as f:
        key = list(f.keys())[0]
        props = f[key]
        C11 = props[0, index]
        C12 = props[1, index]
        C22 = props[2, index]
        C66 = props[3, index]

    Q_example = np.array([[C11, C12, 0], [C12, C22, 0], [0, 0, C66]])

    return xPhys, Q_example


if __name__ == '__main__':
    sample_index = 0
    xPhys, Q_example = load_shapespace_example(sample_index)

    Q_code = elasticity(xPhys)
    print(f"\n=== Sample #{sample_index} ===")
    print("\nComputed Elasticity Tensor (Q example):")
    print(np.round(Q_example, 4))

    print("\nReference Elasticity Tensor (Q code):")
    print(np.round(Q_code, 4))