import numpy as np
import h5py
from elasticity_tensor_2d import elasticity

def load_shapespace_example(index, shape_path='ShapeSpace.mat', prop_path='PropertySpace.mat'):
    with h5py.File(shape_path, 'r') as f:
        shape_key = list(f.keys())[0]
        dset = f[shape_key]
        xPhys = dset[index, :, :].astype(np.float64)  # convert to float64
        xPhys = xPhys.T  # Transpose to (nely, nelx)

    with h5py.File(prop_path, 'r') as f:
        prop_key = list(f.keys())[0]
        props = f[prop_key]
        C11 = float(props[0, index])
        C12 = float(props[1, index])
        C22 = float(props[2, index])
        C66 = float(props[3, index])

    Q_example = np.array([[C11, C12, 0.0], [C12, C22, 0.0], [0.0, 0.0, C66]], dtype=np.float64)

    return xPhys, Q_example


if __name__ == '__main__':
    error_list = []
    sample_range = range(500, 520)  # Samples ___ to ____ (inclusive)
    individual_sample_index = None # Set to none if you do not care

    for sample_index in sample_range:
        xPhys, Q_example = load_shapespace_example(sample_index)
        Q_code = elasticity(xPhys)

        # Relative Frobenius norm error
        error = np.linalg.norm(Q_code - Q_example) / np.linalg.norm(Q_example)
        error_list.append(error)
        #Can print every single error if user wants to
        # print(f"Sample {sample_index:3d}: Relative Error = {error:.4%}")

    avg_error = np.mean(error_list)
    print(f"\n📊 Average Relative Error over {len(sample_range)} samples: {avg_error:.4%}")

    if individual_sample_index is not None:
        try:
            xPhys, Q_example = load_shapespace_example(individual_sample_index)
            Q_code = elasticity(xPhys)
            error = np.linalg.norm(Q_code - Q_example) / np.linalg.norm(Q_example)

            print(f"\n🔍 Inspecting Sample {individual_sample_index}:")
            print("\nQ_code (from FEM):")
            print(Q_code)
            print("\nQ_example (from dataset):")
            print(Q_example)
            print(f"\n📉 Relative Frobenius Norm Error: {error:.4%}")
        except Exception as e:
            print(f"\n⚠️ Could not evaluate sample {individual_sample_index}: {e}")