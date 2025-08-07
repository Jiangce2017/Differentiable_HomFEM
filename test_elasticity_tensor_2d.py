import numpy as np
import h5py
from joblib import Parallel, delayed
from elasticity_tensor_2d import elasticity

def load_full_dataset(shape_path='ShapeSpace.mat', prop_path='PropertySpace.mat'):
    with h5py.File(shape_path, 'r') as f:
        shape_key = list(f.keys())[0]
        shape_data = f[shape_key][:].astype(np.float64)
        shape_data = shape_data.transpose((0, 2, 1))

    with h5py.File(prop_path, 'r') as f:
        prop_key = list(f.keys())[0]
        prop_data = f[prop_key][:].astype(np.float64)

    return shape_data, prop_data

def compute_error(index, shape_data, prop_data):
    xPhys = shape_data[index]
    C11 = prop_data[0, index]
    C12 = prop_data[1, index]
    C22 = prop_data[2, index]
    C66 = prop_data[3, index]
    Q_example = np.array([[C11, C12, 0.0], [C12, C22, 0.0], [0.0, 0.0, C66]], dtype=np.float64)

    Q_code = elasticity(xPhys)
    error = np.linalg.norm(Q_code - Q_example) / np.linalg.norm(Q_example)
    return error

def inspect_sample(index, shape_data, prop_data):
    xPhys = shape_data[index]
    C11 = prop_data[0, index]
    C12 = prop_data[1, index]
    C22 = prop_data[2, index]
    C66 = prop_data[3, index]
    Q_example = np.array([[C11, C12, 0.0], [C12, C22, 0.0], [0.0, 0.0, C66]], dtype=np.float64)

    Q_code = elasticity(xPhys)
    error = np.linalg.norm(Q_code - Q_example) / np.linalg.norm(Q_example)

    print(f"\n🔍 Inspecting Sample {index}:")
    print("\nQ_code (from FEM):")
    print(Q_code)
    print("\nQ_example (from dataset):")
    print(Q_example)
    print(f"\n📉 Relative Frobenius Norm Error: {error:.4%}")

if __name__ == '__main__':

    xPhys = np.ones((50, 50))  # Example shape data
    Q = elasticity(xPhys,E0=1.0, Emin=1e-9, nu=0.3)
    print("Computed Elasticity Tensor Q:")
    print(Q)
    E0 = 1
    nu = 0.3
    C = E0/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    print("\nExpected Elasticity Tensor C:")
    print(C)

    xPhys = np.load("results/jury_data.npy")
    #xPhys = np.load("results/lattice_output.npy")
    #xPhys = np.load("results/20_20.nt")
    # print(xPhys.files)

    Q = elasticity(xPhys,E0=1255.0, Emin=1e-9, nu=0.3)
    print("Computed Elasticity Tensor Q:")
    print(Q)

    # sample_range = range(12700, 12940)
    # individual_sample_index = 340  # or an integer index

    # shape_data, prop_data = load_full_dataset()

    # # 🔄 Parallel computation of errors
    # error_list = Parallel(n_jobs=-1)(  # use all available cores
    #     delayed(compute_error)(idx, shape_data, prop_data) for idx in sample_range
    # )

    # avg_error = np.mean(error_list)
    # print(f"\n📊 Average Relative Error over {len(sample_range)} samples: {avg_error:.4%}")

    # # 🧪 Optional individual sample inspection
    # if individual_sample_index is not None:
    #     try:
    #         inspect_sample(individual_sample_index, shape_data, prop_data)
    #     except Exception as e:
    #         print(f"\n⚠️ Could not evaluate sample {individual_sample_index}: {e}")
