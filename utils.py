import h5py

def load_mat(ShapeSpace):
    with h5py.File(ShapeSpace, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v[:]  # Load data into memory
    return data


