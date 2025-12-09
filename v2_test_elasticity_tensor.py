import numpy as np
from elasticity_tensor_2d import elasticity  # assumes elasticity() is defined there

def main():
    # Path to your text file
    txt_path = "results2/lattice_series_0_0.txt"

    # ✅ Load text file of 1's and 0's
    xPhys = np.loadtxt(txt_path)

    # Ensure numeric and 2D
    xPhys = np.array(xPhys, dtype=float)
    if xPhys.ndim != 2:
        raise ValueError(f"Expected a 2D array from {txt_path}, got shape {xPhys.shape}")

    # Compute elasticity tensor
    Q = elasticity(xPhys)

    # Print the resulting tensor
    print("✅ Homogenized Elasticity Tensor (MPa):")
    print(Q)

if __name__ == "__main__":
    main()
