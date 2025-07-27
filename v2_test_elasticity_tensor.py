import numpy as np
from elasticity_tensor_2d import elasticity  # assuming elasticity function is in elasticity.py

def main():
    # Load lattice data from results folder
    xPhys = np.load("results/lattice_output1.npy")

    # Compute elasticity tensor
    Q = elasticity(xPhys)

    # Print the resulting tensor
    print("Homogenized Elasticity Tensor (MPa):")
    print(Q)

if __name__ == "__main__":
    main()
