import numpy as np

# Create a simple dummy lattice: 20x20 checkerboard
lattice = np.indices((20, 20)).sum(axis=0) % 2

# Save it to the correct path
np.save("results/lattice_output.npy", lattice)



