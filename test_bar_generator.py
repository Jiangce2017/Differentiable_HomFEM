import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon, circle

# Specimen and lattice parameters
lattice_size_mm = 1
resolution = 20
pixel_per_mm = resolution  # 1 mm = 20 pixels
width_gauge_mm = 13
width_ends_mm = 19
length_total_mm = 165
length_gauge_mm = 57
gage_start = (length_total_mm - length_gauge_mm)//2

# Specimen in pixels
width_gauge_px = width_gauge_mm * pixel_per_mm
width_ends_px = width_ends_mm * pixel_per_mm
length_total_px = length_total_mm * pixel_per_mm
length_gauge_px = length_gauge_mm * pixel_per_mm
gage_start_px = gage_start * pixel_per_mm

# Create domain
domain = np.zeros((length_total_px, width_ends_px))

# Draw the gauge section (narrow region)
domain[gage_start_px:gage_start_px+length_gauge_px,
       (width_ends_px-width_gauge_px)//2 : (width_ends_px+width_gauge_px)//2] = 1

# Draw ends (rectangular grips)
domain[:gage_start_px, :] = 1
domain[gage_start_px+length_gauge_px:, :] = 1

# Optional: Smooth the transitions with fillet radii for realism

# Generate 1mm x 1mm lattice tile
lattice_tile = np.indices((resolution, resolution)).sum(axis=0) % 2  # Checkerboard

# Tile the lattice across the domain
def tile_lattice(mask, tile):
    out = np.zeros_like(mask)
    for y in range(0, mask.shape[0], tile.shape[0]):
        for x in range(0, mask.shape[1], tile.shape[1]):
            if mask[y:y+tile.shape[0], x:x+tile.shape[1]].shape == tile.shape:
                out[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return out * mask  # Apply the mask

lattice_domain = tile_lattice(domain, lattice_tile)

# Plot
plt.figure(figsize=(8,3))
plt.imshow(lattice_domain, cmap='gray', interpolation='nearest')
plt.title('ASTM D638 Dog Bone Lattice Tiling')
plt.axis('off')
plt.show()
