import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh
import os

def generate_dogbone_mask(length_mm=165, width_gauge_mm=13, width_ends_mm=19, length_gauge_mm=57, resolution=20):
    pixel_per_mm = resolution
    gage_start_mm = (length_mm - length_gauge_mm) // 2

    width_gauge_px = width_gauge_mm * pixel_per_mm
    width_ends_px = width_ends_mm * pixel_per_mm
    length_total_px = length_mm * pixel_per_mm
    length_gauge_px = length_gauge_mm * pixel_per_mm
    gage_start_px = gage_start_mm * pixel_per_mm

    domain = np.zeros((length_total_px, width_ends_px), dtype=np.uint8)

    # Middle gauge section
    domain[gage_start_px:gage_start_px+length_gauge_px,
           (width_ends_px-width_gauge_px)//2:(width_ends_px+width_gauge_px)//2] = 1

    # End grips
    domain[:gage_start_px, :] = 1
    domain[gage_start_px+length_gauge_px:, :] = 1

    return domain

def tile_lattice_to_mask(lattice_tile, mask):
    tiled = np.zeros_like(mask, dtype=np.uint8)
    tile_h, tile_w = lattice_tile.shape

    for y in range(0, mask.shape[0] - tile_h + 1, tile_h):
        for x in range(0, mask.shape[1] - tile_w + 1, tile_w):
            if mask[y:y+tile_h, x:x+tile_w].all():
                tiled[y:y+tile_h, x:x+tile_w] = lattice_tile

    return tiled

def get_unique_filename(base_name="dogbone_lattice", ext=".stl", folder="results/dogbones"):
    i = 0
    while True:
        filename = f"{base_name}{i if i else ''}{ext}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

def extrude_and_export_stl(lattice_2d, thickness_mm, voxel_size_mm=1.0, filename="results/dogbones/dogbone_lattice.stl"):
    # Create 3D voxel array
    voxels = np.repeat(lattice_2d[:, :, np.newaxis], int(thickness_mm), axis=2)

    # Convert to mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(voxels, level=0.5)

    # Scale to mm
    verts *= voxel_size_mm

    # Create mesh and export
    solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            solid.vectors[i][j] = verts[f[j], :]

    solid.save(filename)
    print(f"STL file saved to {filename}")

def main():
    lattice_path = "results/lattice_output.npy"
    if not os.path.exists(lattice_path):
        raise FileNotFoundError(f"Cannot find {lattice_path}")

    lattice_tile = np.load(lattice_path)  # Must be 2D, e.g., 20x20
    assert lattice_tile.ndim == 2, "Lattice must be 2D!"

    resolution = lattice_tile.shape[0]
    domain_mask = generate_dogbone_mask(resolution=resolution)
    tiled_lattice = tile_lattice_to_mask(lattice_tile, domain_mask)

    # Optional: visualize the 2D layout
    plt.imshow(tiled_lattice, cmap='gray')
    plt.title("Tiled Lattice in ASTM D638 Dogbone")
    plt.axis('off')
    plt.show()

    # User-defined thickness
    thickness_mm = float(input("Enter desired thickness (in mm): "))
    filename = get_unique_filename()
    extrude_and_export_stl(tiled_lattice, thickness_mm, filename=filename)

if __name__ == "__main__":
    main()
