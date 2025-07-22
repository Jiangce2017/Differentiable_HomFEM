import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_dogbone_mask(length_mm=165, width_gauge_mm=13, width_ends_mm=19, length_gauge_mm=57, pixel_per_mm = 20): 
    
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

    return domain, gage_start_px,length_gauge_px,width_ends_px,width_gauge_px

def tile_lattice_to_mask(lattice_tile, mask,tentile_direction,gage_start_px,length_gauge_px,width_ends_px,width_gauge_px):
    #tiled = np.zeros_like(mask, dtype=np.uint8)
    tiled = mask.copy()
    #tiled = np.ones_like(mask, dtype=np.uint8)
    #tiled[mask] = 1
    tile_h, tile_w = lattice_tile.shape
    print("tile_h: {}, tile_w: {}".format(tile_h, tile_w))

    # for y in range(0, mask.shape[0] - tile_h + 1, tile_h):
    #     for x in range(0, mask.shape[1] - tile_w + 1, tile_w):
    for y in range(gage_start_px, gage_start_px+length_gauge_px - tile_h + 1, tile_h):
        for x in range((width_ends_px-width_gauge_px)//2, (width_ends_px+width_gauge_px)//2 - tile_w + 1, tile_w):
    
            if mask[y:y+tile_h, x:x+tile_w].all():
                if tentile_direction == 'y':
                    tiled[y:y+tile_h, x:x+tile_w] = lattice_tile
                elif tentile_direction == 'x':
                    tiled[y:y+tile_h, x:x+tile_w] = lattice_tile.transpose()

    return tiled
#Able to get unique file names for different dogbones
def get_unique_filename(tentile_direction,base_name="dogbone_lattice", ext=".stl", folder="results/dogbones"):
    i = 1
    while True:
        filename = f"{base_name}{'_'+tentile_direction+'_'}{i if i else ''}{ext}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

def extrude_and_export_stl(lattice_2d, filename="results/dogbones/dogbone_lattice.stl"):
    # Create 3D voxel array
    lattice_3d = np.repeat(lattice_2d[:, :, np.newaxis], 3, axis=2)
    voxels = np.zeros((lattice_3d.shape[0]+2,lattice_3d.shape[1]+2,lattice_3d.shape[2]+2), dtype=np.uint8)
    voxels[1:-1,1:-1,1:-1] = lattice_3d

    # Convert to mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(voxels, level=0.5)

    # rescale to real dimensions
    length_mm=165
    width_ends_mm=19
    thickness_mm = 3.2
    target_dim = np.array([length_mm,width_ends_mm,thickness_mm])
    verts = rescale_dimension(verts, target_dim)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)
    print(f"STL file saved to {filename}")

def rescale_dimension(verts, target_dim):
    v_min = np.min(verts,axis=0)
    v_max = np.max(verts,axis=0)
    scale = target_dim/(v_max-v_min)
    verts = (verts - v_min)*scale
    return verts




def main():
    length_mm=166
    width_gauge_mm=14
    width_ends_mm=22
    length_gauge_mm=58
    lattice_path = "results/lattice_output.npy"
    tentile_direction = 'x'
    
    if not os.path.exists(lattice_path):
        raise FileNotFoundError(f"Cannot find {lattice_path}")

    lattice_tile = np.load(lattice_path)  # Must be 2D, e.g., 20x20
    assert lattice_tile.ndim == 2, "Lattice must be 2D!"
    plt.imshow(lattice_tile, cmap='Greens')
    plt.title("Lattice")
    plt.axis('off')
    axes = plt.gca()
    cmap = plt.get_cmap('Greens')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="3%", pad="2%")
    cbar = plt.colorbar(m, cax=cax, aspect=0.5)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Density", fontsize=10)
    plt.ticklabel_format(style="plain")
    plt.show()

    lattice_resolution = lattice_tile.shape[0]
    lattice_size_mm = 2
    pixel_per_mm = lattice_resolution // lattice_size_mm
    domain_mask, gage_start_px,length_gauge_px,width_ends_px,width_gauge_px = generate_dogbone_mask(length_mm=length_mm, width_gauge_mm=width_gauge_mm, width_ends_mm=width_ends_mm, length_gauge_mm=length_gauge_mm,pixel_per_mm=pixel_per_mm)
    tiled_lattice = tile_lattice_to_mask(lattice_tile, domain_mask,tentile_direction, gage_start_px,length_gauge_px,width_ends_px,width_gauge_px)

    # Optional: visualize the 2D layout
    plt.imshow(tiled_lattice, cmap='Greens')
    plt.title("Tiled Lattice in ASTM D638 Dogbone")
    plt.axis('off')
    plt.show()
    filename = get_unique_filename(tentile_direction)
    extrude_and_export_stl(tiled_lattice, filename=filename)

if __name__ == "__main__":
    main()