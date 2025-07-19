import numpy as np
import ezdxf
import os


def get_unique_filename(base_name="lattice_cell", ext=".dxf", folder="results"):
    os.makedirs(folder, exist_ok=True)
    i = 0
    while True:
        filename = f"{base_name}{i if i else ''}{ext}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def export_single_lattice_cell_dxf(lattice_tile, desired_size_mm=2.0):
    filename = get_unique_filename()
    doc = ezdxf.new()
    msp = doc.modelspace()

    rows, cols = lattice_tile.shape
    assert rows == cols, "Tile must be square to scale to 2mm x 2mm uniformly"

    voxel_size_mm = desired_size_mm / rows  # adjust voxel size so full tile is 2mm x 2mm

    for y in range(rows):
        for x in range(cols):
            if lattice_tile[y, x] == 1:
                x0 = x * voxel_size_mm
                y0 = (rows - y - 1) * voxel_size_mm  # Flip Y
                x1 = x0 + voxel_size_mm
                y1 = y0 + voxel_size_mm
                polyline = msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                polyline.close(True)

    doc.saveas(filename)
    print(f"✅ Single lattice cell DXF saved to {filename}")


def main():
    lattice_path = "results/lattice_output1.npy" #change based on which .npy you want
    if not os.path.exists(lattice_path):
        raise FileNotFoundError(f"Cannot find {lattice_path}")

    lattice_tile = np.load(lattice_path)
    assert lattice_tile.ndim == 2, "Lattice must be 2D!"

    export_single_lattice_cell_dxf(lattice_tile, desired_size_mm=2.0)


if __name__ == "__main__":
    main()
