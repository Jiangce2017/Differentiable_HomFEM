import numpy as np
import ezdxf
import os

def export_single_lattice_cell_dxf(lattice_tile, voxel_size_mm=2.0, filename="results/lattice_cell.dxf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = ezdxf.new()
    msp = doc.modelspace()

    rows, cols = lattice_tile.shape
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
    lattice_path = "results/lattice_output.npy"
    if not os.path.exists(lattice_path):
        raise FileNotFoundError(f"Cannot find {lattice_path}")

    lattice_tile = np.load(lattice_path)
    assert lattice_tile.ndim == 2, "Lattice must be 2D!"

    export_single_lattice_cell_dxf(lattice_tile, voxel_size_mm=2.0, filename="results/lattice_cell.dxf")


if __name__ == "__main__":
    main()
