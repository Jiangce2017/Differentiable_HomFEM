import numpy as np
import os
from skimage import measure


def get_unique_filename(base_name="lattice_cell", ext=".dxf", folder="results2"):
    os.makedirs(folder, exist_ok=True)
    i = 0
    while True:
        filename = f"{base_name}{i if i else ''}{ext}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def load_lattice_from_txt(file_path):
    """Load 2D binary lattice (1 solid, 0 void)."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    data = [list(map(int, line.strip().split())) for line in lines if line.strip()]
    return np.array(data, dtype=int)


def write_dxf_simple(filename, outer_points, inner_contours):
    """Write a simple DXF file with polylines for outer boundary and inner contours."""
    with open(filename, 'w') as f:
        # HEADER
        f.write('  0\n')
        f.write('SECTION\n')
        f.write('  2\n')
        f.write('HEADER\n')
        f.write('  9\n')
        f.write('$ACADVER\n')
        f.write('  1\n')
        f.write('AC1009\n')  # R12
        f.write('  0\n')
        f.write('ENDSEC\n')

        # TABLES
        f.write('  0\n')
        f.write('SECTION\n')
        f.write('  2\n')
        f.write('TABLES\n')
        f.write('  0\n')
        f.write('TABLE\n')
        f.write('  2\n')
        f.write('LAYER\n')
        f.write(' 70\n')
        f.write('     1\n')
        f.write('  0\n')
        f.write('LAYER\n')
        f.write('  2\n')
        f.write('0\n')
        f.write(' 70\n')
        f.write('     0\n')
        f.write(' 62\n')
        f.write('     7\n')
        f.write('  6\n')
        f.write('CONTINUOUS\n')
        f.write('  0\n')
        f.write('ENDTAB\n')
        f.write('  0\n')
        f.write('ENDSEC\n')

        # BLOCKS
        f.write('  0\n')
        f.write('SECTION\n')
        f.write('  2\n')
        f.write('BLOCKS\n')
        f.write('  0\n')
        f.write('ENDSEC\n')

        # ENTITIES
        f.write('  0\n')
        f.write('SECTION\n')
        f.write('  2\n')
        f.write('ENTITIES\n')

        # Outer square polyline
        f.write('  0\n')
        f.write('POLYLINE\n')
        f.write('  8\n')
        f.write('0\n')
        f.write(' 66\n')
        f.write('     1\n')
        f.write(' 70\n')
        f.write('     1\n')  # closed
        for x, y in outer_points:
            f.write('  0\n')
            f.write('VERTEX\n')
            f.write('  8\n')
            f.write('0\n')
            f.write(f' 10\n{x}\n')
            f.write(f' 20\n{y}\n')
        f.write('  0\n')
        f.write('SEQEND\n')

        # Inner contours polylines
        for contour in inner_contours:
            f.write('  0\n')
            f.write('POLYLINE\n')
            f.write('  8\n')
            f.write('0\n')
            f.write(' 66\n')
            f.write('     1\n')
            f.write(' 70\n')
            f.write('     1\n')  # closed
            for x, y in contour:
                f.write('  0\n')
                f.write('VERTEX\n')
                f.write('  8\n')
                f.write('0\n')
                f.write(f' 10\n{x}\n')
                f.write(f' 20\n{y}\n')
            f.write('  0\n')
            f.write('SEQEND\n')

        f.write('  0\n')
        f.write('ENDSEC\n')
        f.write('  0\n')
        f.write('EOF\n')


def export_lattice_with_outer_square(lattice_tile, desired_size_mm=3.0):
    """Export DXF with an outer square and inner cutout(s) based on 0 regions."""
    filename = get_unique_filename()
    rows, cols = lattice_tile.shape
    scale = desired_size_mm / rows  # scale so full shape is 3 mm x 3 mm

    # 1️⃣ Add the outer square boundary
    outer_square = [
        (0, 0),
        (desired_size_mm, 0),
        (desired_size_mm, desired_size_mm),
        (0, desired_size_mm),
    ]

    # 2️⃣ Find contours for the "holes" (0 regions surrounded by 1’s)
    inverted = 1 - lattice_tile  # flip so 0’s become 1’s
    contours = measure.find_contours(inverted, 0.5)

    inner_contours = []
    for contour in contours:
        coords = [(x * scale, (rows - y) * scale) for y, x in contour]
        inner_contours.append(coords)

    write_dxf_simple(filename, outer_square, inner_contours)

    print(f"✅ DXF saved to {filename}")
    print(f"   Outer size: {desired_size_mm} mm × {desired_size_mm}")


def main():
    txt_path = os.path.join('results2', "lattice_series_1_20.txt")  # Adjusted path to root
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Cannot find {txt_path}")

    lattice_tile = load_lattice_from_txt(txt_path)
    assert lattice_tile.ndim == 2, "Lattice must be 2D!"

    export_lattice_with_outer_square(lattice_tile, desired_size_mm=3.0)


if __name__ == "__main__":
    main()