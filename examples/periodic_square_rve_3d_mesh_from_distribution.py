import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from fiber_matrix.rve import FiberRVE

MESH_SIZE_FACTOR = 0.5
Z_LAYERS = 24
DISTRIBUTION_NAME = "periodic_square_rve_3d_distribution.npz"
MESH_NAME = "periodic_square_rve_3d_mesh_"+ str(MESH_SIZE_FACTOR) + "_z" + str(Z_LAYERS)    




def main():
    # Set to False to disable GUI windows from popping up
    show_guis = False

    # Change working directory to the folder with saved distribution/output files
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    distribution_path = Path(DISTRIBUTION_NAME)
    if not distribution_path.exists():
        raise FileNotFoundError(
            f"Distribution file '{distribution_path}' not found. "
            "Run periodic_square_rve_3d_save_distribution.py first."
        )

    data = np.load(distribution_path)
    centers = data["centers"]
    radii = data["radii"]
    rve_dims = data["rve_dims"]
    num_fibers = int(data["num_fibers"])
    vf = float(data["vf"])
    rve_area = float(rve_dims[0] * rve_dims[1])
    equivalent_avg_diam = 2.0 * np.sqrt((vf * rve_area / num_fibers) / np.pi)

    # Recreate the same periodic square RVE domain.
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=num_fibers,
        vf=vf,
        avg_diam=equivalent_avg_diam,
        rve_aspect_ratio=float(data["rve_aspect_ratio"]),
        fixed_height=float(rve_dims[1]),
    )
    rve.radii = radii.tolist()
    rve.place_initial_fibers(specified_fiber_centers=centers.tolist())

    print(f"Loaded distribution: {distribution_path}")
    print(f"RVE dimensions: {rve.rve_dims}")
    print(f"Generating 3D mesh with mesh_size_factor={MESH_SIZE_FACTOR}")

    rve.create_3d_mesh(
        mesh_name=MESH_NAME,
        thickness=rve.rve_dims[0],
        mesh_size_factor=MESH_SIZE_FACTOR,
        z_layers=Z_LAYERS,
        visualize_gui=show_guis,
        check_periodicity=True,
        periodic_z=False,
        surface_groups=True,
        composite_surface_groups=False,
    )


if __name__ == "__main__":
    main()
