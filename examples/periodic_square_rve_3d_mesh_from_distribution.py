import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from fiber_matrix.rve import FiberRVE

UNIFORM_MESH = False  # Set to False to use different mesh sizes for fiber, matrix, and boundary regions
FIBER_MESH_SIZE = 0.1
MATRIX_MESH_SIZE = 1.5
BOUNDARY_MESH_SIZE = 0.5
INTERFACE_REFINEMENT_DISTANCE = 0.5
BOUNDARY_REFINEMENT_DISTANCE = 0.5
Z_LAYERS = 24
RECOMBINE_PRISMS = True
DISTRIBUTION_NAME = "periodic_square_rve_3d_distribution.npz"
MESH_NAME = (
    f"periodic_square_rve_3d_mesh_z{Z_LAYERS}"
    + ("_prism" if RECOMBINE_PRISMS else "_tet")
    + (f"_uniform_{MATRIX_MESH_SIZE}" if UNIFORM_MESH else f"_fiber{FIBER_MESH_SIZE}_matrix{MATRIX_MESH_SIZE}_boundary{BOUNDARY_MESH_SIZE}_ifdist{INTERFACE_REFINEMENT_DISTANCE}_bdist{BOUNDARY_REFINEMENT_DISTANCE}")
)

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
    print(f"Generating 3D mesh: {MESH_NAME}")

    rve.create_3d_mesh(
        mesh_name=MESH_NAME,
        thickness=rve.rve_dims[0],
        mesh_size_factor=MATRIX_MESH_SIZE,
        z_layers=Z_LAYERS,
        visualize_gui=show_guis,
        check_periodicity=True,
        periodic_z=False,
        surface_groups=True,
        composite_surface_groups=False,
        uniform_mesh=UNIFORM_MESH,
        fiber_mesh_size=FIBER_MESH_SIZE,
        matrix_mesh_size=MATRIX_MESH_SIZE,
        boundary_mesh_size=BOUNDARY_MESH_SIZE,
        interface_refinement_distance=INTERFACE_REFINEMENT_DISTANCE,
        boundary_refinement_distance=BOUNDARY_REFINEMENT_DISTANCE,
        recombine_prisms=RECOMBINE_PRISMS,
    )


if __name__ == "__main__":
    main()
