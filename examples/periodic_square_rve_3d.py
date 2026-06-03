import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(Path(__file__).parent.parent.as_posix())
from fiber_matrix.rve import FiberRVE


def main():
    # Set to False to disable GUI windows from popping up
    show_guis = False

    # Change working directory to a folder for output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    # 1. Initialize RVE
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=77,
        vf=0.60,
        avg_diam=5.0,
        rve_aspect_ratio=1.0,
        # fixed_height = 50,
    )
    print(f"RVE dimensions: {rve.rve_dims}")

    # 2. Place Fibers
    print("Placing fibers...")
    rve.place_initial_fibers(plot_triangulation=show_guis)
    if show_guis:
        rve.draw()
        plt.show()

    # 3. Solve for non-overlapping configuration
    print("Solving to remove overlaps...")
    iterations = rve.solve_fiber_locations(
        min_spacing_ratio=0.1,
        visualization_path=Path(__file__).stem + ".gif",
    )
    print(f"Solved in {iterations} iterations.")
    if show_guis:
        rve.draw()
        plt.show()

    # 4. Generate 3D Mesh (Save to .msh and .vtk files)
    print("Generating 3D mesh...")
    rve.create_3d_mesh(
        mesh_name=Path(__file__).stem,
        thickness=rve.rve_dims[0],
        mesh_size_factor=0.5,
        z_layers=24,
        visualize_gui=show_guis,
        check_periodicity=True,
        periodic_z=False,
        surface_groups=True,
        composite_surface_groups=True,
    )


if __name__ == "__main__":
    main()
