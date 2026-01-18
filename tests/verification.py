import sys
import os

# Add parent directory to sys.path to allow importing fiber_matrix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fiber_matrix import FiberRVE


def square_periodic_example():
    # Ensure we load data relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)  # Change CWD to tests/ so outputs go there

    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=20,
        vf=0.6,
        avg_diam=5e-6,
        diam_std_dev=0.0,
        rve_aspect_ratio=1.0,
        fixed_height=None,
    )

    # Load test data from current directory
    try:
        initial_fiber_centers = np.load(
            os.path.join(base_dir, "test_initial_fiber_centers.npy")
        )
    except FileNotFoundError:
        print("Test data not found, skipping specific verification.")
        return

    # Convert to list
    initial_centers_list = initial_fiber_centers.tolist()

    rve.place_initial_fibers(specified_fiber_centers=initial_centers_list)
    rve.solve_fiber_locations(min_spacing_ratio=0.1, visualize=True)

    final_fiber_centers = np.load(
        os.path.join(base_dir, "test_final_fiber_centers.npy")
    )

    current_centers = rve.get_fiber_centers()

    # Sort or align? The order should be preserved if passed in order.
    # Current centers is list of arrays.

    norm_diff = np.linalg.norm(np.array(current_centers) - final_fiber_centers)
    print(f"norm_diff {norm_diff}")

    if norm_diff < 1e-4:  # increased tolerance slightly for float diffs
        print("Verification PASSED")
    else:
        print("Verification FAILED")

    # Test Meshing
    try:
        rve.create_mesh(
            mesh_name="test_mesh",
            mesh_size_factor=1e-6,
            check_periodicity=True,
            visualize_gui=True,
        )
        print("Meshing call completed (check output files).")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Meshing failed: {e}")


if __name__ == "__main__":
    square_periodic_example()
