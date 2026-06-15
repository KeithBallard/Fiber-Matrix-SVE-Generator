import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from fiber_matrix.rve import FiberRVE


NUM_FIBERS = 77
VF = 0.60
AVG_DIAM = 5.0
RVE_ASPECT_RATIO = 1.0
MIN_SPACING_RATIO = 0.1
RANDOM_SEED = None

DISTRIBUTION_NAME = "periodic_square_rve_3d_distribution"


def main():
    # Set to False to disable GUI windows from popping up
    show_guis = False

    # Change working directory to a folder for output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    # 1. Initialize RVE
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=NUM_FIBERS,
        vf=VF,
        avg_diam=AVG_DIAM,
        rve_aspect_ratio=RVE_ASPECT_RATIO,
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
        min_spacing_ratio=MIN_SPACING_RATIO,
        visualization_path=Path(__file__).stem + ".gif",
    )
    print(f"Solved in {iterations} iterations.")
    if show_guis:
        rve.draw()
        plt.show()

    centers = np.array([fiber.center for fiber in rve.fibers])
    radii = np.array(rve.radii)
    rve_dims = np.array(rve.rve_dims)

    np.savez(
        DISTRIBUTION_NAME + ".npz",
        centers=centers,
        radii=radii,
        rve_dims=rve_dims,
        num_fibers=NUM_FIBERS,
        vf=VF,
        avg_diam=AVG_DIAM,
        rve_aspect_ratio=RVE_ASPECT_RATIO,
        min_spacing_ratio=MIN_SPACING_RATIO,
        iterations=iterations,
    )

    with open(DISTRIBUTION_NAME + ".csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["fiber_id", "x", "y", "radius"])
        for i, fiber in enumerate(rve.fibers):
            writer.writerow([i, fiber.center[0], fiber.center[1], fiber.radius])

    print(f"Saved distribution: {DISTRIBUTION_NAME}.npz")
    print(f"Saved coordinates: {DISTRIBUTION_NAME}.csv")


if __name__ == "__main__":
    main()
