import math
import random
import numpy as np
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .models.boundary import LinearBoundary, BoundaryType
from .models.fiber import Fiber, PeriodicMasterFiber
from .generation.placement import FiberPlacementSolver
from .meshing.gmsh_mesher import GmshMesher
from .visualization import plotting


class FiberRVE:
    """A class that defines a fiber/matrix RVE.

    This facade coordinates the placement of fibers, boundary definitions,
    and meshing/visualization.
    """

    def __init__(self):
        self.fibers: List[PeriodicMasterFiber] = []
        self.boundaries: List[LinearBoundary] = []
        self.fiber_vf: float = 0.0
        self.radii: List[float] = []
        self.rve_dims: List[float] = []

    # ---------------------
    # Initialization
    # ---------------------

    def initialize_rectangle_rve(
        self,
        num_fibers: int,
        vf: float = 0.6,
        avg_diam: float = 5e-6,
        diam_std_dev: float = 0.0,
        rve_aspect_ratio: float = 1.0,
        fixed_height: Optional[float] = None,
    ) -> List[float]:
        """Calculates dimensions of the RVE and generates fiber radii distribution.

        Parameters
        ----------
        num_fibers : int
            Target number of fibers.
        vf : float, optional
            Target fiber volume fraction. Default 0.6.
        avg_diam : float, optional
            Average fiber diameter. Default 5e-6.
        diam_std_dev : float, optional
            Standard deviation of fiber diameter relative to average (scale factor). Default 0.0.
        rve_aspect_ratio : float, optional
            Ratio of width to height (w/h). Default 1.0.
        fixed_height : float, optional
            If provided, fixes RVE height and calculates width to satisfy VF.

        Returns
        -------
        List[float]
            The calculated RVE dimensions [width, height].
        """

        self.radii = [
            a / 2
            for a in diam_std_dev * avg_diam * np.random.randn(num_fibers) + avg_diam
        ]
        total_fiber_area = sum([math.pi * a * a for a in self.radii])
        rve_area = total_fiber_area / vf

        if fixed_height is None:
            h = math.sqrt(rve_area / rve_aspect_ratio)
        else:
            h = fixed_height
        w = rve_area / h
        self.rve_dims = [w, h]

        boundary_points = [
            np.array([0.0, 0.0]),
            np.array([w, 0.0]),
            np.array([w, h]),
            np.array([0.0, h]),
        ]
        boundary_types = [BoundaryType.PERIODIC for _ in boundary_points]
        self._create_boundaries_from_points(boundary_points, boundary_types)

        return self.rve_dims

    def _create_boundaries_from_points(
        self, boundary_points: List[np.ndarray], boundary_types: List[BoundaryType]
    ):
        self.boundaries = []
        num_points = len(boundary_points)
        for i in range(num_points):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % num_points]
            # Indices logic from original code is kept for completeness but simplified
            b = LinearBoundary(
                points=[p1, p2],
                point_indices=[i, (i + 1) % num_points],
                btype=boundary_types[i],
            )
            b.index = i
            self.boundaries.append(b)

        self._assign_boundary_pairs()

    def _assign_boundary_pairs(self):
        """Identifies and assigns periodic pairs for boundaries within the RVE.

        Raises
        ------
        ValueError
            If a periodic boundary is paired with a non-periodic boundary.
        """
        for boundary in self.boundaries:
            for boundary_to_check in self.boundaries:
                if boundary.is_pair(boundary_to_check):
                    boundary.pair = boundary_to_check
                    boundary_to_check.pair = boundary
                    if (
                        boundary.type == BoundaryType.PERIODIC
                        or boundary.pair.type == BoundaryType.PERIODIC
                    ):
                        if boundary.type != boundary.pair.type:
                            raise ValueError(
                                "One boundary of a pair was periodic but the other was not."
                            )

    # ---------------------
    # Placement
    # ---------------------

    def place_initial_fibers(
        self, specified_fiber_centers: List[List[float]] = None
    ) -> Tuple[float, float]:
        """Places fibers randomly within the RVE or at specified locations.

        Parameters
        ----------
        specified_fiber_centers : List[List[float]], optional
            List of [x, y] coordinates to force fiber locations.
            If None or empty, random placement is used.

        Returns
        -------
        Tuple[float, float]
            Actual Fiber Volume Fraction (VF) and Total RVE Area.

        Raises
        ------
        ValueError
            If specified_fiber_centers is shorter than the number of initialized fibers.
        """
        if specified_fiber_centers is None:
            specified_fiber_centers = []

        # Simple random placement if not specified
        # Note: Original code had complex triangulation logic for arbitrary polygons.
        # For rectangular RVE, we can simplify or re-implement the triangulation if needed.
        # For now, implementing simple random placement in bbox if rectangle.

        if not specified_fiber_centers:
            # Random placement within bounds
            # Assuming rectangle for now based on initialize_rectangle_rve reuse
            min_x = min(p[0] for b in self.boundaries for p in b.points)
            max_x = max(p[0] for b in self.boundaries for p in b.points)
            min_y = min(p[1] for b in self.boundaries for p in b.points)
            max_y = max(p[1] for b in self.boundaries for p in b.points)

            for i in range(len(self.radii)):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                specified_fiber_centers.append([x, y])

        num_fibers = len(self.radii)
        if len(specified_fiber_centers) != num_fibers:
            # Just truncate or pad? Original raised error.
            if len(specified_fiber_centers) < num_fibers:
                raise ValueError("Not enough centers specified")

        self.fibers = []
        for i in range(num_fibers):
            r = self.radii[i]
            c = np.array(specified_fiber_centers[i])
            f = PeriodicMasterFiber(c, r, i, self.boundaries)
            self.fibers.append(f)

        for fiber in self.fibers:
            fiber.adjust_for_bounds(self.boundaries)

        # Re-calc VF
        # Need area. For rectangle:
        area = (
            self.rve_dims[0] * self.rve_dims[1] if self.rve_dims else 1.0
        )  # Approximation if arbitrary
        self.fiber_vf = sum([math.pi * r**2 for r in self.radii]) / area
        return (self.fiber_vf, area)

    def solve_fiber_locations(
        self,
        min_spacing_ratio: float,
        visualize: bool = False,
        show_final: bool = False,
    ):
        """Executes the solver to resolve fiber overlaps and enforce spacing.

        Parameters
        ----------
        min_spacing_ratio : float
            Minimum separation between fibers as a ratio of the average diameter.
        visualize : bool, optional
            If True, saves frames of the solving process to 'frames/' directory. Default False.
        show_final : bool, optional
            If True, displays the final RVE plot to screen. Default False.
        """
        solver = FiberPlacementSolver()

        callback = None
        if visualize:
            import shutil
            import os

            if os.path.exists("frames"):
                shutil.rmtree("frames")

            # Use a mutable container for frame counter if needed, or just use iteration count
            # The solver passes iteration count.
            def plot_callback(iteration, fibers, boundaries):
                # We can reuse self.draw, but we need to ensure it saves using the iteration number
                self.draw(frame=iteration)

            callback = plot_callback

        solver.solve_fiber_locations(
            self.fibers, self.boundaries, min_spacing_ratio, iteration_callback=callback
        )

        if show_final:
            self.draw()

    def get_fiber_centers(self) -> List[np.ndarray]:
        """Returns the current centers of all master fibers.

        Returns
        -------
        List[np.ndarray]
            List of fiber center coordinates.
        """
        return [f.center for f in self.fibers]

    # ---------------------
    # Visualization
    # ---------------------

    def draw(
        self, fig=None, ax=None, frame=None, label_fibers=False, label_boundaries=False
    ):
        """Draws the current state of the RVE.

        Parameters
        ----------
        fig : Figure, optional
            Matplotlib figure.
        ax : Axes, optional
            Matplotlib axes.
        frame : int, optional
            Frame number for saving animation sequences.
        label_fibers : bool, optional
            Label fibers with indices.
        label_boundaries : bool, optional
            Label boundaries with indices.
        """
        return plotting.draw_rve(
            self.fibers, self.boundaries, fig, ax, frame, label_fibers, label_boundaries
        )

    # ---------------------
    # Meshing
    # ---------------------

    def create_mesh(
        self, mesh_name="FiberMatrixRVE", mesh_size_factor=1.0, visualize_gui=False
    ):
        """Generates a mesh for the current RVE configuration using GMSH.

        Parameters
        ----------
        mesh_name : str, optional
            Base name for output files (.msh, .vtk). Default "FiberMatrixRVE".
        mesh_size_factor : float, optional
            Global scaling factor for mesh element size. Default 1.0.
        visualize_gui : bool, optional
            If True, opens the GMSH GUI to visualize the mesh after generation. Default False.
        """
        mesher = GmshMesher(mesh_name)
        mesher.create_mesh(
            self.fibers, self.boundaries, mesh_size_factor, visualize_gui
        )
