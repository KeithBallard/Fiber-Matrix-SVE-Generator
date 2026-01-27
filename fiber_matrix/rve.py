import math
import random
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .models.boundary import LinearBoundary, BoundaryType
from .models.fiber import Fiber, PeriodicPrimaryFiber
from .generation.placement import FiberPlacementSolver
from .meshing.gmsh_mesher import GmshMesher
from .visualization import plotting

try:
    from triangle import triangulate
except ImportError:
    triangulate = None


class FiberRVE:
    """A class that defines a fiber/matrix RVE.

    This facade coordinates the placement of fibers, boundary definitions,
    and meshing/visualization.
    """

    def __init__(self):
        self.fibers: List[PeriodicPrimaryFiber] = []
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
        boundary_type: BoundaryType = BoundaryType.PERIODIC,
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
        boundary_types = [boundary_type for _ in boundary_points]
        self._create_boundaries_from_points(boundary_points, boundary_types)

        return self.rve_dims

    def initialize_hexagonal_rve(
        self,
        num_fibers: int,
        vf: float = 0.6,
        avg_diam: float = 5e-6,
        diam_std_dev: float = 0.0,
        boundary_type: BoundaryType = BoundaryType.PERIODIC,
    ) -> List[float]:
        """Calculates dimensions and initializes a regular hexagonal RVE.

        Parameters
        ----------
        num_fibers : int
            Target number of fibers.
        vf : float, optional
            Target fiber volume fraction. Default 0.6.
        avg_diam : float, optional
            Average fiber diameter. Default 5e-6.
        diam_std_dev : float, optional
            Standard deviation of fiber diameter. Default 0.0.
        boundary_type : BoundaryType, optional
            Type of boundary to apply to all edges. Default PERIODIC.

        Returns
        -------
        List[float]
            The side length of the hexagon [side_length].
        """
        self.radii = [
            a / 2
            for a in diam_std_dev * avg_diam * np.random.randn(num_fibers) + avg_diam
        ]
        total_fiber_area = sum([math.pi * a * a for a in self.radii])
        rve_area = total_fiber_area / vf

        # Area of regular hexagon = (3 * sqrt(3) / 2) * s^2
        s = math.sqrt(2.0 * rve_area / (3.0 * math.sqrt(3.0)))
        self.rve_dims = [s]

        # Vertices for a "flat-topped" hexagon
        # V0 = [s, 0]
        # V1 = [s/2, s*sqrt(3)/2]
        # V2 = [-s/2, s*sqrt(3)/2]
        # V3 = [-s, 0]
        # V4 = [-s/2, -s*sqrt(3)/2]
        # V5 = [s/2, -s*sqrt(3)/2]
        # We offset so V4 is roughly at origin-ish area if needed, but [0,0] center is cleaner.
        h_val = s * math.sqrt(3.0) / 2.0
        boundary_points = [
            np.array([s, 0.0]),
            np.array([s / 2.0, h_val]),
            np.array([-s / 2.0, h_val]),
            np.array([-s, 0.0]),
            np.array([-s / 2.0, -h_val]),
            np.array([s / 2.0, -h_val]),
        ]
        boundary_types = [boundary_type for _ in boundary_points]
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

    @property
    def boundary_points(self) -> List[np.ndarray]:
        """Returns the vertices of the RVE boundary in order."""
        return [b.points[0] for b in self.boundaries]

    def place_initial_fibers(
        self,
        specified_fiber_centers: List[List[float]] = None,
        plot_triangulation: bool = False,
    ) -> Tuple[float, float]:
        """Places fibers within the RVE domain, using triangulation for random placement.

        Parameters
        ----------
        specified_fiber_centers : List[List[float]], optional
            List of [x, y] coordinates to force fiber locations.
        plot_triangulation : bool, optional
            If True and matplotlib is available, plots the RVE triangulation.

        Returns
        -------
        Tuple[float, float]
            Actual Fiber Volume Fraction (VF) and Total RVE Area.
        """
        if specified_fiber_centers is None:
            specified_fiber_centers = []

        points = np.array(self.boundary_points)
        if len(points) < 3:
            raise ValueError("RVE must have at least 3 boundary points to triangulate.")

        # Triangulate the polygon
        tri_in = {"vertices": points}
        tri_in["segments"] = [[i, (i + 1) % len(points)] for i in range(len(points))]

        if triangulate is None:
            raise ImportError(
                "The 'triangle' library is required for initial fiber placement. "
                "Please install it with 'pip install triangle'."
            )

        self.triangulation = triangulate(tri_in, "p")

        if plot_triangulation and plt is not None:
            fig, ax = plt.subplots()
            x = points[:, 0]
            y = points[:, 1]
            for t in self.triangulation["triangles"]:
                t_idx = [t[0], t[1], t[2], t[0]]
                ax.plot(x[t_idx], y[t_idx], "k-", alpha=0.3)
            plt.show()

        def get_tri_area(a, b, c):
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

        areas = []
        total_weighted_centroid = np.array([0.0, 0.0])
        for t in self.triangulation["triangles"]:
            a, b, c = points[t[0]], points[t[1]], points[t[2]]
            tri_area = get_tri_area(a, b, c)
            tri_centroid = (a + b + c) / 3.0
            total_weighted_centroid += tri_area * tri_centroid
            areas.append(tri_area)

        rve_area = sum(areas)
        assert rve_area > 0, "RVE area is zero."
        centroid = total_weighted_centroid / rve_area

        def weighted_choice(choices, weights):
            total = sum(weights)
            r = random.uniform(0, total)
            upto = 0
            for i, w in enumerate(weights):
                if upto + w >= r:
                    return choices[i]
                upto += w
            return choices[-1]

        self.fibers = []
        num_fibers = len(self.radii)

        if not specified_fiber_centers:
            if num_fibers == 1:
                specified_fiber_centers.append(centroid.tolist())
            else:
                # Random placement within triangles
                choice_indices = list(range(len(areas)))
                randvals = np.random.rand(num_fibers, 2)
                for i in range(num_fibers):
                    tri_idx = weighted_choice(choice_indices, areas)
                    t = self.triangulation["triangles"][tri_idx]

                    # Unified random point in triangle algorithm
                    # (1 - sqrt(r1))*A + sqrt(r1)*(1 - r2)*B + sqrt(r1)*r2*C
                    r1, r2 = randvals[i]
                    sqrt_r1 = math.sqrt(r1)
                    random_point = (
                        (1 - sqrt_r1) * points[t[0]]
                        + sqrt_r1 * (1 - r2) * points[t[1]]
                        + sqrt_r1 * r2 * points[t[2]]
                    )

                    self.fibers.append(
                        PeriodicPrimaryFiber(
                            random_point, self.radii[i], i, self.boundaries
                        )
                    )
        else:
            if len(specified_fiber_centers) != num_fibers:
                raise ValueError(
                    f"Number of specified centers ({len(specified_fiber_centers)}) "
                    f"does not match number of fibers ({num_fibers})."
                )
            for i in range(num_fibers):
                self.fibers.append(
                    PeriodicPrimaryFiber(
                        np.array(specified_fiber_centers[i]),
                        self.radii[i],
                        i,
                        self.boundaries,
                    )
                )

        for fiber in self.fibers:
            fiber.adjust_for_bounds(self.boundaries)

        self.fiber_vf = sum([math.pi * r**2 for r in self.radii]) / rve_area
        return (self.fiber_vf, rve_area)

    def is_inside_rve(self, point: np.ndarray) -> bool:
        """Checks if a point is strictly inside the RVE boundaries."""
        for b in self.boundaries:
            # Vector from p0 to p1
            v1 = b.points[1] - b.points[0]
            # Vector from p0 to point
            v2 = point - b.points[0]
            # 2D Cross product: x1*y2 - x2*y1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < -1e-12:
                return False
        return True

    def solve_fiber_locations(
        self,
        min_spacing_ratio: float,
        visualization_path: Optional[str] = None,
        show_final: bool = False,
        iterations_max: int = 100,
    ):
        """Executes the solver to resolve fiber overlaps and enforce spacing.

        Parameters
        ----------
        min_spacing_ratio : float
            Minimum separation between fibers as a ratio of the average diameter.
        visualization_path : Optional[str], optional
            If provided, saves an animation of the solution process to the given path. Default None.
        show_final : bool, optional
            If True, displays the final RVE plot to screen. Default False.
        """
        if not self.fibers:
            self.place_initial_fibers()

        solver = FiberPlacementSolver()

        # Visualization setup
        captured_frames = []
        fig = None
        ax = None

        if visualization_path and plt is not None:
            # Create a single figure for the animation
            fig, ax = plt.subplots(figsize=(6, 6))

            def plot_callback(iteration, fibers, boundaries):
                # Update the existing plot
                self.draw(fig=fig, ax=ax, frame=None)

                # Force draw to update canvas
                fig.canvas.draw()

                # Capture the frame from buffer
                # Note: safe approach for different backends
                try:
                    from PIL import Image

                    w, h = fig.canvas.get_width_height()
                    # buffer_rgba() returns a memoryview of the RGBA buffer
                    buf = fig.canvas.buffer_rgba()
                    image = Image.frombuffer("RGBA", (w, h), buf).convert("RGB")
                    captured_frames.append(image)
                except ImportError:
                    print("PIL (Pillow) not found. Cannot save GIF.")
                except AttributeError:
                    # Fallback for older matplotlib or different backends
                    try:
                        buf = fig.canvas.tostring_rgb()
                        image = Image.frombytes("RGB", (w, h), buf)
                        captured_frames.append(image)
                    except Exception as e:
                        print(f"Error capturing frame (fallback failed): {e}")
                except Exception as e:
                    print(f"Error capturing frame: {e}")

            callback = plot_callback
        elif visualization_path:
            print("Matplotlib not installed. Visualization disabled.")
            callback = None

        try:
            iterations = solver.solve_fiber_locations(
                self.fibers,
                self.boundaries,
                min_spacing_ratio,
                iteration_callback=callback,
                iterations_max=iterations_max,
            )
            return iterations
        finally:
            # Save GIF if we have frames
            if captured_frames:
                try:
                    captured_frames[0].save(
                        visualization_path,
                        save_all=True,
                        append_images=captured_frames[1:],
                        optimize=False,
                        duration=200,  # ms per frame
                        loop=0,
                    )
                    print(
                        f"Animation saved to '{visualization_path}' ({len(captured_frames)} frames)."
                    )
                except Exception as e:
                    print(f"Failed to save GIF: {e}")

            # Explicitly close the figure to avoid RuntimeWarning
            if fig is not None:
                plt.close(fig)

        if show_final:
            self.draw()

    def get_fiber_centers(self) -> List[np.ndarray]:
        """Returns the current centers of all primary fibers.

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
        self,
        mesh_name="FiberMatrixRVE",
        mesh_size_factor=1.0,
        visualize_gui=False,
        check_periodicity=False,
    ):
        """Generates a mesh for the current RVE configuration using GMSH.

        Parameters
        ----------
        mesh_name : str, optional
            Base name for output files (.msh, .vtk). Default "FiberMatrixRVE".
        mesh_size_factor : float, optional
            Global scaling factor for mesh element size. Default 1.0.
        visualize_gui : bool, optional
            If True, opens the GMSH GUI to visualize mesh generation steps. Default False.
        check_periodicity : bool, optional
            If True, asserts that the generated mesh nodes on periodic boundaries match. Default False.
        """
        mesher = GmshMesher(mesh_name)
        mesher.create_mesh(
            self.fibers,
            self.boundaries,
            mesh_size_factor,
            visualize_gui,
            check_periodicity,
        )
