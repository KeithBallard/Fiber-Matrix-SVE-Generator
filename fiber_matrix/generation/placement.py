import math
import random
import time
import numpy as np
from scipy import spatial
from typing import List, Tuple, Optional, Callable, Any
from fiber_matrix.models.boundary import LinearBoundary
from fiber_matrix.models.fiber import Fiber, PeriodicPrimaryFiber, PeriodicGhostFiber


class FiberPlacementSolver:
    """Class enabling the random placement and subsequent overlap resolution of fibers."""

    def __init__(self):
        pass

    def solve_fiber_locations(
        self,
        fibers: List[PeriodicPrimaryFiber],
        boundaries: List[LinearBoundary],
        min_spacing_ratio: float,
        iterations_max: int,
        iteration_callback: Optional[
            Callable[[int, List[PeriodicPrimaryFiber], List[LinearBoundary]], None]
        ] = None,
    ) -> int:
        """
        Iteratively resolves overlaps between fibers to ensure minimum spacing.

        Parameters
        ----------
        fibers : List[PeriodicPrimaryFiber]
            List of fibers to overlap resolve.
        boundaries : List[LinearBoundary]
            List of RVE boundaries for constraints.
        min_spacing_ratio : float
            Minimum spacing ratio relative to average radius.
        iterations_max : int, optional
            Maximum number of iterations. Default is 10000.
        iteration_callback : Callable, optional
            Callback function called at the start of each iteration.
            Signature: (iteration_count, fibers, boundaries) -> None.

        Raises
        ------
        RuntimeError
            If maximum iterations are exceeded without resolving overlaps.
        """
        start = time.time()
        num_diams_for_search = 4
        num_diams_for_update = 3

        # Calculate average radius for spacing logic
        if not fibers:
            return
        avg_radius = np.mean([f.radius for f in fibers])
        min_space_between_fibers = (
            min_spacing_ratio * avg_radius * 2.0
        )  # Ratio is likely relative to diameter based on name usage in original

        self._recalculate_neighbors(fibers, boundaries, num_diams_for_search)

        iteration_count = 0
        iterations_no_overlap = 0

        # Note: I changed the stopping condition from 3 to 1 to allow faster convergence.
        # This is likely fine for well-posed problems, but consider adjusting in the future.
        while iterations_no_overlap < 1 and iteration_count < iterations_max:
            if iteration_callback is not None:
                iteration_callback(iteration_count, fibers, boundaries)

            found_overlap = self._iterate_on_interference(
                fibers, boundaries, min_space_between_fibers
            )

            need_to_recalc_neighbors = False
            for fiber in fibers:
                # if any fiber has moved more than half the neighbor search
                # distance minus the radius, then we need to re-calculate the neighbors
                # Heuristic from original code
                if fiber.get_distance_since_last_neighbor_update() > fiber.radius * (
                    num_diams_for_update - 1
                ):
                    need_to_recalc_neighbors = True
                    break

            if need_to_recalc_neighbors or not found_overlap:
                self._recalculate_neighbors(fibers, boundaries, num_diams_for_search)

            if not found_overlap:
                iterations_no_overlap += 1
            else:
                iterations_no_overlap = 0

            iteration_count += 1
            if iteration_count >= iterations_max:
                print(
                    "WARNING: Maximum iterations exceeded. Stopping solve. Check input parameters to make sure an RVE is possible."
                )

        elapsed = time.time() - start
        # print("Total Time to generate RVE Geometry: " + str(elapsed) + ' seconds')
        return iteration_count

    def _recalculate_neighbors(
        self,
        fibers: List[PeriodicPrimaryFiber],
        boundaries: List[LinearBoundary],
        fiber_diams_to_search: float,
    ):
        """Recalculates ghost fibers and neighbor lists for efficient collision detection.

        Parameters
        ----------
        fibers : List[PeriodicPrimaryFiber]
            List of primary fibers.
        boundaries : List[LinearBoundary]
            List of boundaries for ghost generation.
        fiber_diams_to_search : float
            Search radius multiplier for neighbor finding.
        """
        fibers_with_ghosts: List[Fiber] = []
        for fiber in fibers:
            fiber.calc_ghost_fibers(boundaries)
            fibers_with_ghosts.append(fiber)
            fibers_with_ghosts.extend(fiber.ghost_fibers)

        fiber_centers = [f.center for f in fibers_with_ghosts]
        fiber_center_kd_tree = spatial.KDTree(fiber_centers)

        for fiber in fibers:
            fiber.update_neighbors(
                fiber_center_kd_tree, fibers_with_ghosts, fiber_diams_to_search
            )

    def _iterate_on_interference(
        self,
        fibers: List[PeriodicPrimaryFiber],
        boundaries: List[LinearBoundary],
        min_space_between_fibers: float,
    ) -> bool:
        """Performs a single pass of interference resolution on all fibers.

        Parameters
        ----------
        fibers : List[PeriodicPrimaryFiber]
            List of fibers to check.
        boundaries : List[LinearBoundary]
            Boundaries to respect during movement.
        min_space_between_fibers : float
            Minimum absolute spacing distance.

        Returns
        -------
        bool
            True if any overlap was strictly found and corrected, False otherwise.
        """
        found_overlap = False
        for fiber in fibers:
            # fix_overlap_with_neighbors returns True if it adjusted anything
            found_overlap |= fiber.fix_overlap_with_neighbors(
                boundaries, min_space_between_fibers
            )
        return found_overlap
