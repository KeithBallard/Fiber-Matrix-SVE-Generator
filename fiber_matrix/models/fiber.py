import copy
import math
import numpy as np
from typing import List, Set, Optional, Any
from .boundary import LinearBoundary, BoundaryType, BoundaryNode


class Fiber:
    """The basic fiber class representing a circular inclusion.

    Parameters
    ----------
    center : np.ndarray
        The coordinates of the fiber center.
    radius : float
        The radius of the fiber.
    """

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.boundary_nodes: List[BoundaryNode] = []
        self.coord_inside_rve: Optional[List[float]] = None
        self.neighbors: Set["Fiber"] = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

    def __str__(self):
        return f"Fiber Center: {self.center}\tr={self.radius}"

    def get_vector_to_other(self, other_fiber: "Fiber") -> np.ndarray:
        """Calculates the vector pointing from this fiber to another fiber.

        Parameters
        ----------
        other_fiber : Fiber
            The target fiber.

        Returns
        -------
        np.ndarray
            Vector from self to other.
        """
        return np.array(other_fiber.center) - np.array(self.center)

    def move(self, move_vec: np.ndarray):
        """Translates the fiber by the given vector.

        Parameters
        ----------
        move_vec : np.ndarray
            Translation vector.
        """
        self.center = self.center + np.array(move_vec)

    def fix_overlap_with_neighbors(
        self, boundaries: List[LinearBoundary], min_space_between_fibers: float
    ) -> bool:
        """Iteratively resolves overlaps with neighboring fibers.

        Parameters
        ----------
        boundaries : List[LinearBoundary]
            List of RVE boundaries for constraint checking.
        min_space_between_fibers : float
            Minimum allowed distance between fiber surfaces.

        Returns
        -------
        bool
            True if any overlap was found and corrected, False otherwise.
        """
        overlap_found = False
        # Sort neighbors for deterministic behavior
        sorted_neighbors = sorted(
            list(self.neighbors), key=lambda f: f.center[0]
        )  # Simple sort by x-coordinate
        for nf in sorted_neighbors:
            vec_to_other = self.get_vector_to_other(nf)
            dist = np.linalg.norm(vec_to_other)
            sum_radii = self.radius + nf.radius
            if dist < sum_radii + min_space_between_fibers:
                overlap_found = True
                unit_vec_to_other = vec_to_other / dist
                proportion_to_move_this_fiber = nf.radius / sum_radii
                proportion_to_move_other_fiber = 1.0 - proportion_to_move_this_fiber
                total_move_back = (
                    sum_radii + min_space_between_fibers - dist + sum_radii * 1e-3
                )

                this_nudge = (
                    -proportion_to_move_this_fiber * total_move_back * unit_vec_to_other
                )
                other_nudge = (
                    proportion_to_move_other_fiber * total_move_back * unit_vec_to_other
                )

                self.move(this_nudge)
                self.adjust_for_bounds(boundaries)
                nf.move(other_nudge)
                nf.adjust_for_bounds(boundaries)
        return overlap_found

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        """Checks to make sure this fiber does not violate any specified boundaries
        and will move it if necessary.
        """
        for boundary in boundaries:
            if boundary.type == BoundaryType.FINITE and boundary.check_collision(
                self.center, self.radius, eps
            ):
                move_vector = boundary.get_move_vector()
                distance_to_center = boundary.get_distance_to_fiber(self.center)
                self.center = (
                    self.center
                    + abs(distance_to_center - (1.0 + eps) * self.radius) * move_vector
                )

            if boundary.type == BoundaryType.SYMMETRIC and boundary.check_collision(
                self.center, self.radius, eps
            ):
                move_vector = boundary.get_move_vector()
                distance_to_center = boundary.get_distance_to_fiber(self.center)
                if distance_to_center < self.radius / 2.0:
                    self.center = (
                        self.center
                        - abs(distance_to_center - (1.0 + eps) * self.radius)
                        * move_vector
                    )
                else:
                    self.center = (
                        self.center
                        + abs(distance_to_center - (1.0 + eps) * self.radius)
                        * move_vector
                    )


class PeriodicPrimaryFiber(Fiber):
    """A class for storing a fiber that is in the RVE domain."""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        index: int,
        boundaries: Optional[List[LinearBoundary]] = None,
        ignore_ghost_fibers=False,
    ):
        super().__init__(center, radius)
        self.index = index
        self.ghost_fibers: List["PeriodicGhostFiber"] = []
        self.ignore_ghost_fibers = ignore_ghost_fibers
        if boundaries is not None:
            self.calc_ghost_fibers(boundaries)

    def calc_ghost_fibers(self, boundaries: List[LinearBoundary]):
        self.ghost_fibers = []
        if self.ignore_ghost_fibers:
            return

        ghost_centers = []
        for boundary in boundaries:
            if boundary.type == BoundaryType.PERIODIC:
                # Need to use the boundary check logic here
                # accessing helper method on boundary to get distance
                if boundary.get_distance_to_fiber(self.center) < 3.0 * self.radius:
                    ghost_center = self.center + boundary.get_periodic_vector()
                    ghost_centers.append(ghost_center)

        # Check intersection with multiple periodic boundaries (corner cases)
        intersected_periodic = []
        for boundary in boundaries:
            if boundary.type == BoundaryType.PERIODIC and boundary.check_collision(
                self.center, self.radius
            ):
                intersected_periodic.append(boundary)

        if len(intersected_periodic) == 2:
            ghost_centers.append(
                intersected_periodic[0].get_periodic_vector()
                + intersected_periodic[1].get_periodic_vector()
                + self.center
            )

        for ghost_center in ghost_centers:
            self.ghost_fibers.append(
                PeriodicGhostFiber(ghost_center, self.radius, self)
            )

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        super().adjust_for_bounds(boundaries, eps)

    def move(self, move_vec: np.ndarray):
        super().move(move_vec)
        self.vec_moved_since_neighbor_update += np.array(move_vec)
        for ghost in self.ghost_fibers:
            ghost.move_actual(move_vec)

    def get_distance_since_last_neighbor_update(self) -> float:
        return np.linalg.norm(self.vec_moved_since_neighbor_update)

    def get_all_copies(self) -> List[Fiber]:
        fibers = list(self.ghost_fibers)
        fibers.append(self)
        return fibers

    def update_neighbors(
        self,
        fiber_center_kd_tree,
        fibers_with_ghosts: List[Fiber],
        fiber_diams_to_search: float,
    ):
        self.neighbors = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

        # KDTree query returns distances and indices
        dists, indices = fiber_center_kd_tree.query(
            self.center,
            k=100,
            p=2,
            distance_upper_bound=self.radius * 2.0 * fiber_diams_to_search,
        )

        # Filter indices
        valid_indices = [i for i in indices if i < len(fibers_with_ghosts)]

        for i in valid_indices:
            other = fibers_with_ghosts[i]
            # Check logic from original: neighbors should have higher index to avoid double checking pairs
            if isinstance(other, PeriodicPrimaryFiber):
                if other.index > self.index:
                    self.neighbors.add(other)
            elif isinstance(other, PeriodicGhostFiber):
                if other.primary_fiber.index > self.index:
                    self.neighbors.add(other)

        for gf in self.ghost_fibers:
            gf.update_neighbors(
                fiber_center_kd_tree, fibers_with_ghosts, fiber_diams_to_search
            )

    def fix_overlap_with_neighbors(
        self, boundaries: List[LinearBoundary], min_space_between_fibers: float
    ) -> bool:
        found_overlap = super().fix_overlap_with_neighbors(
            boundaries, min_space_between_fibers
        )
        for ghost_fib in self.ghost_fibers:
            found_overlap |= ghost_fib.fix_overlap_with_neighbors(
                boundaries, min_space_between_fibers
            )
        return found_overlap


class PeriodicGhostFiber(Fiber):
    def __init__(
        self, center: np.ndarray, radius: float, primary_fiber: PeriodicPrimaryFiber
    ):
        super().__init__(center, radius)
        self.primary_fiber = primary_fiber

    def move(self, move_vec: np.ndarray):
        self.primary_fiber.move(move_vec)

    def move_actual(self, move_vec: np.ndarray):
        super().move(move_vec)

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        self.primary_fiber.adjust_for_bounds(boundaries, eps)

    def update_neighbors(
        self,
        fiber_center_kd_tree,
        fibers_with_ghosts: List[Fiber],
        fiber_diams_to_search: float,
    ):
        self.neighbors = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

        dists, indices = fiber_center_kd_tree.query(
            self.center,
            k=100,
            p=2,
            distance_upper_bound=self.radius * 2.0 * fiber_diams_to_search,
        )
        valid_indices = [i for i in indices if i < len(fibers_with_ghosts)]

        for i in valid_indices:
            other = fibers_with_ghosts[i]
            if isinstance(other, PeriodicPrimaryFiber):
                if other.index > self.primary_fiber.index:
                    self.neighbors.add(other)
            elif isinstance(other, PeriodicGhostFiber):
                if other.primary_fiber.index > self.primary_fiber.index:
                    self.neighbors.add(other)
