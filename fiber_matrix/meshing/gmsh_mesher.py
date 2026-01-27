import sys
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

try:
    import gmsh
except ImportError:
    gmsh = None

from fiber_matrix.models.boundary import LinearBoundary, BoundaryType
from fiber_matrix.models.fiber import Fiber, PeriodicPrimaryFiber


class GmshMesher:
    """Handles meshing of the RVE using GMSH."""

    def __init__(self, mesh_name: str = "FiberMatrixRVE"):
        """
        Parameters
        ----------
        mesh_name : str, optional
            The name prefix for generated mesh files. Default is "FiberMatrixRVE".
        """
        self.mesh_name = mesh_name
        self._check_gmsh()

    def _check_gmsh(self):
        """Checks if the gmsh library is imported.

        Raises
        ------
        ImportError
            If gmsh is not available.
        """
        if gmsh is None:
            raise ImportError(
                "GMSH python library is not installed. Please install it using 'pip install gmsh'."
            )

    def create_mesh(
        self,
        fibers: List[PeriodicPrimaryFiber],
        boundaries: List[LinearBoundary],
        mesh_size_factor: float = 1.0,
        visualize_gui: bool = False,
        check_periodicity: bool = False,
    ):
        """Creates the mesh using GMSH.

        Parameters
        ----------
        fibers : List[PeriodicPrimaryFiber]
            List of fibers to include in the mesh.
        boundaries : List[LinearBoundary]
            List of boundaries defining the RVE.
        mesh_size_factor : float, optional
            Factor to control mesh refinement. Default is 1.0.
        visualize_gui : bool, optional
            If True, launches GMSH GUI to visualize the geometry/mesh. Default is False.
        check_periodicity : bool, optional
            If True, asserts that the generated mesh nodes on periodic boundaries match. Default is False.

        Notes
        -----
        To ensure robust boolean operations (OpenCASCADE kernel), the geometry is temporarily
        scaled up such that the RVE extent is order ~1.0. This avoids precision issues with
        very small coordinates (e.g. 1e-6). The geometry is scaled back to original size
        after boolean fragmenting and before mesh generation.
        """

        gmsh.initialize()
        gmsh.model.add(self.mesh_name)

        # Use OpenCASCADE kernel for robust boolean operations

        # 1. Create RVE Polygon
        # The boundaries might not be in order. We need to chain them.

        # Gather all points for scaling extent
        all_b_points = []
        for b in boundaries:
            all_b_points.append(b.points[0])
            all_b_points.append(b.points[1])
        all_b_points = np.array(all_b_points)

        # Boolean operations require the points to be sufficiently large in magnitude
        rve_extent = np.linalg.norm(
            np.max(all_b_points, axis=0) - np.min(all_b_points, axis=0)
        )
        scale_factor = 1.0 / rve_extent

        # Sort boundaries to form a continuous loop
        ordered_chain = []  # List of (LinearBoundary, start_point, end_point)
        remaining_boundaries = list(boundaries)

        # Pick the first one
        if not remaining_boundaries:
            raise ValueError("No boundaries provided.")

        current_b = remaining_boundaries.pop(0)
        # Orientation of the first one defines the loop direction
        current_start = current_b.points[0] * scale_factor
        current_end = current_b.points[1] * scale_factor

        ordered_chain.append((current_b, current_start, current_end))

        # Iteratively find the next connected boundary
        while remaining_boundaries:
            found_idx = -1
            found_orientation = 0  # 0: p0->p1, 1: p1->p0

            for i, b in enumerate(remaining_boundaries):
                p0 = b.points[0] * scale_factor
                p1 = b.points[1] * scale_factor

                # Check connectivity to current_end
                if np.linalg.norm(p0 - current_end) < 1e-4:
                    found_idx = i
                    found_orientation = 0
                    break
                elif np.linalg.norm(p1 - current_end) < 1e-4:
                    found_idx = i
                    found_orientation = 1
                    break

            if found_idx == -1:
                raise RuntimeError(
                    f"Could not find connected boundary in loop during meshing. Current tip: {current_end}"
                )

            b = remaining_boundaries.pop(found_idx)
            if found_orientation == 0:
                ordered_chain.append(
                    (b, b.points[0] * scale_factor, b.points[1] * scale_factor)
                )
                current_end = b.points[1] * scale_factor
            else:
                ordered_chain.append(
                    (b, b.points[1] * scale_factor, b.points[0] * scale_factor)
                )
                current_end = b.points[0] * scale_factor

        # Create GMSH Lines
        occ_line_tags = []
        first_pt_tag = gmsh.model.occ.addPoint(
            ordered_chain[0][1][0], ordered_chain[0][1][1], 0
        )
        prev_pt_tag = first_pt_tag

        for i in range(len(ordered_chain)):
            item = ordered_chain[i]
            # If last segment, connect to first point
            if i == len(ordered_chain) - 1:
                next_pt_tag = first_pt_tag
            else:
                p_end = item[2]
                next_pt_tag = gmsh.model.occ.addPoint(p_end[0], p_end[1], 0)

            l_tag = gmsh.model.occ.addLine(prev_pt_tag, next_pt_tag)
            occ_line_tags.append(l_tag)
            prev_pt_tag = next_pt_tag

        rve_wire = gmsh.model.occ.addWire(occ_line_tags)
        rve_face = gmsh.model.occ.addPlaneSurface([rve_wire])

        # 2. Add Fibers (Disks)
        fiber_disks = []

        def add_fiber_disk(f):
            return gmsh.model.occ.addDisk(
                f.center[0] * scale_factor,
                f.center[1] * scale_factor,
                0,
                f.radius * scale_factor,
                f.radius * scale_factor,
            )

        for f in fibers:
            fiber_disks.append(add_fiber_disk(f))
            for g in f.ghost_fibers:
                fiber_disks.append(add_fiber_disk(g))

        gmsh.model.occ.synchronize()

        if visualize_gui:
            gmsh.fltk.run()

        # 3. Clip fibers to RVE using Intersect
        # We want the part of fibers INSIDE the RVE.
        # Object=Fibers, Tool=RVE

        rve_dimtag = (2, rve_face)
        fiber_dimtags = [(2, t) for t in fiber_disks]

        # Intersect(Fibers, RVE). removeObject=True (consume fibers), removeTool=False (keep RVE).
        clipped_fibers_dimtags, _ = gmsh.model.occ.intersect(
            fiber_dimtags, [rve_dimtag], removeObject=True, removeTool=False
        )

        gmsh.model.occ.synchronize()

        if visualize_gui:
            gmsh.fltk.run()

        # 4. Fragment
        # Embed the clipped fibers into the RVE face.
        out_dimtags, out_dimtags_map = gmsh.model.occ.fragment(
            [rve_dimtag], clipped_fibers_dimtags
        )

        if not out_dimtags_map:
            raise RuntimeError(
                f"GMSH Fragment operation returned empty map. "
                f"rve_dimtag={rve_dimtag}, # clipped_fibers={len(clipped_fibers_dimtags)}"
            )

        # Now that boolean operations are done, we can scale the geometry back to its original size.
        gmsh.model.occ.dilate(
            out_dimtags,
            0,
            0,
            0,
            1.0 / scale_factor,
            1.0 / scale_factor,
            1.0 / scale_factor,
        )

        gmsh.model.occ.synchronize()
        if visualize_gui:
            gmsh.fltk.run()

        # 5. Identify Matrix vs Fibers
        # use out_dimtags_map from fragment operation to trace lineage.

        # Collect all tags that come from the fiber inputs
        fiber_surface_tags = set()
        # Inputs to fragment: [rve_dimtag] + clipped_fibers_dimtags
        # Index 0 is RVE. Indices 1..N are Fibers.

        for i in range(len(clipped_fibers_dimtags)):
            # Map index is 1 + i
            generated_dimtags = out_dimtags_map[1 + i]
            for dt in generated_dimtags:
                fiber_surface_tags.add(dt[1])

        # Collect all tags that come from the RVE input
        rve_related_tags = set()
        for dt in out_dimtags_map[0]:
            rve_related_tags.add(dt[1])

        # Matrix surfaces are those in RVE lineage that are NOT in Fiber lineage
        final_matrix_tags = list(rve_related_tags - fiber_surface_tags)
        final_fiber_tags = list(fiber_surface_tags)

        gmsh.model.occ.synchronize()

        # 6. Apply Periodic Conditions
        # Re-fetch lines as they might have been split
        lines = gmsh.model.getEntities(1)
        boundary_line_map = {b: [] for b in boundaries}

        # Calculate a relative tolerance for boundary matching based on RVE extent
        boundary_dist_tolerance = rve_extent * 1e-8

        for l in lines:
            tag = l[1]
            com = gmsh.model.occ.getCenterOfMass(1, tag)
            for b_idx, b in enumerate(boundaries):
                # Distance to segment
                dist = b.get_distance_to_fiber(np.array(com[:2]))
                if dist < boundary_dist_tolerance:
                    boundary_line_map[b].append(tag)

        for b in boundaries:
            if b.type == BoundaryType.PERIODIC and b.pair is not None:
                if hasattr(b, "index") and b.index < b.pair.index:
                    secondary_tags = boundary_line_map[b]
                    primary_tags = boundary_line_map[b.pair]

                    # Simple translation vector from Primary to Secondary
                    # P + T = S  => T = S - P
                    # Let's take midpoint of boundary
                    mid_s = (b.points[0] + b.points[1]) / 2.0
                    mid_p = (b.pair.points[0] + b.pair.points[1]) / 2.0
                    trans = mid_s - mid_p

                    translation = [
                        1,
                        0,
                        0,
                        trans[0],
                        0,
                        1,
                        0,
                        trans[1],
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]

                    if secondary_tags and primary_tags:
                        matched_secondary, matched_primary = (
                            self._get_periodic_boundary_mapping(
                                secondary_tags, primary_tags, translation
                            )
                        )

                        if matched_secondary:
                            if len(matched_secondary) != len(secondary_tags) or len(
                                matched_primary
                            ) != len(primary_tags):
                                print(
                                    f"Warning: Partial periodic match for boundary {b.index} vs {b.pair.index}. Matched {len(matched_secondary)} segments."
                                )
                                print(
                                    f"  Total Secondary: {len(secondary_tags)}, Total Primary: {len(primary_tags)}"
                                )

                            gmsh.model.mesh.setPeriodic(
                                1, matched_secondary, matched_primary, translation
                            )
                        else:
                            print(
                                f"Warning: No matching periodic segments found for boundary {b.index} vs {b.pair.index}."
                            )

        # 7. Physical Groups and Generation
        p_matrix = gmsh.model.addPhysicalGroup(2, final_matrix_tags)
        gmsh.model.setPhysicalName(2, p_matrix, "Matrix")

        p_fibers = gmsh.model.addPhysicalGroup(2, final_fiber_tags)
        gmsh.model.setPhysicalName(2, p_fibers, "Fibers")

        if mesh_size_factor:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_factor)

        gmsh.model.mesh.generate(2)

        if visualize_gui:
            gmsh.fltk.run()

        gmsh.write(self.mesh_name + ".msh")
        gmsh.write(self.mesh_name + ".vtk")

        if check_periodicity:
            self._check_periodicity(
                boundaries, boundary_line_map, boundary_dist_tolerance
            )

        gmsh.finalize()

    def _is_inside_fiber(self, point_3d, fiber):
        """Checks if a point lies strictly inside a fiber (excluding boundary).

        Parameters
        ----------
        point_3d : Sequence[float]
            The 3D point to check (z is ignored).
        fiber : Fiber
            The fiber to check against.

        Returns
        -------
        bool
            True if the point is inside the fiber radius (with a small buffer).
        """
        dist = np.linalg.norm(np.array(point_3d[:2]) - fiber.center)
        return dist < fiber.radius * (1.0 - 1e-6)

    def _check_periodicity(self, boundaries, boundary_line_map, tolerance):
        """Verifies that nodes on periodic boundaries match up.

        Parameters
        ----------
        boundaries : List[LinearBoundary]
            List of boundaries to check.
        boundary_line_map : dict
            Mapping from boundaries to curve tags.
        tolerance : float
            Distance tolerance for node matching.
        """
        for b in boundaries:
            if b.type == BoundaryType.PERIODIC and b.pair is not None:
                # Ensure we only check once per pair
                if hasattr(b, "index") and b.index < b.pair.index:
                    secondary_tags = boundary_line_map[b]
                    primary_tags = boundary_line_map[b.pair]

                    if not secondary_tags or not primary_tags:
                        continue

                    # Calculate translation
                    mid_s = (b.points[0] + b.points[1]) / 2.0
                    mid_p = (b.pair.points[0] + b.pair.points[1]) / 2.0
                    trans = mid_s - mid_p

                    # Get nodes
                    secondary_node_tags = []
                    secondary_coords = []
                    for t in secondary_tags:
                        node_tags, coords, _ = gmsh.model.mesh.getNodes(
                            1, t, includeBoundary=True
                        )
                        secondary_node_tags.extend(node_tags)
                        for i in range(0, len(coords), 3):
                            secondary_coords.append(coords[i : i + 3])

                    primary_node_tags = []
                    primary_coords = []
                    for t in primary_tags:
                        node_tags, coords, _ = gmsh.model.mesh.getNodes(
                            1, t, includeBoundary=True
                        )
                        primary_node_tags.extend(node_tags)
                        for i in range(0, len(coords), 3):
                            # Apply translation to primary nodes
                            p = np.array(coords[i : i + 3])
                            p[0] += trans[0]
                            p[1] += trans[1]
                            primary_coords.append(p)

                    primary_node_tags = np.array(primary_node_tags)
                    secondary_node_tags = np.array(secondary_node_tags)

                    if not secondary_coords or not primary_coords:
                        continue

                    # Robust comparison using KDTree
                    secondary_coords = np.array(secondary_coords)
                    primary_coords = np.array(primary_coords)

                    tree = KDTree(secondary_coords)
                    # Query distances to nearest neighbors
                    dists, closest_indices = tree.query(primary_coords, k=1)

                    unmatched_indices = np.where(dists > tolerance)[0]

                    if len(unmatched_indices) > 0:
                        max_msg = (
                            f"Max discrepancy: {np.max(dists[unmatched_indices]):.2e}"
                        )

                        unmatched_primary_tags = primary_node_tags[unmatched_indices]
                        unmatched_closest_indices = closest_indices[unmatched_indices]
                        unmatched_secondary_tags = secondary_node_tags[
                            unmatched_closest_indices
                        ]

                        # Show up to 10 unmatched nodes for diagnostic purposes
                        num_show = 10
                        unmatched_details = []
                        for i in range(min(num_show, len(unmatched_indices))):
                            idx = unmatched_indices[i]
                            p_tag = unmatched_primary_tags[i]
                            s_tag = unmatched_secondary_tags[i]
                            dist = dists[idx]
                            unmatched_details.append(
                                f"  Primary Tag {p_tag} -> Closest Secondary Tag {s_tag} (dist: {dist:.2e})"
                            )

                        unmatched_str = "\n".join(unmatched_details)
                        if len(unmatched_indices) > num_show:
                            unmatched_str += (
                                f"\n  ... and {len(unmatched_indices) - num_show} more."
                            )

                        raise RuntimeError(
                            f"Periodic mesh verification failed for boundary {b.index} vs {b.pair.index}.\n"
                            f"Total # of Primary Nodes: {len(primary_coords)}\n"
                            f"Total # of Unmatched Nodes (dist > {tolerance}): {len(unmatched_indices)}\n"
                            f"{max_msg}\n"
                            f"Unmatched Node Details:\n{unmatched_str}"
                        )
                    else:
                        avg_dist = np.mean(dists) if len(dists) > 0 else 0
                        print(
                            f"Periodic check passed (KDTree) for boundary {b.index} <-> {b.pair.index} (avg dist: {avg_dist:.2e})"
                        )

    def _get_periodic_boundary_mapping(
        self,
        secondary_tags: List[int],
        primary_tags: List[int],
        translation: List[float],
    ) -> Tuple[List[int], List[int]]:
        """Finds a 1-to-1 mapping between secondary and primary curve segments based on CoM.

        Parameters
        ----------
        secondary_tags : List[int]
            List of curve tags on the secondary boundary.
        primary_tags : List[int]
            List of curve tags on the primary boundary.
        translation : List[float]
            The 16-element transformation matrix (GMSH format).

        Returns
        -------
        Tuple[List[int], List[int]]
            A tuple of (matched_secondary_tags, matched_primary_tags).
        """
        matched_secondary = []
        matched_primary = []
        remaining_primaries = list(primary_tags)
        trans_vec = np.array(translation[3:12:4])  # Extract [tx, ty, tz]

        # In GMSH: Secondary = Primary + Translation, so Primary = Secondary - Translation
        for s_tag in secondary_tags:
            s_com = np.array(gmsh.model.occ.getCenterOfMass(1, s_tag))
            expected_p_com = s_com - trans_vec

            best_idx = -1
            min_dist = 1e-6  # Tolerance

            for i, p_tag in enumerate(remaining_primaries):
                p_com = np.array(gmsh.model.occ.getCenterOfMass(1, p_tag))
                dist = np.linalg.norm(p_com - expected_p_com)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            if best_idx != -1:
                matched_secondary.append(s_tag)
                matched_primary.append(remaining_primaries[best_idx])
                remaining_primaries.pop(best_idx)

        return matched_secondary, matched_primary
