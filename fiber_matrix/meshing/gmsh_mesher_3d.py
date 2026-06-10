from typing import List, Tuple

import numpy as np
from scipy.spatial import KDTree

try:
    import gmsh
except ImportError:
    gmsh = None

from fiber_matrix.models.boundary import BoundaryType, LinearBoundary
from fiber_matrix.models.fiber import Fiber, PeriodicPrimaryFiber


class GmshMesher3D:
    """Handles 3D volume meshing of an RVE using GMSH.

    The 3D mesh is built by creating the same fragmented 2D fiber/matrix
    geometry as :class:`GmshMesher`, then extruding those surfaces through a
    user-provided thickness. Matching exterior surfaces are paired and can be
    checked for periodic node correspondence.
    """

    def __init__(self, mesh_name: str = "FiberMatrixRVE3D"):
        self.mesh_name = mesh_name
        self._check_gmsh()

    def _check_gmsh(self):
        if gmsh is None:
            raise ImportError(
                "GMSH python library is not installed. Please install it using 'pip install gmsh'."
            )

    def create_mesh(
        self,
        fibers: List[PeriodicPrimaryFiber],
        boundaries: List[LinearBoundary],
        thickness: float,
        mesh_size_factor: float = 1.0,
        z_layers: int = 4,
        visualize_gui: bool = False,
        check_periodicity: bool = False,
        periodic_z: bool = False,
        surface_groups: bool = False,
        composite_surface_groups: bool = False,
        anchor_node_groups: bool = False,
        uniform_mesh: bool = True,
        fiber_mesh_size: float = None,
        matrix_mesh_size: float = None,
        boundary_mesh_size: float = None,
        interface_refinement_distance: float = None,
        boundary_refinement_distance: float = None,
        recombine_prisms: bool = False,
    ):
        """Creates a 3D fiber/matrix volume mesh using GMSH.

        Parameters
        ----------
        fibers : List[PeriodicPrimaryFiber]
            List of fibers to include in the mesh.
        boundaries : List[LinearBoundary]
            List of boundaries defining the 2D RVE cross-section.
        thickness : float
            Extrusion thickness in the positive z direction.
        mesh_size_factor : float, optional
            Factor to control mesh refinement. Default is 1.0.
        z_layers : int, optional
            Number of mesh layers through the thickness. Default is 4.
        visualize_gui : bool, optional
            If True, launches GMSH GUI to visualize geometry/mesh. Default is False.
        check_periodicity : bool, optional
            If True, verifies generated nodes on periodic surfaces. Default is False.
        periodic_z : bool, optional
            If True, applies periodic constraints between bottom and top surfaces.
            Default is False.
        surface_groups : bool, optional
            If True, creates material-specific physical surface groups for
            left, right, bottom, top, front, and back. Default is False.
        composite_surface_groups : bool, optional
            If True, creates whole-composite physical surface groups for
            left, right, bottom, top, front, and back. Default is False.
        anchor_node_groups : bool, optional
            If True, creates 0D physical groups named ``anchor_xyz``,
            ``anchor_yz``, and ``anchor_z`` for mechanical constraint boundary
            conditions. Default is False.
        uniform_mesh : bool, optional
            If True, uses ``mesh_size_factor`` as a global uniform size.
            If False, applies separate mesh sizes for fiber, matrix, and
            exterior boundary regions. Default is True.
        fiber_mesh_size : float, optional
            Target element size on fiber surfaces when ``uniform_mesh`` is False.
            Defaults to ``mesh_size_factor``.
        matrix_mesh_size : float, optional
            Target element size on matrix surfaces when ``uniform_mesh`` is False.
            Defaults to ``mesh_size_factor``.
        boundary_mesh_size : float, optional
            Target element size on exterior domain boundaries when
            ``uniform_mesh`` is False. Defaults to the smaller of fiber and
            matrix mesh sizes.
        interface_refinement_distance : float, optional
            Distance away from fiber/matrix interfaces over which the mesh
            transitions from ``fiber_mesh_size`` to ``matrix_mesh_size`` when
            ``uniform_mesh`` is False. Defaults to three times
            ``fiber_mesh_size``.
        boundary_refinement_distance : float, optional
            Distance away from exterior domain boundaries over which the mesh
            transitions from ``boundary_mesh_size`` to ``matrix_mesh_size``
            when ``uniform_mesh`` is False. Defaults to three times
            ``boundary_mesh_size``.
        recombine_prisms : bool, optional
            If True, recombines the structured extrusion into prism/wedge
            elements instead of subdividing into tetrahedra. This can remove
            radial-looking tetrahedral subdivision patterns. Default is False.
        """
        if thickness <= 0:
            raise ValueError("thickness must be positive.")
        if z_layers < 1:
            raise ValueError("z_layers must be at least 1.")

        gmsh.initialize()
        gmsh.model.add(self.mesh_name)

        all_b_points = []
        for b in boundaries:
            all_b_points.append(b.points[0])
            all_b_points.append(b.points[1])
        all_b_points = np.array(all_b_points)

        rve_extent = np.linalg.norm(
            np.max(all_b_points, axis=0) - np.min(all_b_points, axis=0)
        )
        scale_factor = 1.0 / rve_extent

        ordered_chain = self._order_boundary_chain(boundaries, scale_factor)

        occ_line_tags = []
        first_pt_tag = gmsh.model.occ.addPoint(
            ordered_chain[0][1][0], ordered_chain[0][1][1], 0
        )
        prev_pt_tag = first_pt_tag

        for i, item in enumerate(ordered_chain):
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

        fiber_disks = []
        for f in fibers:
            fiber_disks.append(self._add_fiber_disk(f, scale_factor))
            for g in f.ghost_fibers:
                fiber_disks.append(self._add_fiber_disk(g, scale_factor))

        gmsh.model.occ.synchronize()
        if visualize_gui:
            gmsh.fltk.run()

        rve_dimtag = (2, rve_face)
        fiber_dimtags = [(2, tag) for tag in fiber_disks]
        clipped_fibers_dimtags, _ = gmsh.model.occ.intersect(
            fiber_dimtags, [rve_dimtag], removeObject=True, removeTool=False
        )
        gmsh.model.occ.synchronize()

        refinement_dimtags = []
        if not uniform_mesh:
            refinement_distance = self._default_refinement_distance(
                mesh_size_factor,
                fiber_mesh_size,
                interface_refinement_distance,
            )
            boundary_distance = self._default_refinement_distance(
                mesh_size_factor,
                boundary_mesh_size,
                boundary_refinement_distance,
            )
            refinement_dimtags = self._add_refinement_partition_disks(
                fibers,
                rve_dimtag,
                scale_factor,
                refinement_distance,
                rve_extent,
            )
            refinement_dimtags += self._add_boundary_refinement_partition(
                all_b_points,
                scale_factor,
                boundary_distance,
            )

        out_dimtags, out_dimtags_map = gmsh.model.occ.fragment(
            [rve_dimtag], clipped_fibers_dimtags + refinement_dimtags
        )
        if not out_dimtags_map:
            raise RuntimeError("GMSH Fragment operation returned an empty map.")

        fiber_surface_tags = set()
        for i in range(len(clipped_fibers_dimtags)):
            for dt in out_dimtags_map[1 + i]:
                if dt[0] == 2:
                    fiber_surface_tags.add(dt[1])

        rve_related_tags = {
            dt[1] for dt in out_dimtags_map[0] if dt[0] == 2
        }
        final_matrix_tags = list(rve_related_tags - fiber_surface_tags)
        final_fiber_tags = list(fiber_surface_tags)

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

        all_base_surfaces = [(2, tag) for tag in final_matrix_tags + final_fiber_tags]

        gmsh.model.occ.extrude(
            all_base_surfaces,
            0,
            0,
            thickness,
            numElements=[z_layers],
            recombine=recombine_prisms,
        )
        gmsh.model.occ.synchronize()

        all_fiber_copies = self._get_all_fiber_copies(fibers)
        matrix_volume_tags, fiber_volume_tags = self._classify_volumes(
            final_matrix_tags, final_fiber_tags, all_fiber_copies
        )

        side_surface_map, bottom_surface_tags, top_surface_tags = (
            self._collect_periodic_surfaces(boundaries, rve_extent, thickness)
        )

        periodic_surface_pairs = []
        for b in boundaries:
            if b.type == BoundaryType.PERIODIC and b.pair is not None:
                if hasattr(b, "index") and b.index < b.pair.index:
                    secondary_tags = side_surface_map[b]
                    primary_tags = side_surface_map[b.pair]
                    trans = self._boundary_translation(b, b.pair)
                    translation = self._translation_matrix(trans[0], trans[1], 0.0)
                    matched_secondary, matched_primary = self._get_periodic_surface_mapping(
                        secondary_tags, primary_tags, translation, rve_extent
                    )

                    if matched_secondary:
                        periodic_surface_pairs.append(
                            (matched_secondary, matched_primary, translation)
                        )
                    else:
                        print(
                            f"Warning: No matching periodic surfaces found for boundary {b.index} vs {b.pair.index}."
                        )

        if periodic_z:
            translation = self._translation_matrix(0.0, 0.0, thickness)
            matched_top, matched_bottom = self._get_periodic_surface_mapping(
                top_surface_tags, bottom_surface_tags, translation, rve_extent
            )
            if matched_top:
                periodic_surface_pairs.append((matched_top, matched_bottom, translation))
            else:
                print("Warning: No matching periodic surfaces found for top vs bottom.")

        if matrix_volume_tags:
            p_matrix = gmsh.model.addPhysicalGroup(3, matrix_volume_tags)
            gmsh.model.setPhysicalName(3, p_matrix, "Matrix")
        if fiber_volume_tags:
            p_fibers = gmsh.model.addPhysicalGroup(3, fiber_volume_tags)
            gmsh.model.setPhysicalName(3, p_fibers, "Fibers")
        if surface_groups:
            self._add_surface_physical_groups(
                all_b_points,
                thickness,
                matrix_volume_tags,
                fiber_volume_tags,
                rve_extent,
            )
        if composite_surface_groups:
            self._add_composite_surface_physical_groups(
                all_b_points,
                thickness,
                rve_extent,
            )
        if anchor_node_groups:
            self._add_anchor_node_physical_groups(all_b_points, thickness, rve_extent)

        if uniform_mesh and mesh_size_factor:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_factor)
        elif not uniform_mesh:
            self._apply_material_mesh_sizes(
                all_b_points,
                thickness,
                matrix_volume_tags,
                fiber_volume_tags,
                rve_extent,
                mesh_size_factor,
                fiber_mesh_size,
                matrix_mesh_size,
                boundary_mesh_size,
                interface_refinement_distance,
                boundary_refinement_distance,
                periodic_surface_pairs,
            )

        gmsh.model.mesh.generate(3)

        if visualize_gui:
            gmsh.fltk.run()

        periodicity_message = None
        if check_periodicity:
            self._check_periodicity(periodic_surface_pairs, rve_extent)
            periodicity_message = self._check_periodicity(periodic_surface_pairs, rve_extent)

        self._set_periodic_surface_constraints(periodic_surface_pairs)

        gmsh.write(self.mesh_name + ".msh")
        gmsh.write(self.mesh_name + ".vtk")

        gmsh.finalize()
        
        print(periodicity_message)

    def _order_boundary_chain(self, boundaries, scale_factor):
        ordered_chain = []
        remaining_boundaries = list(boundaries)
        if not remaining_boundaries:
            raise ValueError("No boundaries provided.")

        current_b = remaining_boundaries.pop(0)
        current_start = current_b.points[0] * scale_factor
        current_end = current_b.points[1] * scale_factor
        ordered_chain.append((current_b, current_start, current_end))

        while remaining_boundaries:
            found_idx = -1
            found_orientation = 0
            for i, b in enumerate(remaining_boundaries):
                p0 = b.points[0] * scale_factor
                p1 = b.points[1] * scale_factor
                if np.linalg.norm(p0 - current_end) < 1e-4:
                    found_idx = i
                    found_orientation = 0
                    break
                if np.linalg.norm(p1 - current_end) < 1e-4:
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

        return ordered_chain

    def _add_fiber_disk(self, fiber: Fiber, scale_factor: float) -> int:
        return gmsh.model.occ.addDisk(
            fiber.center[0] * scale_factor,
            fiber.center[1] * scale_factor,
            0,
            fiber.radius * scale_factor,
            fiber.radius * scale_factor,
        )

    def _default_refinement_distance(
        self,
        mesh_size_factor,
        fiber_mesh_size,
        interface_refinement_distance,
    ):
        if interface_refinement_distance is not None:
            return interface_refinement_distance
        base_size = mesh_size_factor if mesh_size_factor is not None else 1.0
        fiber_size = fiber_mesh_size if fiber_mesh_size is not None else base_size
        return 3.0 * fiber_size

    def _add_refinement_partition_disks(
        self,
        fibers,
        rve_dimtag,
        scale_factor,
        refinement_distance,
        rve_extent,
    ):
        if refinement_distance <= 0:
            raise ValueError("interface_refinement_distance must be positive.")

        partition_disks = []
        min_radius = max(rve_extent * 1e-8, 1e-9)
        for fiber in self._get_all_fiber_copies(fibers):
            outer_radius = fiber.radius + refinement_distance
            partition_disks.append(
                gmsh.model.occ.addDisk(
                    fiber.center[0] * scale_factor,
                    fiber.center[1] * scale_factor,
                    0,
                    outer_radius * scale_factor,
                    outer_radius * scale_factor,
                )
            )

            inner_radius = fiber.radius - refinement_distance
            if inner_radius > min_radius:
                partition_disks.append(
                    gmsh.model.occ.addDisk(
                        fiber.center[0] * scale_factor,
                        fiber.center[1] * scale_factor,
                        0,
                        inner_radius * scale_factor,
                        inner_radius * scale_factor,
                    )
                )

        if not partition_disks:
            return []

        gmsh.model.occ.synchronize()
        clipped_partitions, _ = gmsh.model.occ.intersect(
            [(2, tag) for tag in partition_disks],
            [rve_dimtag],
            removeObject=True,
            removeTool=False,
        )
        gmsh.model.occ.synchronize()
        return [dt for dt in clipped_partitions if dt[0] == 2]

    def _add_boundary_refinement_partition(
        self,
        boundary_points,
        scale_factor,
        boundary_refinement_distance,
    ):
        if boundary_refinement_distance <= 0:
            raise ValueError("boundary_refinement_distance must be positive.")

        coords_min = np.min(boundary_points, axis=0)
        coords_max = np.max(boundary_points, axis=0)
        inset_min = coords_min + boundary_refinement_distance
        inset_max = coords_max - boundary_refinement_distance

        if np.any(inset_min >= inset_max):
            return []

        corners = [
            (inset_min[0], inset_min[1]),
            (inset_max[0], inset_min[1]),
            (inset_max[0], inset_max[1]),
            (inset_min[0], inset_max[1]),
        ]
        point_tags = [
            gmsh.model.occ.addPoint(x * scale_factor, y * scale_factor, 0)
            for x, y in corners
        ]
        line_tags = [
            gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % 4])
            for i in range(4)
        ]
        wire = gmsh.model.occ.addWire(line_tags)
        face = gmsh.model.occ.addPlaneSurface([wire])
        gmsh.model.occ.synchronize()
        return [(2, face)]

    def _get_all_fiber_copies(self, fibers):
        all_fibers = []
        for f in fibers:
            all_fibers.append(f)
            all_fibers.extend(f.ghost_fibers)
        return all_fibers

    def _classify_volumes(self, matrix_surface_tags, fiber_surface_tags, fiber_copies):
        matrix_surface_tags = set(matrix_surface_tags)
        fiber_surface_tags = set(fiber_surface_tags)
        matrix_volume_tags = []
        fiber_volume_tags = []
        for _, tag in gmsh.model.getEntities(3):
            boundary_surfaces = {
                surface_tag
                for dim, surface_tag in gmsh.model.getBoundary(
                    [(3, tag)], oriented=False, recursive=False
                )
                if dim == 2
            }
            if boundary_surfaces & fiber_surface_tags:
                fiber_volume_tags.append(tag)
                continue
            if boundary_surfaces & matrix_surface_tags:
                matrix_volume_tags.append(tag)
                continue

            com = np.array(gmsh.model.occ.getCenterOfMass(3, tag))
            if any(self._is_inside_fiber(com, f) for f in fiber_copies):
                fiber_volume_tags.append(tag)
            else:
                matrix_volume_tags.append(tag)
        return matrix_volume_tags, fiber_volume_tags

    def _collect_periodic_surfaces(self, boundaries, rve_extent, thickness):
        tolerance = rve_extent * 1e-8
        side_surface_map = {b: [] for b in boundaries}
        bottom_surface_tags = []
        top_surface_tags = []

        for _, tag in gmsh.model.getEntities(2):
            com = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
            if abs(com[2]) < tolerance:
                bottom_surface_tags.append(tag)
                continue
            if abs(com[2] - thickness) < tolerance:
                top_surface_tags.append(tag)
                continue

            for b in boundaries:
                dist = b.get_distance_to_fiber(np.array(com[:2]))
                if dist < tolerance:
                    side_surface_map[b].append(tag)

        return side_surface_map, bottom_surface_tags, top_surface_tags

    def _add_surface_physical_groups(
        self,
        boundary_points,
        thickness,
        matrix_volume_tags,
        fiber_volume_tags,
        rve_extent,
    ):
        coords_min = np.min(boundary_points, axis=0)
        coords_max = np.max(boundary_points, axis=0)
        tolerance = max(rve_extent * 1e-8, 1e-9)
        matrix_volume_tags = set(matrix_volume_tags)
        fiber_volume_tags = set(fiber_volume_tags)
        grouped_surfaces = {
            ("Matrix", "Left"): [],
            ("Matrix", "Right"): [],
            ("Matrix", "Bottom"): [],
            ("Matrix", "Top"): [],
            ("Matrix", "Front"): [],
            ("Matrix", "Back"): [],
            ("Fibers", "Left"): [],
            ("Fibers", "Right"): [],
            ("Fibers", "Bottom"): [],
            ("Fibers", "Top"): [],
            ("Fibers", "Front"): [],
            ("Fibers", "Back"): [],
        }

        for _, tag in gmsh.model.getEntities(2):
            com = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
            sides = self._surface_sides(com, coords_min, coords_max, thickness, tolerance)
            if not sides:
                continue

            material = self._surface_material(tag, matrix_volume_tags, fiber_volume_tags)
            if material is None:
                continue

            for side in sides:
                grouped_surfaces[(material, side)].append(tag)

        for (material, side), surface_tags in grouped_surfaces.items():
            if not surface_tags:
                continue
            group = gmsh.model.addPhysicalGroup(2, surface_tags)
            gmsh.model.setPhysicalName(2, group, f"{material}_{side}")

    def _add_composite_surface_physical_groups(
        self,
        boundary_points,
        thickness,
        rve_extent,
    ):
        coords_min = np.min(boundary_points, axis=0)
        coords_max = np.max(boundary_points, axis=0)
        tolerance = max(rve_extent * 1e-8, 1e-9)
        grouped_surfaces = {
            "left": [],
            "right": [],
            "bottom": [],
            "top": [],
            "front": [],
            "back": [],
        }

        for _, tag in gmsh.model.getEntities(2):
            com = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
            sides = self._surface_sides(com, coords_min, coords_max, thickness, tolerance)
            for side in sides:
                grouped_surfaces[side.lower()].append(tag)

        for side, surface_tags in grouped_surfaces.items():
            if not surface_tags:
                continue
            group = gmsh.model.addPhysicalGroup(2, surface_tags)
            gmsh.model.setPhysicalName(2, group, f"composite_{side}")

    def _add_anchor_node_physical_groups(self, boundary_points, thickness, rve_extent):
        coords_min = np.min(boundary_points, axis=0)
        coords_max = np.max(boundary_points, axis=0)
        tolerance = max(rve_extent * 1e-6, 1e-9)
        anchors = {
            "anchor_xyz": np.array([coords_min[0], coords_min[1], 0.0]),
            "anchor_yz": np.array([coords_max[0], coords_min[1], 0.0]),
            "anchor_z": np.array([coords_min[0], coords_max[1], 0.0]),
        }

        for name, target in anchors.items():
            point_tag = self._find_nearest_point_entity(target, tolerance)
            group = gmsh.model.addPhysicalGroup(0, [point_tag])
            gmsh.model.setPhysicalName(0, group, name)

    def _find_nearest_point_entity(self, target, tolerance):
        best_tag = None
        best_dist = np.inf
        for _, tag in gmsh.model.getEntities(0):
            coords = np.array(gmsh.model.getValue(0, tag, []))
            dist = np.linalg.norm(coords - target)
            if dist < best_dist:
                best_dist = dist
                best_tag = tag

        if best_tag is None or best_dist > tolerance:
            raise RuntimeError(
                "Could not find geometric point for anchor node group near "
                f"{target.tolist()}."
            )
        return best_tag

    def _surface_sides(self, com, coords_min, coords_max, thickness, tolerance):
        sides = []
        if abs(com[0] - coords_min[0]) < tolerance:
            sides.append("Left")
        if abs(com[0] - coords_max[0]) < tolerance:
            sides.append("Right")
        if abs(com[2]) < tolerance:
            sides.append("Bottom")
        if abs(com[2] - thickness) < tolerance:
            sides.append("Top")
        if abs(com[1] - coords_min[1]) < tolerance:
            sides.append("Front")
        if abs(com[1] - coords_max[1]) < tolerance:
            sides.append("Back")
        return sides

    def _surface_material(self, surface_tag, matrix_volume_tags, fiber_volume_tags):
        adjacent_volumes, _ = gmsh.model.getAdjacencies(2, surface_tag)
        adjacent_volumes = set(int(tag) for tag in adjacent_volumes)
        if adjacent_volumes & fiber_volume_tags:
            return "Fibers"
        if adjacent_volumes & matrix_volume_tags:
            return "Matrix"
        return None

    def _apply_material_mesh_sizes(
        self,
        boundary_points,
        thickness,
        matrix_volume_tags,
        fiber_volume_tags,
        rve_extent,
        mesh_size_factor,
        fiber_mesh_size,
        matrix_mesh_size,
        boundary_mesh_size,
        interface_refinement_distance,
        boundary_refinement_distance,
        periodic_surface_pairs,
    ):
        base_size = mesh_size_factor if mesh_size_factor is not None else 1.0
        fiber_size = fiber_mesh_size if fiber_mesh_size is not None else base_size
        matrix_size = matrix_mesh_size if matrix_mesh_size is not None else base_size
        boundary_size = (
            boundary_mesh_size
            if boundary_mesh_size is not None
            else min(fiber_size, matrix_size)
        )
        for name, size in [
            ("fiber_mesh_size", fiber_size),
            ("matrix_mesh_size", matrix_size),
            ("boundary_mesh_size", boundary_size),
        ]:
            if size <= 0:
                raise ValueError(f"{name} must be positive.")

        coords_min = np.min(boundary_points, axis=0)
        coords_max = np.max(boundary_points, axis=0)
        tolerance = max(rve_extent * 1e-8, 1e-9)
        matrix_volume_tags = set(matrix_volume_tags)
        fiber_volume_tags = set(fiber_volume_tags)

        interface_points = set()
        exterior_points = set()

        for _, tag in gmsh.model.getEntities(2):
            points = self._get_surface_boundary_points(tag)
            adjacent_volumes, _ = gmsh.model.getAdjacencies(2, tag)
            adjacent_volumes = set(int(volume_tag) for volume_tag in adjacent_volumes)

            has_matrix = bool(adjacent_volumes & matrix_volume_tags)
            has_fiber = bool(adjacent_volumes & fiber_volume_tags)
            if has_matrix and has_fiber:
                interface_points.update(points)

            com = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
            if self._surface_sides(com, coords_min, coords_max, thickness, tolerance):
                exterior_points.update(points)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(fiber_size, matrix_size, boundary_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", matrix_size)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), matrix_size)
        if interface_points:
            gmsh.model.mesh.setSize([(0, tag) for tag in interface_points], fiber_size)
        if exterior_points:
            gmsh.model.mesh.setSize([(0, tag) for tag in exterior_points], boundary_size)

    def _set_periodic_surface_constraints(self, periodic_surface_pairs):
        for secondary_tags, primary_tags, translation in periodic_surface_pairs:
            try:
                gmsh.model.mesh.setPeriodic(2, secondary_tags, primary_tags, translation)
            except Exception as err:
                print(f"Warning: Could not set GMSH periodic metadata: {err}")

    def _get_surface_boundary_points(self, surface_tag):
        return {
            point_tag
            for dim, point_tag in gmsh.model.getBoundary(
                [(2, surface_tag)], oriented=False, recursive=True
            )
            if dim == 0
        }

    def _boundary_translation(self, secondary_boundary, primary_boundary):
        mid_s = (secondary_boundary.points[0] + secondary_boundary.points[1]) / 2.0
        mid_p = (primary_boundary.points[0] + primary_boundary.points[1]) / 2.0
        return mid_s - mid_p

    def _translation_matrix(self, tx, ty, tz):
        return [
            1,
            0,
            0,
            tx,
            0,
            1,
            0,
            ty,
            0,
            0,
            1,
            tz,
            0,
            0,
            0,
            1,
        ]

    def _is_inside_fiber(self, point_3d, fiber):
        dist = np.linalg.norm(np.array(point_3d[:2]) - fiber.center)
        return dist < fiber.radius * (1.0 - 1e-6)

    def _get_periodic_surface_mapping(
        self,
        secondary_tags: List[int],
        primary_tags: List[int],
        translation: List[float],
        rve_extent: float,
    ) -> Tuple[List[int], List[int]]:
        matched_secondary = []
        matched_primary = []
        remaining_primaries = list(primary_tags)
        trans_vec = np.array(translation[3:12:4])
        tolerance = max(rve_extent * 1e-6, 1e-9)

        for s_tag in secondary_tags:
            s_com = np.array(gmsh.model.occ.getCenterOfMass(2, s_tag))
            expected_p_com = s_com - trans_vec

            best_idx = -1
            min_dist = tolerance
            for i, p_tag in enumerate(remaining_primaries):
                p_com = np.array(gmsh.model.occ.getCenterOfMass(2, p_tag))
                dist = np.linalg.norm(p_com - expected_p_com)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            if best_idx != -1:
                matched_secondary.append(s_tag)
                matched_primary.append(remaining_primaries[best_idx])
                remaining_primaries.pop(best_idx)

        return matched_secondary, matched_primary

    def _check_periodicity(self, periodic_surface_pairs, rve_extent):
        tolerance = max(rve_extent * 1e-6, 1e-9)
        message = None
        for secondary_tags, primary_tags, translation in periodic_surface_pairs:
            secondary_coords = self._get_unique_surface_nodes(secondary_tags)
            primary_coords = self._get_unique_surface_nodes(primary_tags)
            if len(secondary_coords) == 0 or len(primary_coords) == 0:
                continue

            trans_vec = np.array(translation[3:12:4])
            translated_primary = primary_coords + trans_vec
            tree = KDTree(secondary_coords)
            dists, _ = tree.query(translated_primary, k=1)
            unmatched = np.where(dists > tolerance)[0]
            if len(unmatched) > 0:
                raise RuntimeError(
                    "Periodic 3D mesh verification failed. "
                    f"Unmatched nodes: {len(unmatched)}; "
                    f"max discrepancy: {np.max(dists[unmatched]):.2e}"
                )

            avg_dist = np.mean(dists) if len(dists) > 0 else 0.0
            message = f"3D periodic check passed (avg dist: {avg_dist:.2e})"
            print(message)
        return message

    def _get_unique_surface_nodes(self, surface_tags):
        nodes = {}
        for tag in surface_tags:
            node_tags, coords, _ = gmsh.model.mesh.getNodes(
                2, tag, includeBoundary=True
            )
            for i, node_tag in enumerate(node_tags):
                nodes[int(node_tag)] = coords[3 * i : 3 * i + 3]
        return np.array(list(nodes.values()))
