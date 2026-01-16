import sys
from typing import List, Optional, Tuple
import numpy as np

try:
    import gmsh
except ImportError:
    gmsh = None

from fiber_matrix.models.boundary import LinearBoundary, BoundaryType
from fiber_matrix.models.fiber import Fiber, PeriodicMasterFiber

class GmshMesher:
    """Handles meshing of the RVE using GMSH."""
    
    def __init__(self, mesh_name: str = "FiberMatrixRVE"):
        self.mesh_name = mesh_name
        self._check_gmsh()
        
    def _check_gmsh(self):
        if gmsh is None:
            raise ImportError("GMSH python library is not installed. Please install it using 'pip install gmsh'.")

    def create_mesh(self, 
                    fibers: List[PeriodicMasterFiber], 
                    boundaries: List[LinearBoundary], 
                    mesh_size_factor: float = 1.0,
                    visualize_gui: bool = False):
        
        gmsh.initialize()
        gmsh.model.add(self.mesh_name)
        # gmsh.model.geo.removeAll() # Invalid call
        
        # Use OpenCASCADE kernel for robust boolean operations
        
        # 1. Create RVE Polygon
        # We assume the boundaries form a closed loop.
        boundary_points = []
        for b in boundaries:
            boundary_points.append(b.points[0])
            
        occ_pt_tags = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in boundary_points]
        # Close the loop explicitly for lines
        occ_pt_tags_loop = occ_pt_tags + [occ_pt_tags[0]]
        
        occ_line_tags = []
        for i in range(len(occ_pt_tags)):
            occ_line_tags.append(gmsh.model.occ.addLine(occ_pt_tags_loop[i], occ_pt_tags_loop[i+1]))
            
        rve_wire = gmsh.model.occ.addWire(occ_line_tags)
        rve_face = gmsh.model.occ.addPlaneSurface([rve_wire])
        
        # 2. Add Fibers (Disks)
        fiber_disks = []
        
        def add_fiber_disk(f):
            return gmsh.model.occ.addDisk(f.center[0], f.center[1], 0, f.radius, f.radius)
            
        for f in fibers:
            fiber_disks.append(add_fiber_disk(f))
            for g in f.ghost_fibers:
                fiber_disks.append(add_fiber_disk(g))
                print(g.center)
        
        gmsh.model.occ.synchronize()

        if visualize_gui:
            gmsh.fltk.run()

        # 3. Clip fibers to RVE using Intersect
        # We want the part of fibers INSIDE the RVE.
        # Object=Fibers, Tool=RVE
        
        rve_dimtag = (2, rve_face)
        fiber_dimtags = [(2, t) for t in fiber_disks]
        
        # Intersect(Fibers, RVE). removeObject=True (consume fibers), removeTool=False (keep RVE).
        clipped_fibers_dimtags, _ = gmsh.model.occ.intersect(fiber_dimtags, [rve_dimtag], removeObject=True, removeTool=False)
        
        gmsh.model.occ.synchronize()

        if visualize_gui:
             gmsh.fltk.run()
        
        # 4. Fragment
        # Embed the clipped fibers into the RVE face.
        out_dimtags, out_dimtags_map = gmsh.model.occ.fragment([rve_dimtag], clipped_fibers_dimtags)
        
        gmsh.model.occ.synchronize()
        if visualize_gui:
             gmsh.fltk.run()
        
        # 5. Identity Matrix vs Fibers
        final_fiber_tags = []
        final_matrix_tags = []
        
        surfaces = gmsh.model.getEntities(2)
        
        for s in surfaces:
            tag = s[1]
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            
            # Check if inside RVE (using boundary logic)
            is_inside_rve = True
            for b in boundaries:
                # Assuming boundaries define the "inside" to be on one side (cross product < 0 or > 0 depending on winding)
                # The provided `get_point_relative_position` assumes winding order.
                # Assuming simple convex polygon/rectangle.
                if b.get_point_relative_position(np.array(com[:2])) < -1e-6:
                     is_inside_rve = False
                     break
            
            if not is_inside_rve:
                gmsh.model.occ.remove([(2, tag)], recursive=True)
                continue
                
            is_fiber = False
            for f in fibers:
                 if self._is_inside_fiber(com, f) or any(self._is_inside_fiber(com, g) for g in f.ghost_fibers):
                     is_fiber = True
                     break
            
            if is_fiber:
                final_fiber_tags.append(tag)
            else:
                final_matrix_tags.append(tag)
                
        gmsh.model.occ.synchronize()
        
        # 6. Apply Periodic Conditions
        # Re-fetch lines as they might have been split
        lines = gmsh.model.getEntities(1)
        boundary_line_map = {b: [] for b in boundaries}
        
        for l in lines:
            tag = l[1]
            com = gmsh.model.occ.getCenterOfMass(1, tag)
            for b_idx, b in enumerate(boundaries):
                # Distance to segment
                dist = b.get_distance_to_fiber(np.array(com[:2])) 
                if dist < 1e-5:
                     boundary_line_map[b].append(tag)
        
        for b in boundaries:
            if b.type == BoundaryType.PERIODIC and b.pair is not None:
                 if hasattr(b, 'index') and b.index < b.pair.index:
                     slave_tags = boundary_line_map[b]
                     master_tags = boundary_line_map[b.pair]
                     
                     # Simple translation vector from Master to Slave
                     # M + T = S  => T = S - M
                     # Let's take midpoint of boundary
                     mid_s = (b.points[0] + b.points[1]) / 2.0
                     mid_m = (b.pair.points[0] + b.pair.points[1]) / 2.0
                     trans = mid_s - mid_m
                     
                     translation = [1, 0, 0, trans[0], 0, 1, 0, trans[1], 0, 0, 1, 0, 0, 0, 0, 1]
                     
                     if slave_tags and master_tags:
                         if len(slave_tags) == len(master_tags):
                            gmsh.model.mesh.setPeriodic(1, slave_tags, master_tags, translation)
                         else:
                            print(f"Warning: Mismatch in periodic boundary segments for boundary {b.index} vs {b.pair.index} ({len(slave_tags)} vs {len(master_tags)}). Skipping periodic constraint.")

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
        
        gmsh.finalize()

    def _is_inside_fiber(self, point_3d, fiber):
        dist = np.linalg.norm(np.array(point_3d[:2]) - fiber.center)
        return dist < fiber.radius * (1.0 - 1e-6)
