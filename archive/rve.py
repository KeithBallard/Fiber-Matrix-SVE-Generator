"""
TODO
"""

import copy
from enum import Enum
import math
import os
import random
import shutil
import time
from typing import *
from scipy import spatial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from triangle import triangulate, plot as tplot

#from ..mesh import BetaMesh2 as BM

class FiberRVE(object):
    """A class that defines a fiber/matrix RVE, which is composed of a 
        description of the boundaries of the RVE and the fibers within it.
        This class is responsible for placing fibers within the RVE, which
        could be at fixed locations or randomly without overlapping fibers.
        
        Attributes:
            fibers: a list of fibers within the RVE
            boundaries: a list of boundaries that define the domain
    """

    fibers: List['Fiber']
    fiber_vf: float
    boundaries: List['LinearBoundary']
    mesh_points: List[List[float]]
    mesh_segments: List[List[int]]
    mesh_segment_boundary_markers: List[int]
    mesh_fiber_centers: List[List[float]]
    mesh_fiber_radii: List[float]
    mesh_fiber_internal_coords: List[List[float]]

    def __init__(self):
        # For RVE description
        self.fibers = []
        self.fiber_vf = 0.0
        self.boundaries = []
        # For meshing
        self.mesh_points = []
        self.mesh_segments = []
        self.mesh_segment_boundary_markers = []
        self.mesh_fiber_centers = []
        self.mesh_fiber_radii = []
        self.mesh_fiber_internal_coords = []
    
    # ---------------------  #
    # Initialization methods #
    # ---------------------  #        

    def initialize_rectangle_rve(self,
        num_fibers,
        vf=0.6,
        avg_diam=5e-6,
        diam_std_dev=0.0,
        rve_aspect_ratio=1.0,
        fixed_height=None):
        """Calculate the dimensions of the RVE.  Also creates a distribution
        of radii for the fibers.
        """
        self.radii = [a/2 for a in diam_std_dev*avg_diam * np.random.randn(num_fibers) + avg_diam]
        total_fiber_area = sum([math.pi*a*a for a in self.radii])
        rve_area = total_fiber_area/vf
        if fixed_height == None: h = math.sqrt(rve_area/rve_aspect_ratio)   
        else: h = fixed_height
        w = rve_area/h
        rve_dims = [w,h]

        boundary_points = [(0., 0.), (w, 0.), (w, h), (0., h)]
        boundary_types = [self.BoundaryType.PERIODIC for p in boundary_points]
        self._create_boundaries_from_points(boundary_points, boundary_types)

        return rve_dims

    def initialize_polygon_rve(self,
        num_fibers,
        boundary_points,
        avg_diam=5e-6,
        diam_std_dev=0.0):
        """Calculate the dimensions of the RVE.  Also creates a distribution
        of radii for the fibers.
        """
        self.radii = [a/2 for a in diam_std_dev*avg_diam * np.random.randn(num_fibers) + avg_diam]

        boundary_types = [self.BoundaryType.FINITE for p in boundary_points]
        self._create_boundaries_from_points(boundary_points, boundary_types)

    def initialize_polygon_rve(self,
        fiber_radii,
        boundary_points,
        boundary_types):
        """Instead of letting this class decide on th radii of fibers, the caller provides the
        radii and boundary points.
        """
        self.radii = fiber_radii
        self._create_boundaries_from_points(boundary_points, boundary_types)

    # -------------------- #
    # Boundary methods #
    # -------------------- #
        
    def _create_boundaries_from_points(self, boundary_points, boundary_types):
        """boundary_points: list of coordinates  that define a closed spline
                (the last coordinate will automatically be connected back to
                the first coordinate to form the closed spline)
            boundary_types: list of BoundaryType enums that indicate
                what type each boundary is, should be of size
                len(boundary_points)
        """
        self.boundary_points = boundary_points
        self.boundaries = [self.LinearBoundary(points = [boundary_points[i], boundary_points[i+1]], point_indices = [i, i+1], btype=boundary_types[i]) for i in range(len(boundary_points)-1)]
        self.boundaries.append( self.LinearBoundary(points = [boundary_points[-1], boundary_points[0]], point_indices = [len(boundary_points)-1, 0], btype=boundary_types[-1]) )
        self._assign_boundary_pairs()
        self.fibers = []
        for i in range(len(self.boundaries)): self.boundaries[i].index = i
        
    def _assign_boundary_pairs(self):
        """If a boundary lies directly across from another boundary in such
            a way that it could be periodic, then it is considered a boundary
            pair.  This method will check for any that exist and assign the
            pairs, which is stored as an attribute within the boundary objects.
            
            Currently this is an n^2 algorithm, but there should not be too many
            boundaries for the RVE considered right now.  Profile later to see
            if improvement is needed.
        """
        for boundary in self.boundaries:
            for boundary_to_check in self.boundaries:
                if boundary.is_pair(boundary_to_check):
                    boundary.pair = boundary_to_check
                    boundary_to_check.pair = boundary
                    if boundary.type == self.BoundaryType.PERIODIC or boundary.pair.type == self.BoundaryType.PERIODIC:
                        if boundary.type != boundary.pair.type:
                            raise ValueError('One boundary of a pair was periodic but the other was not.')
    
    # ----------------------------------- #
    # Methods to find fiber configuration #
    # ----------------------------------- #

    def place_initial_fibers(self,
        specified_fiber_centers=[],
        plot_triangulization=False):
        """TODO
        Returns tuple (Vf, RVE_area)
        """

        # Triangulate the polygon
        points = np.array(self.boundary_points)
        x = points[:,0]
        y = points[:,1]
        #cens,edg,tri,neig = triang.delaunay(x,y)
        tri_in = {'vertices': points}
        tri_in['segments'] = [[i, i+1] for i in range(len(points))]
        tri_in['segments'].append([len(points)-1, 0])
        self.triangulation = triangulate(tri_in, 'p')

        if plot_triangulization:
            for t in self.triangulation['triangles']:
                # t[0], t[1], t[2] are the points indexes of the triangle
                t_i = [t[0], t[1], t[2], t[0]]
                plt.plot(x[t_i],y[t_i])
            plt.show()

        def area(a, b, c) :
            return 0.5 * np.linalg.norm( np.cross( b-a, c-a ) )
    
        areas = []
        centroid = [0.0, 0.0]
        for t in self.triangulation['triangles']:
            a = np.array([x[t[0]], y[t[0]]])
            b = np.array([x[t[1]], y[t[1]]])
            c = np.array([x[t[2]], y[t[2]]])
            tri_area = area(a, b, c)
            tri_centroid = np.array([(a[i] + b[i] + c[i]) / 3.0 for i in range(2)])
            for i in range(2):
                centroid[i] += tri_area * (a[i] + b[i] + c[i]) / 3.0
            areas.append(tri_area)
        rve_area = sum(areas)
        centroid /= rve_area

        def weighted_choice(choice_weight_pairs):
            # Source: https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
            total = sum(w for c, w in choice_weight_pairs)
            r = random.uniform(0, total)
            upto = 0
            for choice, weight in choice_weight_pairs:
                if upto + weight >= r:
                    return choice
                upto += weight
            assert False, "Shouldn't get here"

        self.fibers = []
        num_fibers = len(self.radii)

        if len(specified_fiber_centers) == 0 and num_fibers == 1:
            specified_fiber_centers.append(list(centroid))
            
        # This is the case where random fibers are placed
        if len(specified_fiber_centers) == 0:
            # For each fiber, choose a random triangle to seed within
            # Values to seed random location within a triangle
            randvals = np.random.rand(num_fibers,2)
            choice_weight_pairs = [(i, areas[i]) for i in range(len(areas))]
            for i in range(num_fibers):
                tri_index = weighted_choice(choice_weight_pairs)
                bary_coord = randvals[i]
                # Constraint of barycentric coordinates (l1 + l2 + l3 = 1)
                bary_coord = bary_coord / np.sum(bary_coord)
                t = self.triangulation['triangles'][tri_index]
                A = np.column_stack((points[t[0]], points[t[1]], points[t[2]]))
                random_point = (1 - math.sqrt(randvals[i][0]))*points[t[0]] + math.sqrt(randvals[i][0])*(1 - randvals[i][1])*points[t[1]] + math.sqrt(randvals[i][0])*randvals[i][1]*points[t[2]]
                self.fibers.append(
                    self.PeriodicMasterFiber(random_point,
                    self.radii[i],
                    i,
                    self.boundaries))
                    
        # This is the case if fiber centers were specified
        else:
            num_fibers = len(self.radii)
            if num_fibers != len(specified_fiber_centers):
                raise ValueError('The number of fiber centers given does not match the number of fibers specified earlier in the initialize call.')
            for i in range(num_fibers):
                self.fibers.append(
                        self.PeriodicMasterFiber(specified_fiber_centers[i],
                        self.radii[i],
                        i,
                        self.boundaries))

        for fiber in self.fibers:
            fiber.adjust_for_bounds(self.boundaries)

        self.fiber_vf = sum([math.pi*r**2 for r in self.radii])/rve_area
        return (self.fiber_vf, rve_area)
        
    def get_fiber_centers(self):
        fiber_centers = []
        for fiber in self.fibers:
            fiber_centers.append(fiber.center)
        return fiber_centers
    
    def solve_fiber_locations(self,
        min_spacing_ratio,
        visualize=False,
        show_final=False):
        """Assumes that the RVE contains a bunch of fibers, but the placement
        of them may violate boundaries or overlap with other fibers.  This method
        will take that arrangement of fibers and solve for the non-overlapping
        configuration.
        """
        start = time.time()
        num_diams_for_search = 4
        num_diams_for_update = 3
        self._recalculate_neighbors(num_diams_for_search)
        min_space_between_fibers = min_spacing_ratio * np.average(self.radii)
        
        frame = None
        fig = None
        ax = None
        if visualize:
            if os.path.isdir('frames'):
                shutil.rmtree('frames')
            os.mkdir('frames')
            frame=0
            fig = plt.figure()
            (fig, ax) = self.draw(fig, ax, frame)

        iteration_count = 0
        iteration_max = 10000
        iterations_no_overlap = 0
        while(iterations_no_overlap < 3):
            found_overlap = self._iterate_on_interference(self.boundaries, min_space_between_fibers)
            if visualize:
                frame += 1
                self.draw(fig, ax, frame)
            need_to_recalc_neighbors = False
            for fiber in self.fibers:
                # if any fiber has moved more than half the neighbor search
                # distance minus the radius, then we need to re-calculate the
                # neighbors
                if fiber.get_distance_since_last_neighbor_update() > \
                   fiber.radius * (num_diams_for_update - 1):
                    need_to_recalc_neighbors = True
                    break
            if need_to_recalc_neighbors or found_overlap == False:
                self._recalculate_neighbors(num_diams_for_search)
            if found_overlap == False: iterations_no_overlap += 1
            else: iterations_no_overlap = 0
            iteration_count += 1
            if iteration_count >= iteration_max:
                raise RuntimeError('Maximum iterations exceeded.  Check input parameters to make sure an RVE is possible.')

        elapsed = time.time() - start
        #print("Total Time to generate RVE Geometry: " + str(elapsed) + ' seconds')
        
        if show_final:
            self.draw(fig,ax)
            plt.savefig('RandomFiberMatrix.png',bbox_inches='tight')
            #plt.show()
    
    def _recalculate_neighbors(self, fiber_diams_to_search):
        """Recalculates ghost fibers and neighbor fibers"""
        fibers_with_ghosts = []
        for fiber in self.fibers:
            fiber.calc_ghost_fibers(self.boundaries)
            fibers_with_ghosts.append(fiber)
            fibers_with_ghosts.extend(fiber.ghost_fibers)
        
        fiber_centers = []
        for fiber in fibers_with_ghosts:
            fiber_centers.append(fiber.center)
        fiber_center_kd_tree = spatial.KDTree(fiber_centers)
        
        for fiber in self.fibers:
            fiber.update_neighbors(fiber_center_kd_tree,
                 fibers_with_ghosts,
                 fiber_diams_to_search)

    def _iterate_on_interference(self, boundaries, min_space_between_fibers):
        """Loops through all fibers and resolves interference if it is found to exist
        """
        found_overlap = False
        for fiber in self.fibers:
            found_overlap |= fiber.fix_overlap_with_neighbors(boundaries, min_space_between_fibers)
        return found_overlap

    # ---------------------------------------------------- #
    # Method for modifying the current fiber configuration #
    # ---------------------------------------------------- #

    def lower_fiber_vf(self, target_fiber_vf):
        if target_fiber_vf >= self.fiber_vf:
            raise RuntimeError('The target fiber volume fraction must be lower the current fiber volume fraction.')
        vf_ratio = target_fiber_vf / self.fiber_vf
        for fiber in self.fibers:
            fiber.radius = math.sqrt(vf_ratio) * fiber.radius
        self.fiber_vf = target_fiber_vf
    
    # ------------------- #
    # Drawing methods #
    # ------------------- #
    
    def draw(self, fig=None, ax=None, frame=None, label_fibers=False, label_boundaries=False):
        """Draws the RVE on the given figure and axes.  If None is
            given for the figure and axes object, a new one will be
            created and returned after drawing.
        """
        show_fig = False
        if fig is None:
            fig = plt.figure()
            show_fig = True
        if ax is None: ax = plt.axes()
        else: ax.clear()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect('equal')
        
        self._draw_boundaries(ax, label_boundaries)
        for fiber in self.fibers: fiber.draw(ax, label_fibers=label_fibers)
        
        ax.autoscale()
        ax.margins(0.1)
        
        # Create a postscript drawing for the frame if the frame
        # number is provided
        if frame is not None:
            plt.savefig(os.path.join('frames',
                str(frame)+'_RVE.png'),
                transparent=True,
                bbox_inches='tight')

        if show_fig: plt.show()
                
        return (fig, ax)
        
    def _draw_boundaries(self, ax, label_boundaries):
        lines = [ [tuple(point) for point in boundary.points] for boundary in self.boundaries ]
        colors = np.array([boundary.type.get_color() for boundary in self.boundaries])
        lc = matplotlib.collections.LineCollection(lines, colors=colors, linewidths=2)
        ax.add_collection(lc)
        if label_boundaries:
            for i in range(len(self.boundaries)):
                avg_x = np.average([p[0] for p in self.boundaries[i].points])
                avg_y = np.average([p[1] for p in self.boundaries[i].points])
                ax.text(avg_x, avg_y, str(i), multialignment='right', fontsize=18)

    # --------------- #
    # Meshing methods #
    # --------------- #
   
    def create_triangle_input(self, ax=None, num_nodes_around_fiber=20):
        # Create points around fibers
        all_fibers = self._create_fiber_boundary_nodes(num_nodes_around_fiber)

        # Create segments for fibers
        fiber_segments = []
        for fiber_index in range(len(all_fibers)):
            fiber = all_fibers[fiber_index]
            # Create segments around the fiber
            for i in range(len(fiber.boundary_nodes)-1):
                # Boundary segments along edge of RVE will be generated later
                if fiber.boundary_nodes[i].lies_on_rve_boundary == False or fiber.boundary_nodes[i+1].lies_on_rve_boundary == False:
                    fiber_segments.append(self.BoundarySegment([fiber.boundary_nodes[i], fiber.boundary_nodes[i+1]], fiber_index + 1))
            # The last term will catch the case that there are no points for the fiber other than the intersection
            # points, which can happen if a tiny sliver of the fiber crosses the boundary.  We need a segment between them
            if len(fiber.boundary_nodes) > 1:
                if (fiber.boundary_nodes[-1].lies_on_rve_boundary == False or fiber.boundary_nodes[0].lies_on_rve_boundary == False) or (len(fiber.boundary_nodes) == 2 and fiber.boundary_nodes[0].boundary != fiber.boundary_nodes[1].boundary):
                    fiber_segments.append(self.BoundarySegment([fiber.boundary_nodes[-1], fiber.boundary_nodes[0]], fiber_index + 1))
            
        # Now create points along RVE boundary
        rve_vertices_nodes = [self.BoundaryNode(p, lies_on_rve_boundary=True) for p in self.boundary_points]
        rve_boundary_nodes = set()
        rve_boundary_segments = []
        for b_index in range(len(self.boundaries)):
            boundary = self.boundaries[b_index]
            boundary_nodes = [rve_vertices_nodes[i] for i in boundary.point_indices]
            boundary_nodes.extend(boundary.fiber_intersection_nodes)
            boundary_nodes = sorted(boundary_nodes, key = lambda n: np.linalg.norm(n.point - boundary.points[0]))
            rve_boundary_nodes.update(boundary_nodes)
            for i in range(len(boundary_nodes)-1):
                rve_boundary_segments.append(self.BoundarySegment([boundary_nodes[i], boundary_nodes[i+1]], -(b_index+1)))

        # Plot for the user to see the input data
        if ax is not None: self._draw_triangle_input(ax, fiber_segments, rve_boundary_segments)

        # Build the single list of points
        all_nodes = set()
        for fiber in all_fibers:
            all_nodes.update(fiber.boundary_nodes)
        all_nodes.update(rve_boundary_nodes)
        curr_num = 0
        self.mesh_points = []
        for n in all_nodes:
            n.index = curr_num
            curr_num += 1
            self.mesh_points.append(list(n.point))

        self.mesh_segments = []
        self.mesh_segment_boundary_markers = []
        for segment in fiber_segments:
            point_indices = [n.index for n in segment.boundary_nodes]
            self.mesh_segments.append(point_indices)
            # Markers need to be 1 based because any new edges are given a 0
            self.mesh_segment_boundary_markers.append(int(segment.marker))
        rve_marker = np.max(self.mesh_segment_boundary_markers)
        for segment in rve_boundary_segments:
            point_indices = [int(n.index) for n in segment.boundary_nodes]
            self.mesh_segments.append(point_indices)
            self.mesh_segment_boundary_markers.append(int(segment.marker))
        self.mesh_fiber_centers = [list(f.center) for f in all_fibers]
        self.mesh_fiber_radii = [float(f.radius) for f in all_fibers]
        self.mesh_fiber_internal_coords = [list(f.coord_inside_rve) for f in all_fibers]

    def _create_fiber_boundary_nodes(self, num_nodes_around_fiber):
        # A fast sorting function
        def get_sort_tuple(x, radius):
                quadrant = 0
                if x[0] >= 0.0 and x[1] >= 0.0: quadrant = 1
                elif x[0] < 0.0 and x[1] >= 0.0: quadrant = 2
                elif x[0] < 0.0 and x[1] < 0.0: quadrant = 3
                else: quadrant = 4
                if x[0] == 0.0: theta = math.copysign(1.0, x[1])*math.pi/2.0
                else: theta = math.atan(x[1] / x[0])
                #if quadrant == 2:
                #signed_value = math.copysign(1.0, x[1]) * (radius - x[0])
                return (quadrant, theta)

        # Get a list with all fibers
        fibers_with_ghosts: List['Fiber'] = []
        for fiber in self.fibers:
            fiber.calc_ghost_fibers(self.boundaries)
            fibers_with_ghosts.append(fiber)
            fibers_with_ghosts.extend(fiber.ghost_fibers)

        # Clear the boundary nodes around each fiber
        for fiber in list(fibers_with_ghosts):
            fiber.boundary_nodes.clear()

        # Connect fibers that touch with a node
        # I decided that it is not useful to connect the fibers
        #self._connect_fibers(fibers_with_ghosts)
    
        # Calculate boundary intersections and remove
        # fibers that do not lie in the RVE domain (since ghost
        # fibers could be outside)
        for fiber in list(fibers_with_ghosts):
            # Get intersection info
            intersection_nodes = []
            intersected_boundaries = []
            for i in range(len(self.boundaries)):
                boundary = self.boundaries[i]
                if boundary.check_collision(fiber):
                    intersected_boundaries.append(boundary)
                    for intersection in boundary.get_intersection_points(fiber):
                        intersection_nodes.append(self.BoundaryNode(intersection, lies_on_rve_boundary=True, boundary=boundary))
                        boundary.fiber_intersection_nodes.append(intersection_nodes[-1])
            
            # Check that the fiber is within the RVE domain or at least
            # intersecting a boundary
            is_in_RVE = False
            for t in self.triangulation['triangles']:
                v0 = self.triangulation['vertices'][t[0]]
                v1 = self.triangulation['vertices'][t[1]] - v0
                v2 = self.triangulation['vertices'][t[2]] - v0
                det = np.cross(v1, v2)
                if det == 0: continue
                a = (np.cross(fiber.center, v2) - np.cross(v0, v2))/np.cross(v1, v2)
                b = -(np.cross(fiber.center, v1) - np.cross(v0, v1))/np.cross(v1, v2)
                if a >= 0 and b >= 0 and a + b <= 1.0:
                    is_in_RVE = True
                    break
            # If it is outside with no intersections, skip this fiber
            if is_in_RVE == False and len(intersected_boundaries) == 0:
                fibers_with_ghosts.remove(fiber)
                continue

            # Start with evenly spaced nodes around the fiber
            nodes_around_fiber = []
            for i in range(num_nodes_around_fiber):
                theta = 2.0*math.pi / num_nodes_around_fiber * i
                offset = np.array([fiber.radius*math.cos(theta),
                    fiber.radius*math.sin(theta)])
                nodes_around_fiber.append(self.BoundaryNode(fiber.center + offset))
                nodes_around_fiber[-1].index = i
            # Remove nodes happen to lie on a connection node
            for n in fiber.boundary_nodes:
                for n2 in  list(nodes_around_fiber):
                    if np.linalg.norm(n.point - n2.point) < 1e-10:
                        nodes_around_fiber.remove(n2)
            fiber.boundary_nodes.extend(nodes_around_fiber)
			
			# Remove nodes happen to lie on an intersection node
            tolerance = 2.0 * math.pi * fiber.radius / 20.0 / 10.0
            for n in intersection_nodes:
                for n2 in  list(fiber.boundary_nodes):
                    if not n2.lies_on_rve_boundary and np.linalg.norm(n.point - n2.point) < tolerance:
                        fiber.boundary_nodes.remove(n2)
						
            # Add intersections and sort by distance from starting point
            fiber.boundary_nodes.extend(intersection_nodes)
            fiber.boundary_nodes = sorted(fiber.boundary_nodes, key=lambda n: get_sort_tuple(n.point - fiber.center, fiber.radius))
						
            # Check there is at least one node between intersections
            additional_boundary_nodes: List[BoundaryNode] = []
            for i in range(len(fiber.boundary_nodes)):
                # If two boundary points in a row lie on the boundary, then we need to add one between them
                if fiber.boundary_nodes[i-1].lies_on_rve_boundary and fiber.boundary_nodes[i].lies_on_rve_boundary:
                    avg_x = (fiber.boundary_nodes[i-1].point[0] + fiber.boundary_nodes[i].point[0]) / 2.0
                    avg_y = (fiber.boundary_nodes[i-1].point[1] + fiber.boundary_nodes[i].point[1]) / 2.0
                    avg_point = np.array([avg_x, avg_y])
                    center_to_avg = avg_point - np.array(fiber.center)
                    distance_to_center = np.linalg.norm(center_to_avg)
                    scale_factor = fiber.radius / distance_to_center
                    point_on_circle = scale_factor * center_to_avg + np.array(fiber.center)
                    additional_boundary_nodes.append(self.BoundaryNode(point_on_circle))
            fiber.boundary_nodes.extend(additional_boundary_nodes)
            fiber.boundary_nodes = sorted(fiber.boundary_nodes, key=lambda n: get_sort_tuple(n.point - fiber.center, fiber.radius))

            # Throw out points that do not lie within the RVE, note the point
            # just has to lie within one intersected boundary to be in the RVE
            for n in list(fiber.boundary_nodes):
                if not n.lies_on_rve_boundary:
                    node_lies_inside_rve = True
                    for b in intersected_boundaries:
                        relative_position = b.get_point_relative_position(n.point)
                        if relative_position < 0.0: node_lies_inside_rve = False
                    if not node_lies_inside_rve:
                        fiber.boundary_nodes.remove(n)

            fiber.coord_inside_rve = [ np.average([n.point[0] for n in fiber.boundary_nodes]),
                np.average([n.point[1] for n in fiber.boundary_nodes]) ]

        return fibers_with_ghosts

    def _connect_fibers(self, fibers_with_ghosts):
        # Now make sure there is a node where fibers touch
        for i in range(len(fibers_with_ghosts)):
            for j in  range(i+1, len(fibers_with_ghosts)):
                fiber = fibers_with_ghosts[i]
                neighbor = fibers_with_ghosts[j]
                # Make sure the neighbor is not already connected
                if neighbor not in fiber.connected_fibers:
                    vec_between_fibers = np.array(neighbor.center) - np.array(fiber.center)
                    dist_between_fibers = np.linalg.norm(vec_between_fibers) - fiber.radius - neighbor.radius
                    # If the distance between the fibers is 1% of the average of the radii
                    if dist_between_fibers < np.average([fiber.radius, neighbor.radius]) * 0.01:
                        unit_vec_between_fibers = vec_between_fibers / np.linalg.norm(vec_between_fibers)
                        connection_point = np.array(fiber.center) + unit_vec_between_fibers*(fiber.radius + dist_between_fibers/2.0)
                        connection_node = self.BoundaryNode(connection_point)
                        fiber.boundary_nodes.append(connection_node)
                        fiber.connected_fibers.append(neighbor)
                        neighbor.boundary_nodes.append(connection_node)
                        neighbor.connected_fibers.append(fiber)
    
    def _draw_triangle_input(self, ax, fiber_segments, rve_boundary_segments):
        # Plot the input data
        ax.clear()

        lines = [ [tuple(n.point) for n in segment.boundary_nodes] for segment in fiber_segments ]
        colors = np.array([(0, 0, 1, 1) for segment in fiber_segments])
        lc = matplotlib.collections.LineCollection(lines, colors=colors, linewidths=2)
        ax.add_collection(lc)

        lines = [ [tuple(n.point) for n in segment.boundary_nodes] for segment in rve_boundary_segments ]
        colors = np.array([(1, 0, 0, 1) for segment in rve_boundary_segments])
        lc = matplotlib.collections.LineCollection(lines, colors=colors, linewidths=2)
        ax.add_collection(lc)

        ax.autoscale()
        ax.margins(0.1)

    def create_mesh(self, mesh_name, max_triangle_area=-1.0, output_mesh_input=True, output_debug_meshes=False, output_vtk=True, output_pfec=True, output_plt=True, ax=None):
        """Creates the mesh for the RVE, write the mesh input data to a file
        entitled [mesh_name].mesh_input for debugging purposes, plots the mesh
        if it is small enough to be plotted in matplotlib reasonably, and
        writes the final mesh to [mesh_name].vtk
        Returns True if it tried to plot but the mesh was too big and
        False if there is no error.
        """
        # Update indices
        for b_index in range(len(self.boundaries)):
            self.boundaries[b_index].index = b_index

        # Create pairs of markers for periodic boundaries (markers are
        # negative and 1 based for RVE boundaries)
        periodic_marker_pairs = []
        periodicity_vectors = []
        periodic_boundary_set = set()
        for boundary in self.boundaries:
            if boundary.type == self.BoundaryType.PERIODIC and boundary not in periodic_boundary_set:
                periodic_marker_pairs.append([-int(boundary.index+1), -int(boundary.pair.index+1)])
                periodicity_vectors.append(list(boundary.get_periodic_vector()))
                periodic_boundary_set.add(boundary)
                periodic_boundary_set.add(boundary.pair)
        
        # For debugging, dump mesh data
        if output_mesh_input:
            out = open(mesh_name + '.mesher_input', 'w')
            out.write(str(len(self.mesh_points)) + '\n')
            for p in self.mesh_points:
                out.write(str(p[0]) + '\t' + str(p[1]) + '\n')
            out.write(str(len(self.mesh_segments)) + '\n')
            for s in self.mesh_segments:
                out.write(str(s[0]) + '\t' + str(s[1]) + '\n')
            out.write(str(len(self.mesh_segment_boundary_markers)) + '\n')
            for s in self.mesh_segment_boundary_markers:
                out.write(str(s) + '\n')
            out.write(str(len(periodic_marker_pairs)) + '\n')
            for pair in periodic_marker_pairs:
                out.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
            for vec in periodicity_vectors:
                out.write(str(vec[0]) + '\t' + str(vec[1]) + '\n')
            out.write(str(len(self.mesh_fiber_centers)) + '\n')
            for vec in self.mesh_fiber_centers:
                out.write(str(vec[0]) + '\t' + str(vec[1]) + '\n')
            for s in self.mesh_fiber_radii:
                out.write(str(s) + '\n')
            for vec in self.mesh_fiber_internal_coords:
                out.write(str(vec[0]) + '\t' + str(vec[1]) + '\n')
            out.write(str(max_triangle_area) + '\n')
            out.close()

        mesh = BM.PyMesh()
        BM.PyMeshCreationFiberMatrix.set_output_debug_flag(output_debug_meshes)
        BM.PyMeshCreationFiberMatrix.make_fiber_matrix_rve_mesh(mesh,
            self.mesh_points,
            self.mesh_segments,
            self.mesh_segment_boundary_markers,
            periodic_marker_pairs,
            periodicity_vectors,
            self.mesh_fiber_centers,
            self.mesh_fiber_radii,
            self.mesh_fiber_internal_coords,
            max_triangle_area)
        if output_vtk:
            BM.PyMeshIO.write_VTK_mesh_file(mesh,
                mesh_name + '.vtk',
                False,
                False)
        if output_plt:
            BM.PyMeshIO.write_plt_mesh_file(mesh,
                mesh_name + '.plt')
            BM.PyMeshIO.write_elemat_file(mesh,
                mesh_name + '.elemat')
        if output_pfec:
            BM.PyMeshIODistributed.write_pfec_files(mesh,
                mesh_name)

        # If an axes object is given and the mesh is small enough then draw it
        if ax is not None:
            # https://matplotlib.org/api/collections_api.html#matplotlib.collections.PolyCollection 
            vertices = PyMeshCreationFiberMatrix.get_plot_polys(mesh, 100000)
            colors = PyMeshCreationFiberMatrix.get_plot_poly_colors(mesh, 100000)
            if len(vertices) == 0: return True
            ax.clear()
            pc = matplotlib.collections.PolyCollection(vertices, facecolors=colors)
            ax.add_collection(pc)
        
        return mesh  

    # -------------- #    
    # Output methods #
    # -------------- #

    def write_rve(self, filepath):
        out = open(filepath, 'w')
        out.write('Boundaries:\n')
        boundary_index_map = { self.boundaries[i]: i for i in range(len(self.boundaries)) }
        out.write(str(len(self.boundaries)) + '\n')
        for boundary in self.boundaries:
            if boundary.pair is not None:
                boundary.write(out, boundary_index_map[boundary.pair])
            else:
                boundary.write(out)
        out.write('Fibers:\n')
        out.write(str(len(self.fibers)) + '\n')
        for fiber in self.fibers:
            fiber.write(out)
        out.close()

    # This doesn't work because too much info is missing from the RVE file
    # def read_rve(self, filepath):
    #     rve_file = open(filepath)
    #     rve_file.readline() # Expected to be 'Boundaries:'
    #     num_boundaries = int(rve_file.readline())
    #     for i in range(num_boundaries):
    #         self.boundaries.append(self.LinearBoundary.read(rve_file))
    #     for boundary in self.boundaries:
    #         if boundary.pair_index > 0:
    #             boundary.pair = self.boundaries[boundary.pair_index]
    #             del boundary.pair_index
    #         boundary.btype = self.BoundaryType[boundary.btype_name]
    #         del boundary.btype_name
    #     rve_file.readline() # Expected to be 'Fibers:'
    #     num_fibers = int(rve_file.readline())
    #     for i in range(num_fibers):
    #         self.fibers.append(self.PeriodicMasterFiber.read(index=i, rve_file=rve_file))
    #     rve_file.close()

    @classmethod
    def test_mesher_input_file(cls, filepath, max_triangle_area):

        in_file = open(filepath)

        num_mesh_points = int(in_file.readline())
        mesh_points = []
        for i in range(num_mesh_points):
            tokens = in_file.readline().split('\t')
            mesh_points.append(list(map(float, tokens)))

        num_mesh_segments = int(in_file.readline())
        mesh_segments = []
        for i in range(num_mesh_segments):
            tokens = in_file.readline().split('\t')
            mesh_segments.append(list(map(int, tokens)))

        num_mesh_segment_boundary_markers = int(in_file.readline())
        mesh_segment_boundary_markers= []
        for i in range(num_mesh_segment_boundary_markers):
            mesh_segment_boundary_markers.append(int(in_file.readline()))

        num_periodic_pairs = int(in_file.readline())
        periodic_marker_pairs = []
        for i in range(num_periodic_pairs):
            tokens = in_file.readline().split('\t')
            periodic_marker_pairs.append(list(map(int, tokens)))

        periodicity_vectors = []
        for i in range(num_periodic_pairs):
            tokens = in_file.readline().split('\t')
            periodicity_vectors.append(list(map(float, tokens)))

        num_mesh_fibers = int(in_file.readline())
        mesh_fiber_centers = []
        for i in range(num_mesh_fibers):
            tokens = in_file.readline().split('\t')
            mesh_fiber_centers.append(list(map(float, tokens)))
        mesh_fiber_radii = []
        for i in range(num_mesh_fibers):
            tokens = in_file.readline().split('\t')
            mesh_fiber_radii.append(float(tokens[0]))
        mesh_fiber_internal_coords = []
        for i in range(num_mesh_fibers):
            tokens = in_file.readline().split('\t')
            mesh_fiber_internal_coords.append(list(map(float, tokens)))
        in_file.close()

        mesh = BM.PyMesh()
        BM.PyMeshCreationFiberMatrix.set_output_debug_flag(True)
        BM.PyMeshCreationFiberMatrix.make_fiber_matrix_rve_mesh(mesh,
            mesh_points,
            mesh_segments,
            mesh_segment_boundary_markers,
            periodic_marker_pairs,
            periodicity_vectors,
            mesh_fiber_centers,
            mesh_fiber_radii,
            mesh_fiber_internal_coords,
            max_triangle_area)
        BM.PyMeshIO.write_VTK_mesh_file(mesh,
            'FiberMatrixRVE_2D.vtk',
            False,
            False)
        return mesh

    # ------------------- #
    # Mesh helper classes #
    # ------------------- #

    class BoundaryNode:
        point: np.array
        lies_on_rve_boundary: bool
        index: int
        boundary: 'LinearBoundary'

        """Node info for the boundary of fibers or the rve."""
        def __init__(self, point, lies_on_rve_boundary=False, boundary=None):
            self.point = np.array(point)
            self.index = -1
            self.lies_on_rve_boundary = lies_on_rve_boundary
            self.boundary = boundary

    class BoundarySegment:
        def __init__(self, boundary_nodes, marker = -1):
            self.boundary_nodes = boundary_nodes
            self.marker = marker

    # ------------------  #    
    # Boundary classes #
    # ------------------  #
    
    class BoundaryType(Enum):
        """Enumeration class for denoting a type of boundary.
                PERIODIC: fibers can exist across boundary pair to be periodic
                FINITE: fibers cannot cross the boundary
                SYMMETRIC: fibers are only allow to lie exactly half way or not at all
            """
        FINITE = 0
        SYMMETRIC = 1
        PERIODIC = 2
        
        def get_color(self):
            if self.name == 'PERIODIC': return (1, 0, 0, 1)
            if self.name == 'FINITE': return (0, 0, 0, 1)
            if self.name == 'SYMMETRIC': return (0, 0.75, 0, 1)
    
    class LinearBoundary(object):
        """A boundary of the RVE that is defined by a line segment.
            Any boundary class is resposible for testing whether a fiber has
            crossed it.  It is also responsible  for determinging how the fiber 
            should be moved if the fiber is not allowed to cross it or where a
            periodic ghost fiber should be placed if the boundary is periodic.
            
            Attributes:
                points: list of coordinates that define the line segment
                pair: another boundary that is considered a pair fo this boundary
                type: type of boundary, see BoundaryType class
            
            Other types of boundaries such as one defined by a quadratic line
            could be implemented in the future.
        """
        def __init__(self, points, point_indices, btype):
            self.points = [np.array(point) for point in points]
            self.point_indices = point_indices
            self.pair = None
            self.type = btype
            self.fiber_intersection_nodes = []
            self.index = 0
            
        def get_length(self):
            return np.linalg.norm(self.points[1] - self.points[0])
            
        def is_pair(self, other_boundary):
            # Check they are the same class and length
            if self.__class__.__name__ != other_boundary.__class__.__name__:
                return False
            if self.get_length() != other_boundary.get_length():
                return False
            # Check that vectors from an endpoint on one boundary to the other
            # boundary are all the same
            vec1 = self.points[0] - other_boundary.points[1]
            vec2 = self.points[1] - other_boundary.points[0]
            if np.linalg.norm(vec2 - vec1) > 1e-8:
                return False
            # They are a pair if they none of the checks have hit so far
            return True

        def get_shared_points(self, other_boundary, tol=1e-12):
            shared_points = []
            for p1 in self.points:
                for p2 in other_boundary.points:
                    if np.linalg.norm(p2 - p1) < tol:
                        shared_points.append(p1)
            return shared_points

        def get_move_vector(self):
            """Get the vector that should be used to move a fiber if it intersects this boundary."""
            segment_vector = self.points[1] - self.points[0]
            segment_vector = np.append(segment_vector, 0.0)
            segment_vector  = segment_vector / np.linalg.norm(segment_vector)
            z_vector = np.array([0., 0., 1.])
            return np.cross(z_vector, segment_vector)[0:2]
            
        def get_periodic_vector(self):
            if self.pair is not None:
                periodic_vec = self.pair.points[1] - self.points[0]
                return periodic_vec
            return None

        def get_point_relative_position(self, point):
            """Returns: 1 if point lies inside, -1 if it lies outside,
            and 0 if it lies right on the boundary.
            """
            segment_vector = self.points[1] - self.points[0]
            segment_vector_mag = np.linalg.norm(segment_vector)
            unit_segment_vector = segment_vector / segment_vector_mag
            point_to_segment_start = point - self.points[0]
            unit_point_to_segment_start = point_to_segment_start / np.linalg.norm(point_to_segment_start)
            cross_product = np.cross(unit_segment_vector, unit_point_to_segment_start)
            return cross_product
            
        def check_collision(self, fiber, eps=0.0):
            """Returns whether a fiber intersects this boundary"""
            closest_point = self._closest_point_to_fiber(fiber)
            fiber_to_circle_shortest_vector = fiber.center - closest_point
            fiber_to_circle_shortest_vector_mag = np.linalg.norm(fiber_to_circle_shortest_vector)
            if fiber_to_circle_shortest_vector_mag <= (1.0 + eps)*fiber.radius:
                return True
            return False
            
        def get_distance_to_fiber(self, fiber):
            closest_point = self._closest_point_to_fiber(fiber)
            fiber_to_circle_shortest_vector = fiber.center - closest_point
            return np.linalg.norm(fiber_to_circle_shortest_vector)
            
        def _closest_point_to_fiber(self, fiber, within_boundary_segment=True):
            """Find the closest point on the boundary to the fiber."""
            segment_vector = self.points[1] - self.points[0]
            segment_vector_mag = np.linalg.norm(segment_vector)
            circle_to_point_a = fiber.center - self.points[0]
            unit_segment_vector = segment_vector / segment_vector_mag
            projection = np.dot(circle_to_point_a, unit_segment_vector)
            if projection <= 0 and within_boundary_segment: return self.points[0]
            if projection >= segment_vector_mag and within_boundary_segment: return self.points[1]
            projection_vector = unit_segment_vector * projection
            closest_point = projection_vector + self.points[0]
            return closest_point

        def get_intersection_points(self, fiber):
            """Returns the actual x, y coordinates of the fiber/boundary
            intersection.
            """
            closest_point = self._closest_point_to_fiber(fiber, within_boundary_segment=False)
            segment_vector = self.points[1] - self.points[0]
            segment_vector_mag = np.linalg.norm(segment_vector)
            unit_segment_vector = segment_vector / segment_vector_mag
            closest_to_center = fiber.center - closest_point
            c = np.linalg.norm(closest_to_center)
            if c == fiber.radius:
                return [closest_point]
            else:
                ds = math.sqrt(fiber.radius**2 - c**2)
                i1 = closest_point + ds*unit_segment_vector
                i2 = closest_point - ds*unit_segment_vector
                intersections = []
                for p in [i1, i2]:
                    dot_product = np.dot(p - self.points[0], segment_vector)
                    if dot_product >= 0 and dot_product <= segment_vector_mag**2:
                        intersections.append(p)
                return intersections
          
        def write(self, out, pair_index=None):
            out.write(str(self.index) + '\t')
            if self.pair is None: out.write('-1\t')
            else: out.write(str(self.pair.index) + '\t')
            out.write(str(self.type.name) + '\t')
            out.write(str(len(self.points)) + '\t')
            for pi in self.point_indices:
                out.write(str(pi) + '\t')
            for p in self.points:
                out.write(str(p[0]) + '\t')
                out.write(str(p[1]) + '\t')
            out.write('\n')

        @classmethod
        def read(cls, rve_file):
            """This is a factory method to create an instance based on what is read from a
            .rve file.
            """
            tokens = rve_file.readline().split('\t')
            index = int(tokens[0])
            pair_index = int(tokens[1]) # Temporary, must be resolved to set self.pair
            btype_name = tokens[2]
            num_points = int(tokens[3])
            point_indices = [int(tokens[i]) for i in range(4, 4 + num_points)]
            points = []
            for i in range(num_points):
                points.append(np.array([float(tokens[2*i + 4 + num_points]), float(tokens[2*i + 5 + num_points])]))
            newBoundary = cls(points=points, point_indices=point_indices, btype=None)
            newBoundary.index = index
            newBoundary.pair_index = pair_index
            newBoundary.btype_name = btype_name
            return newBoundary

    # -------------  #    
    # Fiber classes #
    # -------------  #
    
    class Fiber:
        radius: float
        boundary_nodes: List['BoundaryNode']
        connected_fibers: List['Fiber']
        neighbors: Set['Fiber']

        """The basic fiber class"""
        def __init__(self,center,radius):
            """Create a fiber with a given center coordinate and radius"""
            self.center = copy.deepcopy(center) # A list of form (x,y)
            self.radius = copy.deepcopy(radius) #radius of fiber
            self.boundary_nodes = [] # Will hold the nodes that define the boundary
            self.connected_fibers = []
            self.coord_inside_rve = [] # will hold a coordinate that lies in the fiber inside the rve
            # A list of nearby fibers to watch.  Will need to be updated occasionally
            self.neighbors = set([])

        def __str__(self):
            return "Fiber Center: " + str(self.center) + '\tr=' + str(self.radius)
            
        def overlaps_fiber(self,OtherFiber, min_space_between_fibers):
            """return true if this fiber overlaps the other fiber, false otherwise"""
            return sqrt(  (OtherFiber.center[0]-self.center[0]) * 
                          (OtherFiber.center[0]-self.center[0]) + 
                          (OtherFiber.center[1]-self.center[1]) * 
                          (OtherFiber.center[1]-self.center[1]) ) < \
                   self.radius + otherFiber.radius + min_space_between_fibers
        
        def get_vector_to_other(self,OtherFiber):
            """Get a vector from this fiber's center to another fiber's center"""
            return [a-b for (a,b) in zip(OtherFiber.center,self.center)]
            
        def move(self,MoveVec):
            """Move the fiber by some vector"""
            self.center = [a+b for (a,b) in zip(self.center,MoveVec)]

        def fix_overlap_with_neighbors(self, boundaries, min_space_between_fibers):
            """Fixes any overlap with neighbors.  
               Return true if any overlap was found.
            """
            overlap_found = False
            for nf in self.neighbors:
                #if nf.__class__.__name__ == 'PeriodicMasterFiber' and nf.get_index() < self.index:
                #    continue #Only move fibers with higher indices
                vec_to_other = self.get_vector_to_other(nf)
                dist = np.linalg.norm(vec_to_other)
                sum_radii = self.radius + nf.radius
                if dist < sum_radii + min_space_between_fibers:
                    overlap_found = True
                    #Now move both fibers back from one another.
                    unit_vec_to_other = vec_to_other / dist
                    proportion_to_move_this_fiber = nf.radius/sum_radii
                    proportion_to_move_other_fiber = 1.0 - proportion_to_move_this_fiber
                    total_move_back = sum_radii + min_space_between_fibers - dist + sum_radii * 1e-3
                    this_nudge_vector = \
                      [-proportion_to_move_this_fiber*total_move_back*a for a in unit_vec_to_other]
                    other_nudge_vector = \
                      [proportion_to_move_other_fiber*total_move_back*a for a in unit_vec_to_other]
                    self.move(this_nudge_vector)
                    self.adjust_for_bounds(boundaries)
                    nf.move(other_nudge_vector)
                    nf.adjust_for_bounds(boundaries)
            return overlap_found
            
        def draw(self, axes, label='', color='b'):
            """Draws the fiber on a matplotlib axes object"""
            axes.add_patch(matplotlib.patches.Circle(self.center,
                self.radius,
                linewidth=0.1,
                edgecolor='b',
                facecolor=color))
            plt.text(self.center[0]-self.radius/3.0, self.center[1]-self.radius/4.0, label, multialignment='center')

        def write(self, out):
            out.write(str(self.center[0]) + '\t')
            out.write(str(self.center[1]) + '\t')
            out.write(str(self.radius) + '\n')

        @classmethod
        def read(cls, rve_file):
            """This is a factory method to create an instance based on what is read from a
            .rve file.
            """
            tokens = rve_file.readline().split('\t')
            center = np.array([float(tokens[0]), float(tokens[1])])
            radius = float(tokens[2])
            return cls(center=center, radius=radius)

                                                     
    class PeriodicMasterFiber(Fiber):
        """A class for storing a fiber that is in the RVE domain
            
            Note: currently limited to rectangular RVEs, need to generalize.
        """
        def __init__(self,
            center,
            radius,
            index,
            boundaries=None,
            ignore_ghost_fibers=False):
            """Generate a PeriodicMasterFiber object"""
            super().__init__(center, radius)
            # If the fiber is within this distance from a boundary, make a periodic ghost fiber
            self.index = copy.deepcopy(index)
            # Create ghost fibers
            self.ghost_fibers = []
            self.ignore_ghost_fibers = ignore_ghost_fibers
            if boundaries is not None:
                self.calc_ghost_fibers(boundaries)
            # How far this fiber has moved since the neighbors were last updated
            self.vec_moved_since_neighbor_update = [0.0, 0.0]
        
        def __str__(self):
            return '\t'.join([str(self.index),
                              str(self.center[0]),
                              str(self.center[1]),
                              str(self.radius)])
                    
        def calc_ghost_fibers(self, boundaries):
            """Make periodic ghost fibers as needed"""
            self.ghost_fibers = []
            if self.ignore_ghost_fibers:
                return
            ghost_centers = []
            for boundary in boundaries:
                if boundary.type == FiberRVE.BoundaryType.PERIODIC and \
                    boundary.get_distance_to_fiber(self) < 3.0*self.radius:
                    #boundary.check_collision(self):
                    ghost_center = np.array([a for a in self.center])
                    ghost_center = ghost_center + boundary.get_periodic_vector()
                    ghost_centers.append(ghost_center)
            intersected_boundaries = []
            for boundary in boundaries:
                if boundary.check_collision(self):
                    intersected_boundaries.append(boundary)
            # If the fiber intersects two boundaries, then move it to the opposite corner as well
            if len(intersected_boundaries) == 2 and intersected_boundaries[0].type == FiberRVE.BoundaryType.PERIODIC \
                and intersected_boundaries[1].type == FiberRVE.BoundaryType.PERIODIC:
                ghost_centers.append(intersected_boundaries[0].get_periodic_vector() + intersected_boundaries[1].get_periodic_vector() + self.center)
            for ghost_center in ghost_centers:
                self.ghost_fibers.append(FiberRVE.PeriodicGhostFiber(ghost_center, self.radius, self))

        def adjust_for_bounds(self, boundaries, eps = 5.0e-2):
            """Checks to make sure this fiber does not violate any specified boundaries
            and will move it if necessary.
            """
            for boundary in boundaries:
                if boundary.type == FiberRVE.BoundaryType.FINITE and \
                    boundary.check_collision(self, eps):
                    move_vector = boundary.get_move_vector()
                    distance_to_center = boundary.get_distance_to_fiber(self)
                    self.center = self.center + abs(distance_to_center - (1.0 + eps)*self.radius)*move_vector[0:2]
                    if np.isnan(self.center).any():
                        raise ValueError()
                    
                if boundary.type == FiberRVE.BoundaryType.SYMMETRIC and \
                    boundary.check_collision(self, eps):
                    move_vector = boundary.get_move_vector()
                    distance_to_center = boundary.get_distance_to_fiber(self)
                    if distance_to_center < self.radius / 2.0: # Moves center of fiber to line
                        self.center = self.center - abs(distance_to_center - (1.0 + eps)*self.radius)*move_vector
                    else:
                        self.center = self.center + abs(distance_to_center - (1.0 + eps)*self.radius)*move_vector
                
        def move(self, move_vec):
            """Move this fiber and all of its ghost fibers"""
            super().move(move_vec)
            self.vec_moved_since_neighbor_update = \
                [a+b for a,b in zip(move_vec,
                                    self.vec_moved_since_neighbor_update)]
            for ghost in self.ghost_fibers:
                ghost.move_actual(move_vec)
                
        def get_distance_since_last_neighbor_update(self):
            """Returns total distance moved since neighbors were last updated"""
            return math.sqrt(sum( 
                [a*a for a in self.vec_moved_since_neighbor_update] ) )
        
        def get_index(self):
            """Retuns the index"""
            return self.index

        def get_all_copies(self):
            """Returns a list of this fiber and all of the ghost copies of it."""
            fibers = list(self.ghost_fibers)
            fibers.append(self)
            return fibers
            
        def update_neighbors(self,
            fiber_center_kd_tree,
            fibers_with_ghosts,
            fiber_diams_to_search):
            """Finds Neighbor Fibers
            
            Finds fibers within a certain distance of this one and adds to
            NeighborFibers
            
            Relies on a KD tree search to improve performance
            """
            self.neighbors = set([])
            self.vec_moved_since_neighbor_update = [0.0, 0.0]
            # Get the index numbers of the fibers within a certain distance of this one
            neighbor_fiber_indices = set(
                fiber_center_kd_tree.query(self.center,
                     k=100,
                     p=2,
                     distance_upper_bound = self.radius*2.0*fiber_diams_to_search)[1])
            # When you ask the KD tree for more nearest-neighbors than there
            # actaully are within the given tolerance, it pads the result with the
            # number of nearest neighbor values you asked for (in this case,
            # len(AllFibers)).  We don't want this value in NeighborFiberIndices.
            neighbor_fiber_indices.remove(len(fibers_with_ghosts))
            for i in neighbor_fiber_indices:
                if fibers_with_ghosts[i].get_index() > self.index:
                    self.neighbors.add(fibers_with_ghosts[i])
            for gf in self.ghost_fibers:
                gf.update_neighbors(fiber_center_kd_tree,
                                          fibers_with_ghosts,
                                          fiber_diams_to_search)
        
        def get_vector_to_other_periodic(self, other_fiber):
            """
            Returns the shortest vector going from this fiber to the other fiber
            
            Accounts for periodicity        
            """
            shortest_distance = None
            shortest_vec = None
            for f1 in self.get_all_copies():
                for f2 in other_fiber.get_all_copies():
                    vec = self.get_vector_to_other(other_fiber)
                    vec_length = np.linalg.norm(vec)
                    if shortest_distance is None or \
                        vec_length < shortest_distance:
                        shortest_vec = vec
                        shortest_distance = vec_length
            return np.array(shortest_vec)

        def fix_overlap_with_neighbors(self, boundaries, min_space_between_fibers):
            found_overlap = False
            found_overlap |= super().fix_overlap_with_neighbors(boundaries, min_space_between_fibers)
            for ghost_fib in self.ghost_fibers:
                found_overlap |= ghost_fib.fix_overlap_with_neighbors(boundaries, min_space_between_fibers)
            return found_overlap

        def draw(self, axes, label_fibers=False):
            """Draws this fiber and all ghost fibers on the matplotlib axes object"""
            label = ''
            if label_fibers: label = str(self.index)
            super().draw(axes, label=label)
            for ghost in self.ghost_fibers:
                ghost.draw(axes, label=label, color='gray')

        @classmethod
        def read(cls, index, rve_file):
            """This is a factory method to create an instance based on what is read from a
            .rve file.
            """
            tokens = rve_file.readline().split('\t')
            center = np.array([float(tokens[0]), float(tokens[1])])
            radius = float(tokens[2])
            return cls(center=center, radius=radius, index=index)

    class PeriodicGhostFiber(Fiber):
        """
        This class stores extra fibers that are periodic to a master fiber.
        When the master fiber moves, the periodic fibers should move as well.
        Similarly, moving this fiber will move the master fiber
        """
        def __init__(self,center,radius,MasterFiber):
            """Create PeriodicGhostFiber with a reference to the master fiber"""
            super().__init__(center, radius)
            self.MasterFiber = MasterFiber
        
        def move(self,MoveVec):
            """
            Move the master fiber
            Note that this will move all the master's ghost fibers including this
            one
            """
            self.MasterFiber.move(MoveVec)
        
        def move_actual(self,MoveVec):
            """A method for actually moving this fiber and only this fiber"""
            super().move(MoveVec)
        
        def get_index(self):
            """Retuns the index of the master fiber"""
            return self.MasterFiber.get_index()

        def adjust_for_bounds(self, boundaries, eps = 5.0e-2):
            self.MasterFiber.adjust_for_bounds(boundaries, eps)

        def update_neighbors(self,
            fiber_center_kd_tree,
            fibers_with_ghosts,
            fiber_diams_to_search):
            """Finds Neighbor Fibers
            
            Finds fibers within a certain distance of this one and adds to
            NeighborFibers
            
            Relies on a KD tree search to improve performance
            """
            self.neighbors = set([])
            self.vec_moved_since_neighbor_update = [0.0, 0.0]
            # Get the index numbers of the fibers within a certain distance of this one
            neighbor_fiber_indices = set(
                fiber_center_kd_tree.query(self.center,
                     k=100,
                     p=2,
                     distance_upper_bound = self.radius*2.0*fiber_diams_to_search)[1])
            # When you ask the KD tree for more nearest-neighbors than there
            # actaully are within the given tolerance, it pads the result with the
            # number of nearest neighbor values you asked for (in this case,
            # len(AllFibers)).  We don't want this value in NeighborFiberIndices.
            neighbor_fiber_indices.remove(len(fibers_with_ghosts))
            for i in neighbor_fiber_indices:
                if fibers_with_ghosts[i].get_index() > self.MasterFiber.index:
                    self.neighbors.add(fibers_with_ghosts[i])
    
# -------- #
# Examples #
# -------- #


def square_periodic_example():
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=20,
        vf=0.6,
        avg_diam=5e-6,
        diam_std_dev=0.0,
        rve_aspect_ratio=1.0,
        fixed_height=None
    )
    initial_fiber_centers = np.load('test_initial_fiber_centers.npy')
    rve.place_initial_fibers(specified_fiber_centers=initial_fiber_centers)
    rve.solve_fiber_locations(min_spacing_ratio=0.1, visualize=True)

    final_fiber_centers = np.load('test_final_fiber_centers.npy')
    norm_diff = np.linalg.norm(np.array(rve.get_fiber_centers()) - final_fiber_centers)
    print(f'norm_diff {norm_diff}')
    assert norm_diff < 1e-5

if __name__=="__main__":
    square_periodic_example()
    
# Archived code
"""
def square_finite_example():
    rve = FiberRVE()
    rve.initialize_rectangle_rve(num_fibers=20,
        vf=0.6,
        avg_diam=5e-6,
        diam_std_dev=0.0,
        rve_aspect_ratio=1.0,
        fixed_height=None,
        num_diams_for_ghost_fibers=4,
        boundary_types=[FiberRVE.BoundaryType.FINITE, FiberRVE.BoundaryType.FINITE, \
            FiberRVE.BoundaryType.FINITE, FiberRVE.BoundaryType.FINITE])
    
    fig, ax = rve.draw()
    plt.draw()
    plt.show()

def square_mixed_example():
    rve = FiberRVE()
    rve.create_rectangular_rve(num_fibers=20,
        vf=0.6,
        avg_diam=5e-6,
        diam_std_dev=0.0,
        rve_aspect_ratio=1.0,
        fixed_height=None,
        num_diams_for_ghost_fibers=4,
        boundary_types=[FiberRVE.BoundaryType.PERIODIC, FiberRVE.BoundaryType.FINITE, \
            FiberRVE.BoundaryType.PERIODIC, FiberRVE.BoundaryType.FINITE])
    
    fig, ax = rve.draw()
    plt.draw()
    plt.show()

def polygon_example():
    points = [(0.5, 0.0),
        (1.0, 0.5),
        (0.5, 1.0),
        (-0.5, 1.0),
        (-1.0, 0.5),
        (-0.5, 0.0)]
    types = [FiberRVE.BoundaryType.PERIODIC,
        FiberRVE.BoundaryType.FINITE,
        FiberRVE.BoundaryType.FINITE,
        FiberRVE.BoundaryType.PERIODIC,
        FiberRVE.BoundaryType.FINITE,
        FiberRVE.BoundaryType.FINITE]
    rve = FiberRVE()
    vf = rve.create_polygon_rve(boundary_points=points,
        boundary_types=types,
        num_fibers=100,
        avg_diam=0.1)
    print('Volume fraction: ', vf)
    
    #fig, ax = rve.draw()
    #plt.draw()
    #plt.show()

def textile_example():
    points = np.loadtxt('CrossSectionPoints.txt')
    types = [FiberRVE.BoundaryType.FINITE for i in range(points.shape[0])]
    rve = FiberRVE()
    vf = rve.create_polygon_rve(boundary_points=points,
        boundary_types=types,
        num_fibers=3000,
        avg_diam=1e-2)
    print('Volume fraction: ', vf)

    fig, ax = rve.draw(label_fibers=False)
    plt.draw()
    plt.show()
"""

