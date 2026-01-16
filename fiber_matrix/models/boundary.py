from enum import Enum
import numpy as np
from typing import List, Optional, Tuple

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

class LinearBoundary:
    """A boundary of the RVE that is defined by a line segment.
    """
    def __init__(self, points: List[np.ndarray], point_indices: List[int], btype: BoundaryType):
        self.points = [np.array(point) for point in points]
        self.point_indices = point_indices
        self.pair: Optional['LinearBoundary'] = None
        self.type = btype
        self.fiber_intersection_nodes = []
        self.index = 0
        
    def get_length(self) -> float:
        return np.linalg.norm(self.points[1] - self.points[0])
        
    def is_pair(self, other_boundary: 'LinearBoundary') -> bool:
        if self.get_length() != other_boundary.get_length():
            return False
        vec1 = self.points[0] - other_boundary.points[1]
        vec2 = self.points[1] - other_boundary.points[0]
        if np.linalg.norm(vec2 - vec1) > 1e-8:
            return False
        return True

    def get_periodic_vector(self) -> Optional[np.ndarray]:
        if self.pair is not None:
            return self.pair.points[1] - self.points[0]
        return None

    def get_move_vector(self) -> np.ndarray:
        """Get the vector that should be used to move a fiber if it intersects this boundary."""
        segment_vector = self.points[1] - self.points[0]
        segment_vector = np.append(segment_vector, 0.0)
        segment_vector  = segment_vector / np.linalg.norm(segment_vector)
        z_vector = np.array([0., 0., 1.])
        return np.cross(z_vector, segment_vector)[0:2]

    def get_point_relative_position(self, point: np.ndarray) -> float:
        """Returns: positive if point lies inside, negative if outside (approximately)."""
        segment_vector = self.points[1] - self.points[0]
        segment_vector_mag = np.linalg.norm(segment_vector)
        unit_segment_vector = segment_vector / segment_vector_mag
        point_to_segment_start = point - self.points[0]
        unit_point_to_segment_start = point_to_segment_start / np.linalg.norm(point_to_segment_start)
        cross_product = np.cross(unit_segment_vector, unit_point_to_segment_start)
        return cross_product

    def _closest_point_to_fiber(self, fiber_center: np.ndarray, within_boundary_segment=True) -> np.ndarray:
        segment_vector = self.points[1] - self.points[0]
        segment_vector_mag = np.linalg.norm(segment_vector)
        circle_to_point_a = fiber_center - self.points[0]
        unit_segment_vector = segment_vector / segment_vector_mag
        projection = np.dot(circle_to_point_a, unit_segment_vector)
        if projection <= 0 and within_boundary_segment: return self.points[0]
        if projection >= segment_vector_mag and within_boundary_segment: return self.points[1]
        projection_vector = unit_segment_vector * projection
        return projection_vector + self.points[0]

    def check_collision(self, fiber_center: np.ndarray, fiber_radius: float, eps=0.0) -> bool:
        closest_point = self._closest_point_to_fiber(fiber_center)
        fiber_to_circle_shortest_vector = fiber_center - closest_point
        fiber_to_circle_shortest_vector_mag = np.linalg.norm(fiber_to_circle_shortest_vector)
        if fiber_to_circle_shortest_vector_mag <= (1.0 + eps)*fiber_radius:
            return True
        return False

    def get_distance_to_fiber(self, fiber_center: np.ndarray) -> float:
        closest_point = self._closest_point_to_fiber(fiber_center)
        fiber_to_circle_shortest_vector = fiber_center - closest_point
        return np.linalg.norm(fiber_to_circle_shortest_vector)

    def get_intersection_points(self, fiber_center: np.ndarray, fiber_radius: float) -> List[np.ndarray]:
        closest_point = self._closest_point_to_fiber(fiber_center, within_boundary_segment=False)
        segment_vector = self.points[1] - self.points[0]
        segment_vector_mag = np.linalg.norm(segment_vector)
        unit_segment_vector = segment_vector / segment_vector_mag
        closest_to_center = fiber_center - closest_point
        c = np.linalg.norm(closest_to_center)
        if c == fiber_radius:
            return [closest_point]
        else:
            ds = np.sqrt(fiber_radius**2 - c**2)
            i1 = closest_point + ds*unit_segment_vector
            i2 = closest_point - ds*unit_segment_vector
            intersections = []
            for p in [i1, i2]:
                dot_product = np.dot(p - self.points[0], segment_vector)
                if dot_product >= 0 and dot_product <= segment_vector_mag**2:
                    intersections.append(p)
            return intersections

class BoundaryNode:
    """Node info for the boundary of fibers or the rve."""
    def __init__(self, point: np.ndarray, lies_on_rve_boundary=False, boundary: Optional[LinearBoundary]=None):
        self.point = np.array(point)
        self.index = -1
        self.lies_on_rve_boundary = lies_on_rve_boundary
        self.boundary = boundary

class BoundarySegment:
    def __init__(self, boundary_nodes: List[BoundaryNode], marker = -1):
        self.boundary_nodes = boundary_nodes
        self.marker = marker
