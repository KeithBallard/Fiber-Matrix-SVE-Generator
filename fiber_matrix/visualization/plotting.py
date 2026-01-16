try:
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches
except ImportError:
    matplotlib = None
    plt = None
import numpy as np
from typing import List, Optional, Tuple, Any
from fiber_matrix.models.boundary import LinearBoundary
from fiber_matrix.models.fiber import Fiber

def draw_rve(fibers: List[Fiber], boundaries: List[LinearBoundary], fig=None, ax=None, frame=None, label_fibers=False, label_boundaries=False) -> Tuple[Optional[Any], Optional[Any]]:
    """Draws the RVE on the given figure and axes."""
    if matplotlib is None:
        print("Matplotlib not installed, skipping visualization.")
        return (None, None)

    if fig is None:
        fig = plt.figure()
    if ax is None: 
        ax = plt.axes()
    else: 
        ax.clear()
        
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_aspect('equal')
    
    _draw_boundaries(boundaries, ax, label_boundaries)
    for fiber in fibers:
        # We need to manually call draw on fiber or implement it here.
        # Ideally, we move the draw logic here to avoid matplotlib dependency in models.
        _draw_fiber(fiber, ax, label_fibers)
    
    ax.autoscale()
    ax.margins(0.1)
    
    # Create a postscript drawing for the frame if the frame number is provided
    if frame is not None:
        if not os.path.exists('frames'):
            os.mkdir('frames')
        plt.savefig(os.path.join('frames', str(frame)+'_RVE.png'), transparent=True, bbox_inches='tight')
            
    return (fig, ax)

def _draw_boundaries(boundaries: List[LinearBoundary], ax, label_boundaries):
    lines = [ [tuple(point) for point in boundary.points] for boundary in boundaries ]
    colors = np.array([boundary.type.get_color() for boundary in boundaries])
    lc = matplotlib.collections.LineCollection(lines, colors=colors, linewidths=2)
    ax.add_collection(lc)
    if label_boundaries:
        for i in range(len(boundaries)):
            avg_x = np.average([p[0] for p in boundaries[i].points])
            avg_y = np.average([p[1] for p in boundaries[i].points])
            ax.text(avg_x, avg_y, str(i), multialignment='right', fontsize=18)

def _draw_fiber(fiber: Fiber, ax, label_fibers, color='b'):
    """Draws a fiber and its ghosts."""
    _draw_circle(fiber.center, fiber.radius, ax, str(fiber.index) if hasattr(fiber, 'index') and label_fibers else '', color)
    if hasattr(fiber, 'ghost_fibers'):
        for ghost in fiber.ghost_fibers:
             _draw_circle(ghost.center, ghost.radius, ax, str(fiber.index) if hasattr(fiber, 'index') and label_fibers else '', 'gray')

def _draw_circle(center, radius, ax, label, color):
    ax.add_patch(matplotlib.patches.Circle(center, radius, linewidth=0.1, edgecolor='b', facecolor=color))
    if label:
        plt.text(center[0]-radius/3.0, center[1]-radius/4.0, label, multialignment='center')
