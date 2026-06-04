# Examples

These scripts show how to generate 2D RVEs, extruded 3D RVEs, and fixed fiber
distributions for mesh convergence studies. Run commands from the repository
root.

All examples write files into `examples/output/`.

## 2D RVE Examples

Use these examples when you want a 2D fiber/matrix cross-section mesh.

```bash
python examples/periodic_square_rve.py
python examples/finite_square_rve.py
python examples/periodic_hex_rve.py
python examples/finite_hex_rve.py
```

Each script:

- Initializes the selected RVE geometry.
- Places fibers.
- Solves fiber overlap/spacing.
- Writes `.msh`, `.vtk`, and `.gif` files in `examples/output/`.

The periodic examples create periodic boundary-compatible meshes. The finite
examples keep fibers inside finite exterior boundaries.

## Direct 3D RVE Example

Use this example when you want to generate a new random fiber distribution and
mesh it in one run:

```bash
python examples/periodic_square_rve_3d.py
```

This script:

- Creates a periodic square 2D RVE.
- Solves the fiber distribution.
- Extrudes the 2D RVE into a 3D volume mesh.
- Writes `periodic_square_rve_3d.msh` and `periodic_square_rve_3d.vtk`.

Important controls inside the script:

```python
mesh_size_factor=0.5
z_layers=24
surface_groups=True
composite_surface_groups=False
```

Decrease `mesh_size_factor` for a finer in-plane mesh. Increase `z_layers` for
more through-thickness refinement.

For MOOSE/libMesh, avoid overlapping physical groups. The recommended setting is:

```python
surface_groups=True
composite_surface_groups=False
```

This creates non-overlapping physical groups:

```text
Matrix_Left    Matrix_Right    Matrix_Bottom
Matrix_Top     Matrix_Front    Matrix_Back
Fibers_Left    Fibers_Right    Fibers_Bottom
Fibers_Top     Fibers_Front    Fibers_Back
Matrix         Fibers
```

For a whole composite face in MOOSE, combine matrix and fiber boundaries in the
input file instead of writing an overlapping physical group. For example:

```text
boundary = 'Matrix_Left Fibers_Left'
```

For a matrix-only concentration boundary condition, use only:

```text
boundary = 'Matrix_Left'
```

## Fixed Distribution Workflow For Mesh Convergence

Use this workflow when you want the same fiber distribution for multiple mesh
sizes. This avoids accidentally changing the microstructure every time you
rerun a mesh convergence study.

### Step 1: Generate And Save One Distribution

```bash
python examples/periodic_square_rve_3d_save_distribution.py
```

This creates one solved fiber distribution and writes:

```text
examples/output/periodic_square_rve_3d_distribution.npz
examples/output/periodic_square_rve_3d_distribution.csv
examples/output/periodic_square_rve_3d_save_distribution.gif
```

The `.npz` file is used by the mesh-generation example. The `.csv` file is for
inspection and records:

```text
fiber_id, x, y, radius
```

Useful controls inside the save-distribution script:

```python
NUM_FIBERS = 77
VF = 0.60
AVG_DIAM = 5.0
RVE_ASPECT_RATIO = 1.0
MIN_SPACING_RATIO = 0.1
RANDOM_SEED = None
```

Set `RANDOM_SEED` to an integer if you want reproducible distribution generation
from scratch.

### Step 2: Mesh The Saved Distribution

```bash
python examples/periodic_square_rve_3d_mesh_from_distribution.py
```

This loads:

```text
examples/output/periodic_square_rve_3d_distribution.npz
```

and writes:

```text
examples/output/periodic_square_rve_3d_from_distribution.msh
examples/output/periodic_square_rve_3d_from_distribution.vtk
```

For mesh convergence studies, edit only these values:

```python
MESH_SIZE_FACTOR = 0.5
Z_LAYERS = 24
```

Example convergence sequence:

```python
MESH_SIZE_FACTOR = 1.0
Z_LAYERS = 12
```

```python
MESH_SIZE_FACTOR = 0.75
Z_LAYERS = 18
```

```python
MESH_SIZE_FACTOR = 0.5
Z_LAYERS = 24
```

Each run uses the same saved fiber distribution. Change `MESH_NAME` if you want
to keep multiple mesh files side by side:

```python
MESH_NAME = "periodic_square_rve_3d_h05"
```

## Output Notes

- `.msh`: Gmsh mesh file for importing into MOOSE/libMesh or other solvers.
- `.vtk`: VTK output for visualization.
- `.gif`: fiber placement/overlap-solve animation.
- `.npz`: saved NumPy distribution data for remeshing.
- `.csv`: human-readable fiber coordinates and radii.

The current 3D examples are written for periodic square RVEs with fibers
extruded along the z direction. The same API can be adapted for other dimensions
or mesh-refinement settings by changing the constants near the top of each
script.
