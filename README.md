# Fiber Matrix SVE Generator

![Fiber Matrix Logo](docs/assets/logo.png)

A Python library for generating Representative Volume Elements (RVEs) of fiber-reinforced composites. It handles random fiber placement, periodic boundary conditions, and generates high-quality unstructured meshes using [GMSH](https://gmsh.info/).

## Features

- **Geometric Generation**:
  - Random fiber placement with overlap resolution.
  - Periodic boundary condition enforcement (ghost fibers).
  - Customizable fiber radius, volume fraction, and RVE dimensions.
- **Robust Meshing**:
  - Unstructured meshing via GMSH Python API.
  - Robust boolean operations (Cut/Intersect) handling periodic boundaries.
  - 2D surface meshes and extruded 3D volume meshes.
  - Automatic physical group tagging for Matrix, Fibers, and optional material-specific 3D boundary surfaces.
- **Visualization**:
  - Real-time visualization of fiber packing convergence.
  - Matplotlib plotting of RVE geometry.

## Installation

### Prerequisites

- Python 3.8+
- [GMSH SDK](https://gmsh.info/) (usually installed via pip)

### Install Dependencies

```bash
pip install numpy scipy matplotlib gmsh
```

For building documentation:
```bash
pip install .[docs]
```

## Packaging

To build the library as a `.whl` (wheel) file for distribution:

1. Ensure you have `build` installed:
   ```bash
   pip install build
   ```

2. Run the build command:
   ```bash
   python -m build
   ```

The output `.whl` and `.tar.gz` files will be located in the `dist/` directory.

## Quick Start

### 1. Generate and Mesh an RVE

Here is a minimal example to generate a square SVE with periodic boundaries and create a mesh.

```python
from fiber_matrix.rve import FiberRVE

# 1. Initialize RVE
rve = FiberRVE()
rve.initialize_rectangle_rve(
    num_fibers=20,
    vf=0.45,
    avg_diam=5.0,  # microns
    rve_aspect_ratio=1.0
)

# 2. Place Fibers (Solve for non-overlapping configuration)
print("Solving fiber placement...")
iterations = rve.solve_fiber_locations(
    min_spacing_ratio=0.05, 
    visualize=True  # Save frames of the packing process
)
print(f"Solved in {iterations} iterations.")

# 3. Generate Mesh (Save to .msh file)
print("Generating mesh...")
rve.create_mesh(
    mesh_name="my_composite_rve",
    mesh_size_factor=1.0,
    visualize_gui=False
)
print("Mesh generated: my_composite_rve.msh")
```

### 2. Generate a 3D Volume Mesh

The 2D RVE geometry can also be extruded into a 3D volume mesh. For a cubic
RVE, use the generated square RVE width as the extrusion thickness.

```python
from fiber_matrix.rve import FiberRVE

rve = FiberRVE()
rve.initialize_rectangle_rve(
    num_fibers=20,
    vf=0.45,
    avg_diam=5.0,
    rve_aspect_ratio=1.0
)

iterations = rve.solve_fiber_locations(
    min_spacing_ratio=0.1,
    visualization_path="my_composite_rve_3d.gif"
)
print(f"Solved in {iterations} iterations.")

rve.create_3d_mesh(
    mesh_name="my_composite_rve_3d",
    thickness=rve.rve_dims[0],
    mesh_size_factor=1.0,
    z_layers=16,
    visualize_gui=False,
    check_periodicity=True,
    periodic_z=False,
    surface_groups=True,
    composite_surface_groups=False,
    anchor_node_groups=True,
    uniform_mesh=False,
    fiber_mesh_size=0.25,
    matrix_mesh_size=0.75,
    boundary_mesh_size=0.25,
    interface_refinement_distance=0.75,
    boundary_refinement_distance=0.75,
    recombine_prisms=True,
)
print("3D mesh generated: my_composite_rve_3d.msh")
```

When `surface_groups=True`, the 3D mesh includes volume groups named `Matrix`
and `Fibers`, plus material-specific physical surface groups:

```text
Matrix_Left    Matrix_Right    Matrix_Bottom
Matrix_Top     Matrix_Front    Matrix_Back
Fibers_Left    Fibers_Right    Fibers_Bottom
Fibers_Top     Fibers_Front    Fibers_Back
```

When `composite_surface_groups=True`, the 3D mesh also includes whole-face
surface groups that combine matrix and fiber patches:

```text
composite_left    composite_right    composite_bottom
composite_top     composite_front    composite_back
```

When `anchor_node_groups=True`, the 3D mesh includes three 0D physical groups
for mechanical constraints:

```text
anchor_xyz    left-front-bottom corner
anchor_yz     right-front-bottom point along the x direction
anchor_z      left-back-bottom point in the xy plane
```

For MOOSE/libMesh, avoid overlapping physical groups by keeping
`composite_surface_groups=False` and applying whole-face boundary conditions to
both material-specific surfaces in the input file, for example
`boundary = 'Matrix_Left Fibers_Left'`.

Use `mesh_size_factor` with `uniform_mesh=True` for a single global mesh size.
For non-uniform meshes, set `uniform_mesh=False` and control material/domain
sizes with `fiber_mesh_size`, `matrix_mesh_size`, and `boundary_mesh_size`.
The non-uniform mode adds refinement partitions before meshing:
`fiber_mesh_size` applies near fiber/matrix interfaces, `boundary_mesh_size`
applies near exterior domain boundaries, and `matrix_mesh_size` applies away
from those refinement bands. Use `interface_refinement_distance` and
`boundary_refinement_distance` to control how far the fine mesh extends from
those surfaces. Use `z_layers` to control refinement through the extrusion
direction.
Set `recombine_prisms=True` to keep the structured extrusion as prism/wedge
elements, which avoids the radial tetrahedral subdivision pattern inside each
fiber. Set it to `False` if a downstream solver requires tetrahedral elements.

### 3. Visualize Results

You can visualize the generated geometry using the built-in plotting tools or optional callbacks.

```python
import matplotlib.pyplot as plt
from fiber_matrix.visualization import plotting

fig, ax = plt.subplots()
plotting.draw_rve(rve.fibers, rve.boundaries, ax=ax)
plt.show()
```

## Documentation

Full API documentation is published at:

https://uta-dasp.github.io/Fiber-Matrix-SVE-Generator/

You can also build and view it locally:

```bash
# Build and serve the documentation site
python build_docs.py --serve
```

Then open your browser to `http://127.0.0.1:8000`.

You can also build a static offline version:

```bash
python build_docs.py --offline
```

Then open `site/index.html`, or deploy the generated `site/` directory to
GitHub Pages.

## Project Structure

- `fiber_matrix/`: Main package source.
  - `models/`: Geometric entities (Fiber, Boundary).
  - `generation/`: Packing algorithms.
  - `meshing/`: GMSH interface and boolean logic.
  - `visualization/`: Matplotlib helpers.
- `docs/`: Documentation source files.
- `examples/`: Example scripts.

## License

[Apache 2.0](LICENSE)
