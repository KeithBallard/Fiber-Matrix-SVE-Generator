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
    min_spacing_ratio=1.05, 
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
    composite_surface_groups=True,
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
surface groups that combine matrix and fiber patches for solid mechanics
boundary conditions:

```text
composite_left    composite_right    composite_bottom
composite_top     composite_front    composite_back
```

Use `mesh_size_factor` to control the in-plane element size and `z_layers` to
control refinement through the extrusion direction.

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

Full API documentation is available. You can build and view it locally:

```bash
# Build and serve the documentation site
python build_docs.py --serve
```

Then open your browser to `http://127.0.0.1:8000`.

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
