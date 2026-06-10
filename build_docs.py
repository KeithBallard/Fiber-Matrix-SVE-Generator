import os
import subprocess
import sys


def main():
    print("Initializing Documentation Structure...")

    # 1. Create docs directory
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # 2. Create docs/index.md
    index_content = """# Fiber Matrix SVE Generator

Welcome to the documentation for the `fiber_matrix` package.

## Introduction

This package generates Representative Volume Elements (RVEs) for fiber-matrix composites. It handles:
- **Generation**: Random or specified placement of fibers.
- **Microstructure**: Periodic boundary handling and overlap resolution.
- **Meshing**: High-quality mesh generation using GMSH with robust Set Operation logic.
- **Visualization**: Visualization of the packing process.

## Getting Started

Check out the [API Reference](api/rve.md) to see how to use the `FiberRVE` class.
For 3D workflows, see [3D RVE Meshing](3d_meshing.md).
"""
    with open("docs/index.md", "w") as f:
        f.write(index_content)

    # 3. Create docs/api directory
    if not os.path.exists("docs/api"):
        os.makedirs("docs/api")

    # 4. Create API Documentation Pages using mkdocstrings
    api_pages = {
        "rve.md": "::: fiber_matrix.rve.FiberRVE\n    options:\n      show_root_heading: true\n      show_source: false",
        "models.md": "# Models\n\n## Boundary\n::: fiber_matrix.models.boundary\n\n## Fiber\n::: fiber_matrix.models.fiber",
        "generation.md": "::: fiber_matrix.generation.placement",
        "meshing.md": "# 2D Meshing\n\n::: fiber_matrix.meshing.gmsh_mesher",
        "meshing_3d.md": "# 3D Meshing\n\n::: fiber_matrix.meshing.gmsh_mesher_3d",
        "visualization.md": "::: fiber_matrix.visualization.plotting",
    }

    for filename, content in api_pages.items():
        with open(os.path.join("docs/api", filename), "w") as f:
            f.write(content)

    # 5. Create workflow documentation pages
    three_d_content = """# 3D RVE Meshing

The 3D workflow extrudes a periodic 2D fiber/matrix RVE into a volume mesh.
The main entry point is `FiberRVE.create_3d_mesh`.

## Basic 3D Mesh

```python
rve.create_3d_mesh(
    mesh_name="periodic_square_rve_3d",
    thickness=rve.rve_dims[0],
    mesh_size_factor=0.75,
    z_layers=24,
    check_periodicity=True,
    surface_groups=True,
    composite_surface_groups=False,
    anchor_node_groups=True,
    uniform_mesh=False,
    fiber_mesh_size=0.25,
    matrix_mesh_size=1.5,
    boundary_mesh_size=0.5,
    interface_refinement_distance=0.5,
    boundary_refinement_distance=0.5,
    recombine_prisms=True,
)
```

## Physical Groups

The 3D meshes always include volume groups named `Matrix` and `Fibers`.

When `surface_groups=True`, material-specific side groups are created:

```text
Matrix_Left    Matrix_Right    Matrix_Bottom
Matrix_Top     Matrix_Front    Matrix_Back
Fibers_Left    Fibers_Right    Fibers_Bottom
Fibers_Top     Fibers_Front    Fibers_Back
```

When `anchor_node_groups=True`, three 0D physical groups are created for
mechanical constraints:

```text
anchor_xyz    left-front-bottom corner
anchor_yz     right-front-bottom point along the x direction
anchor_z      left-back-bottom point in the xy plane
```

For MOOSE/libMesh, avoid assigning one geometric entity to multiple boundary
IDs. Use `surface_groups=True` and `composite_surface_groups=False`, then combine
material-specific groups in the input file when a whole-composite face is needed.

## Non-Uniform Mesh Controls

Set `uniform_mesh=False` to enable separate sizing controls:

- `fiber_mesh_size`: fine size near fiber/matrix interfaces.
- `matrix_mesh_size`: coarse size away from refinement bands.
- `boundary_mesh_size`: fine size near exterior domain boundaries.
- `interface_refinement_distance`: width of the interface refinement band.
- `boundary_refinement_distance`: width of the exterior boundary refinement band.

The non-uniform mode adds refinement partitions before meshing, which gives Gmsh
real geometric locations where the mesh can transition from fine to coarse while
preserving periodic matching.

## Fixed Distribution Workflow

For mesh convergence studies, use:

```bash
python examples/periodic_square_rve_3d_save_distribution.py
python examples/periodic_square_rve_3d_mesh_from_distribution.py
```

The first script saves fiber centers and radii. The second script reloads that
same distribution and generates meshes with different sizing parameters.
"""

    offline_content = """# Offline And GitHub Pages Documentation

`build_docs.py` supports both a live preview server and a static offline build.

## Live Preview

```bash
python build_docs.py --serve
```

Open `http://127.0.0.1:8000` in a browser.

## Static Offline Build

```bash
python build_docs.py --offline
```

This creates a static site in the `site/` directory. The generated HTML uses
relative links, so it can be opened locally from:

```text
site/index.html
```

The same `site/` directory is also deployable to GitHub Pages.

## GitHub Pages Deployment

One simple deployment path is:

```bash
python build_docs.py --offline
python -m mkdocs gh-deploy
```

Alternatively, upload the generated `site/` directory as a GitHub Pages artifact
or publish it from a deployment branch.
"""

    workflow_pages = {
        "3d_meshing.md": three_d_content,
        "offline.md": offline_content,
    }

    for filename, content in workflow_pages.items():
        with open(os.path.join("docs", filename), "w") as f:
            f.write(content)

    # 6. Create mkdocs.yml
    mkdocs_yaml = """site_name: Fiber Matrix SVE
use_directory_urls: false
theme:
  name: material
  logo: assets/logo.png
  features:
    - navigation.sections
    - navigation.indexes

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          options:
            docstring_style: numpy
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - 3D RVE Meshing: 3d_meshing.md
  - API Reference:
    - RVE Interface: api/rve.md
    - Models: api/models.md
    - Generation: api/generation.md
    - 2D Meshing: api/meshing.md
    - 3D Meshing: api/meshing_3d.md
    - Visualization: api/visualization.md
  - Offline / GitHub Pages: offline.md
"""

    with open("mkdocs.yml", "w") as f:
        f.write(mkdocs_yaml)

    print("Configuration and content generated.")

    # 7. Run MkDocs
    # Ensure current directory is in PYTHONPATH so mkdocstrings can find the package
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.dirname(os.path.abspath(__file__))
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    # Check for --serve argument
    if "--serve" in sys.argv:
        print("Starting MkDocs Live Preview...")
        try:
            # interactive process, let it take over stdout/stderr
            subprocess.call([sys.executable, "-m", "mkdocs", "serve"], env=env)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    if "--offline" in sys.argv:
        print("Building offline documentation site...")
    else:
        print("Building documentation site...")

    try:
        # Use python -m mkdocs to ensure we use the module in the current venv
        subprocess.check_call([sys.executable, "-m", "mkdocs", "build"], env=env)
        print("\\nDocumentation build successful!")
        print("Static site generated in 'site/' directory.")
        print("Offline entry point: site/index.html")
        print("To preview, run: python build_docs.py --serve")
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
