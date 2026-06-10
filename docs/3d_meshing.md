# 3D RVE Meshing

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
