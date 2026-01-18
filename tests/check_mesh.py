import gmsh
import sys
import os


def check_mesh(mesh_file):
    gmsh.initialize()
    try:
        gmsh.open(mesh_file)

        # Get all physical groups
        physical_groups = gmsh.model.getPhysicalGroups()

        found_matrix = False
        found_fibers = False

        print("Physical Groups Found:")
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            print(f"  Dim: {dim}, Tag: {tag}, Name: '{name}'")

            # Check element count
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            total_elements = 0
            for e in entities:
                types, elementTags, nodeTags = gmsh.model.mesh.getElements(dim, e)
                for t in elementTags:
                    total_elements += len(t)

            print(f"    -> Contains {total_elements} elements.")

            if name == "Matrix" and total_elements > 0:
                found_matrix = True
            if name == "Fibers" and total_elements > 0:
                found_fibers = True

        if found_matrix and found_fibers:
            print(
                "\nSUCCESS: Both Matrix and Fibers physical groups found with elements."
            )
        else:
            print("\nFAILURE: Missing Matrix or Fibers physical groups.")
            if not found_matrix:
                print("  - Matrix group not found or empty.")
            if not found_fibers:
                print("  - Fibers group not found or empty.")
            sys.exit(1)

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        # Default relative to tests/ location
        mesh_file = os.path.join(os.path.dirname(__file__), "test_mesh.msh")

    check_mesh(mesh_file)
