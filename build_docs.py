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
        "meshing.md": "::: fiber_matrix.meshing.gmsh_mesher",
        "visualization.md": "::: fiber_matrix.visualization.plotting",
    }

    for filename, content in api_pages.items():
        with open(os.path.join("docs/api", filename), "w") as f:
            f.write(content)

    # 5. Create mkdocs.yml
    mkdocs_yaml = """site_name: Fiber Matrix SVE
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
  - API Reference:
    - RVE Interface: api/rve.md
    - Models: api/models.md
    - Generation: api/generation.md
    - Meshing: api/meshing.md
    - Visualization: api/visualization.md
"""

    with open("mkdocs.yml", "w") as f:
        f.write(mkdocs_yaml)

    print("Configuration and content generated.")

    # 6. Run MkDocs
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

    print("Building documentation site...")

    try:
        # Use python -m mkdocs to ensure we use the module in the current venv
        subprocess.check_call([sys.executable, "-m", "mkdocs", "build"], env=env)
        print("\\nDocumentation build successful!")
        print("Static site generated in 'site/' directory.")
        print("To preview, run: python build_docs.py --serve")
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
