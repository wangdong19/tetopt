# Simplex Mesh Optimization - Radius Ratio Based Sliver Elimination
This repository contains the implementation of the algorithm proposed in the paper  
**"Global Energy Minimization for Simplex Mesh Optimization: A Radius Ratio Approach to Sliver Elimination"**.  
It provides complete support for reproducing all the numerical examples and results presented in the paper.

## File Structure
├── MeshModel.py # Construct mesh examples
├── opt_example.py # Run mesh optimization examples
├── mesh/
│ ├── mesh_base/ # Base classes for mesh
│ ├── mesh_data_structure/ # Mesh data structure implementation
│ ├── plotting/ # Visualization utilities
│ ├── tetrahedron_mesh.py # Tetrahedral mesh class
│ └── triangle_mesh.py # Triangle mesh class
├── vtkCellTypes.py, vtk_extent.py # Export mesh data to VTU format
├── opt/
│ ├── line_search.py # Wolfe line search implementation
│ ├── odt.py # 3D ODT local mesh smoothing
│ ├── optimizer_base.py # Base class for optimization algorithms
│ ├── PLBFGSAlg.py # PLBFGS optimization algorithm
│ ├── PNLCGAlg.py # PNLCG optimization algorithm
│ ├── TetMehsProblem.py # Tetrahedral radius ratio energy, gradient, and preconditioner
│ └── TriRadiusRatio.py # Triangle radius ratio energy, gradient, and preconditioner


## How to Reproduce Results

### 2D Triangle Mesh Examples

Run the following command:

```bash
python3 opt_example.py --exam <parameter>
```

Available parameter options:

* tri: Triangle domain with radius ratio energy optimization

* sq: Square with a hole with radius ratio energy optimization

* triodt: Triangle domain with ODT smoothing

* sqodt: Square with a hole with ODT smoothing

### 3D Tetrahedral Mesh Examples
```bash
python3 opt_example.py --exam <parameter> --nodt <nodt_param> --optmethod <method> --p <preconditioner>
```

Available exam options:

* sp: Sphere example

* ls: L-shape example with an empty sphere domain inside example

* tsp: 12 intersecting spheres example

Available nodt options:

* 0, 15, 75 (number of ODT smoothing steps)

Available optmethod options:

* LBFGS: Use the LBFGS optimization algorithm

* NLCG: Use the NLCG optimization algorithm

Available p options:

* 0: Without preconditioner

* 1: With preconditioner

## Required Dependencies
The following library versions were used in our experiments:

* Gmsh: 4.12.2

* NumPy: 2.1.2

* Matplotlib: 3.8.4

* SciPy: 1.1.3

* VTK: 9.3.0
