# CuMesh: High-Performance Geometry Processing for PyTorch

CuMesh is a GPU-accelerated library for high-performance 3D geometry processing in the
PyTorch ecosystem. It provides mesh cleaning, topology repair, simplification, remeshing,
BVH queries, and UV unwrapping without requiring downstream applications to compile their
own CUDA or C++ extensions.

## Installation and support

The published wheel contains the three native modules used by CuMesh:

* `cumesh._C` for CuMesh topology, cleaning, simplification, charting, and remeshing.
* `cumesh._cubvh` for cuBVH and reusable mesh utility operations.
* `cumesh._cumesh_xatlas` for CPU xatlas chart packing and LSCM parameterization.

Installing a published wheel only unpacks these platform-specific binaries. It does not
invoke a compiler, NVCC, Ninja, or `setup.py`.

The supported release matrix is:

| Component | Supported configuration |
| --- | --- |
| Python | CPython 3.10, 3.11, and 3.12 |
| Platforms | Linux x86-64 and Windows x86-64 |
| PyTorch | 2.5.1 with the CUDA 12.4 distribution |
| GPU architectures | Set at build time with `TORCH_CUDA_ARCH_LIST` |

The source retains the existing HIP/ROCm switch, but that path is not covered by this
repository's tests. macOS is not supported because the CuMesh and cuBVH extensions require
CUDA or HIP.

### Runtime installation

Install a matching CUDA-enabled PyTorch release, then install the CuMesh wheel and its
runtime dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy scipy tqdm
python -m pip install cumesh-0.9.0-<platform>.whl
```

The compiled extension is linked against the PyTorch C++ ABI selected by the build's
PyTorch installation. Consumers should use the same PyTorch release and a compatible CUDA
runtime.

## Building a wheel from source

Source builds are intended for package maintainers and require a CUDA-enabled PyTorch
installation, CUDA Toolkit 12.4, a C++ compiler, and Eigen 3 headers. If Eigen is not
available in the selected PyTorch include tree, set `EIGEN3_INCLUDE_DIR` to the directory
containing the `Eigen/` folder.

Clone the repository without submodule initialization; the cuBVH sources are vendored in the
normal `third_party/cubvh` project directory:

```bash
git clone https://github.com/GENERIO-ai/CuMesh.git
cd CuMesh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy scipy tqdm setuptools wheel
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

On PowerShell, set the architecture list with:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.6;8.9;9.0"
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Build once for each target platform and CUDA architecture set. The `--no-build-isolation`
flag is intentional: it makes the explicitly installed PyTorch version the one used for the
extension ABI. Standard PEP 517 isolation remains possible, but it uses the exact build
requirements declared in `pyproject.toml`.

## Clean-environment validation

On a clean CUDA-capable machine with the documented PyTorch runtime:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy scipy tqdm
python -m pip install --no-deps dist/cumesh-0.9.0-*.whl
python -c "import cumesh; import cumesh._C, cumesh._cubvh, cumesh._cumesh_xatlas; print(cumesh.__version__)"
python -m unittest discover -s tests -v
```

The `--no-deps` install is useful for proving that wheel installation itself does not compile
anything. Use a normal install when the runtime dependencies are not already present. The
native extension import and geometry tests must run on a CUDA-capable host.

## API overview

### `cumesh.CuMesh`

* `init(vertices, faces)`: initialize a mesh with `[V, 3]` and `[F, 3]` CUDA tensors.
* `read()`: return the current vertex and face tensors.
* `simplify(target_num_faces, verbose=False, options={})`: GPU mesh decimation.
* `uv_unwrap(verbose=False, ...)`: generate UVs using accelerated clustering and xatlas.
* `merge_micro_charts(...)`: merge small adjacent UV charts.
* `fill_holes(max_hole_perimeter)`: triangulate and close boundary loops.
* `repair_non_manifold_edges()`: split edges to resolve non-manifold geometry.
* `remove_degenerate_faces()`: remove zero-area or invalid faces.
* `remove_duplicate_faces()`: remove faces with identical vertex indices.
* `remove_small_connected_components(min_area)`: delete isolated small components.
* `unify_face_orientations()`: reorient faces to consistent winding.
* `compute_face_normals()`, `compute_vertex_normals()`: calculate normals.
* `get_connected_components()`, `get_boundary_loops()`: query mesh connectivity.
* Properties: `num_vertices`, `num_faces`, `num_edges`, and `num_boundaries`.

`uv_unwrap(..., preserve_cumesh_charts=True)` keeps CuMesh chart boundaries while using
native LSCM parameterization and xatlas packing. `return_stats=True` is additive and returns
chart-cleanup statistics; existing call signatures and return formats remain available by
default.

### `cumesh.remeshing`

`remesh_narrow_band_dc(...)` performs Dual Contouring reconstruction based on the unsigned
distance field of the input mesh.

### `cumesh.cuBVH` and mesh utilities

`cumesh.bvh.cuBVH` provides `ray_trace`, `unsigned_distance`, and `signed_distance`. The same
module exposes the integrated cuBVH hash-table, sparse marching-cubes, hole-filling,
vertex-merging, and CPU decimation wrappers, including `cuHashTable`, `HashTable`,
`floodfill`, `sparse_marching_cubes`, `fill_holes`, `merge_vertices`, and `decimate`.

### `cumesh.Atlas`

`Atlas` wraps xatlas and accepts CPU tensors for mesh operations:

* `add_mesh(vertices, faces, normals=None, uvs=None)`: register mesh geometry.
* `compute_charts(max_chart_area, ...)`: segment geometry into UV charts.
* `pack_charts(resolution, padding, ...)`: pack charts into a texture atlas.
* `add_uv_mesh(uvs, faces, face_materials=None)`: register an already parameterized mesh.
* `get_mesh(index)`: retrieve vertex map, faces, and UVs for a packed mesh.

`cumesh.xatlas.parameterize_lscm(vertices, faces)` is the native CPU LSCM parameterizer.

## Extended UV unwrapping capabilities

CuMesh's UV pipeline has been extended for workflows that need stable, reusable chart
boundaries in addition to standard xatlas unwrapping. The extended path can:

* clean up micro-charts and split disconnected or filament-like chart regions;
* preserve CuMesh chart boundaries during UV generation;
* parameterize charts with the native CPU LSCM implementation;
* pack already-parameterized charts through `Atlas.add_uv_mesh`; and
* return optional chart-cleanup statistics without changing the default return format.

The functionality is implemented in CuMesh's public modules and native extensions:

| Capability | CuMesh destination | Notes |
| --- | --- | --- |
| Chart cleanup and preserved-chart UVs | `cumesh/cumesh.py`, `src/` | Existing APIs are retained; chart splitting, micro-chart merging, preserved-chart UVs, and optional stats are additive. |
| Atlas and LSCM | `cumesh/xatlas.py`, `third_party/xatlas/` | Adds `add_uv_mesh` and `parameterize_lscm` without changing existing Atlas APIs. |
| cuBVH and reusable mesh utilities | `cumesh/bvh.py`, `third_party/cubvh/` | Packaged as `cumesh._cubvh`; CUDA-only distance and ray operations remain CUDA-only. |
| Remeshing support | `cumesh/remeshing.py`, `src/remesh/` | Existing CuMesh kernels and the packaged BVH backend are reused. |

Application-specific orchestration, such as mesh extraction, trimesh conversion, normal
transfer, option policy, and CUDA-memory cleanup, remains outside the core CuMesh library.

The native extension boundaries are:

| Extension | Role |
| --- | --- |
| `cumesh._C` | CuMesh topology, cleaning, simplification, charting, chart cleanup, and remeshing kernels. |
| `cumesh._cubvh` | cuBVH and reusable BVH/mesh utility bindings. |
| `cumesh._cumesh_xatlas` | CPU xatlas chart packing, UV-mesh input, and LSCM. |

Application-specific code is intentionally kept outside this package; only reusable library
functionality belongs in CuMesh.

## Acknowledgements

CuMesh builds upon and integrates code from several open-source projects:

* [cubvh](https://github.com/ashawkey/cubvh) for high-performance CUDA BVH acceleration.
* [xatlas](https://github.com/jpcy/xatlas) for UV parameterization and atlas packing.
* [pamo](https://github.com/SarahWeiii/pamo) for the reference GPU parallel edge-collapse implementation.

## License

[MIT License](LICENSE)
