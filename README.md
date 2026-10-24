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
python -m pip install cumesh-0.9.1-<platform>.whl
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
python -m pip install --no-deps dist/cumesh-0.9.1-*.whl
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
* `normalize(min_area_abs=1e-24, min_area_rel=1e-12, iterations=1, verbose=False)`: split non-manifold edges once and collapse small interior triangles on the GPU.
* `uv_unwrap(verbose=False, debug_charts=False, flatten=["project"], min_island_triangles=16, fix_overlaps=True, ...)`: generate packed UVs from CuMesh charts; optionally return a chart-colored debug texture.
* `compute_charts(..., chart_merge_target=32, chart_merge_strength=0.8)`: generate GPU charts with a size-dependent merge-cost discount.
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

GPU chart merging discounts the existing full merge cost according to the
smaller chart's current triangle count `n`:

```text
fade = clamp((chart_merge_target - n) / (chart_merge_target - 1), 0, 1)
adjusted_cost = original_cost * (1 - chart_merge_strength * fade)
```

The defaults are `chart_merge_target=32` and `chart_merge_strength=0.8`.
At one triangle the cost is multiplied by 0.2; at 32 or more it is unchanged. Strength
must be finite and in `[0, 1]`, and the target must be an integer of at least
2. Strength 0 restores the original costs; strength 1 makes singleton-involving
merges cost zero. Eligibility still requires a shared indexed manifold edge
and the existing mutual-neighbor selection. The discount applies on every
merge round from the start, using refreshed GPU counts for both selection
and threshold acceptance. Counts are retained from the existing chart-normal
calculation; no mesh download or separate counting pass is added.

Pass the options through `uv_unwrap(compute_charts_kwargs={
"chart_merge_target": 32, "chart_merge_strength": 0.8})`.
`examples/uv_unwrap.py` exposes both explicitly. Verbose chart-generation logs
and returned stats include the selected values. Larger charts can still
produce projection overlaps; this setting does not enforce a minimum island
size or change the overlap-splitting toggle.

After rebuilding, `tests/test_chart_merging.py` covers configuration validation
and GPU merges, including refreshed counts and shared-edge eligibility.
`tests/chart_merge_test.cpp` is a standalone host test of the multiplier used
by the CUDA kernel, covering the linear fade, endpoints, and symmetry.

`uv_unwrap(...)` normalizes the mesh, computes CuMesh charts, and tries the requested
flattening stages in order. Supported methods are `"lscm"`, `"project"`, `"tutte"`, and
`"pca"`; for example, `flatten=["lscm", "tutte"], min_island_triangles=0` uses
Tutte only for charts where LSCM fails, with small-island merging disabled.
The default is `flatten=["project"]`. If all requested stages fail, a
last-resort planar UV is still returned. The parameterized charts are added to xatlas and
packed with four-pixel padding by default. The method always returns
`(vertices, faces, uvs, chart_texture)`, with `chart_texture=None` unless `debug_charts=True`.
When enabled, `chart_texture` is a CPU `uint8` RGBA tensor containing one deterministic color
per chart. With `verbose=True`, the final log includes disjoint counts for
charts handled by native LSCM, the native graph fallback, and the Python PCA fallback.
`return_vmaps=True` and `return_stats=True` remain additive; the latter includes the same
method counts under `final_parameterization_methods` plus native fallback reason counters.
The `"project"` stage is reported as `python_project` in those method counts.
Problematic charts, dropped faces, missing mappings, packing fallbacks, and failed final
validation are reported as console warnings with affected counts and percentages. The
returned stats include `atlas_complete`, `dropped_charts`, `dropped_faces`,
`unmapped_vertices`, `self_overlapping_charts`, `self_overlap_pairs`,
`overlap_split_charts`, `distortion_optimized_charts`, and `uv_warning_level`.
After flattening and optional small-island repair, a small native
symmetric-Dirichlet pass optimizes charts with a four-step budget, moving boundaries while
rejecting flipped or degenerate triangles. Charts already close to the energy
minimum are skipped. The single `fix_overlaps` step then untangles local folds
and partitions unresolved overlaps before packing. Final validation checks the
packed UV result.
The Project stage computes a direction-independent median of the chart triangle normals,
aligns the chart to that plane, and projects all chart vertices onto it. The Tutte stage
projects each chart boundary to a local 2D plane, uses its convex hull as the
fixed Tutte polygon, and samples that polygon in the original boundary order using physical
edge lengths. The old Tutte-only nonlinear pass remains disabled with
`kEnableTutteDistortionOptimizer` in `third_party/xatlas/safe_uv.h` because the
shared post-flatten pass supersedes it.
The stage helpers are kept separately in `cumesh/flatten_lscm.py`,
`cumesh/flatten_project.py`, `cumesh/flatten_tutte.py`, and
`cumesh/flatten_pca.py`.

`uv_unwrap(...)` targets a minimum UV island size of 16 triangles by default;
set `min_island_triangles` to change the target. After initial flattening and **before overlap fixing**,
the repair pass builds a permissive surface-adjacency graph and merges the
smallest face groups toward the target. It reparameterizes each affected
group once with the selected `flatten` method, accepting overlaps at this
stage. Distortion optimization runs next, followed by one optional overlap-fixing
step and packing. No UV optimizer runs after overlap fixing.

`fix_overlaps=True` (the default, also enabled in `examples/uv_unwrap.py`)
first detects overlapping triangle pairs with a native BVH. Triangles involved
in conflicts are visited in a deterministic order, favoring free corners,
small vertex fans, and fewer conflicts. A triangle can keep a shared anchor
edge and move its opposite UV vertex: reflection across the edge is tried
first, then a shorter reflected altitude. All aliases of the UV vertex move
together, preserving existing shared edges and seams. Geometry and source
face-corner mappings do not change.

Trials use the actual float32 coordinates and check every affected triangle,
including against distant parts of the chart. A move must clear all conflicts
of its candidate triangle, preserve finite, nondegenerate triangles, and create
no new overlap pair anywhere in the affected fan. Rejected trials restore the
original coordinates. Stretching and changes in projected winding are allowed.
The BVH is refitted only on accepted moves, and the conflict graph only loses
edges. Work is bounded by six positions per visit, at most two visits per
triangle (a revisit requires a nearby successful edit), a maximum fan of 12
faces, and a per-chart query-work budget. Budget exhaustion rejects an
incompletely checked trial and leaves its conflicts for partitioning.

For remaining conflicts, the splitter grows connected surface patches simultaneously. It seeds
two patches from an overlapping pair and prioritizes patches below
`min_island_triangles` when assigning adjacent triangles. A triangle joins a
patch only if it overlaps none of that patch's triangles. New patches are
seeded only when the existing frontiers cannot grow. Growth favors triangles
with few remaining compatible neighbors and defers consuming a last connector
when that would strand a neighbor blocked from the receiving patch.

A single boundary-balancing pass can transfer leaf triangles from larger
patches into undersized neighbors. Shared-edge attachment and a maintained
spanning tree preserve both patches' connectivity; conflict counts reject
overlapping moves, and donors stay at or above the target. Each triangle can
move between patches at most once. Partitioning keeps the repaired UV
coordinates and returns one connected island per group.
The existing dense-overlap safety limit remains and is reported if reached.

`fix_overlaps=False` skips both local untangling and overlap partitioning for
timing comparisons. Distortion optimization and final UV validation still run.
The old `split_overlaps` argument has been replaced by `fix_overlaps`.
Returned `fix_*` stats include initial/residual conflict pairs, repaired
triangles, moved UV variables, trial counts, fan/budget skips, singleton and
undersized-island counts, boundary transfers, and detection/repair/partition
timings. `fix_blocked_singletons` counts singleton outputs with no remaining
non-overlapping shared-edge neighbor in this layout; other singletons can
reflect the bounded repair and growth heuristics. The minimum remains a target,
not a guarantee, and performance and island-count improvements depend on the mesh.

Shared edges may connect non-manifold, coincident, or overlapping coplanar
faces. Endpoint matching allows a tolerance of `1e-6` times the geometry's
bounding-box diagonal; both endpoints must match, so point contact alone is
insufficient. Matching only welds the temporary parameterization mesh.
Original positions and source face-corner mappings are preserved, including
distinct source IDs at matched endpoints.

Repair requires exactly one selected method: `flatten=["project"]`,
`flatten=["pca"]`, or `flatten=["lscm"]`. It does not invoke Tutte, try alternate
flatteners, or fit the old UV layouts. The default target is 16 triangles;
values 0 and 1 disable repair. The implementation uses the selected flattening, overlap-fixing,
and optimization APIs.

The minimum is a target, not a condition for returning UVs. Components without
an edge neighbor keep their existing UVs. If re-flattening a group fails,
its original islands are retained and other groups still proceed. There is
no retry with another parameterizer. Distortion optimization preserves shared
UV variables before the single overlap-fixing step. Splitting unresolved conflicts can create
islands below the target again; the final counts report this rather than
aborting the unwrap. Final UV validation continues to report warnings.

With `return_stats=True`, `repair_islands_before/after`,
`repair_undersized_before/after`, `repair_merges`,
`repair_reparameterized_groups`, `repair_failed_groups`,
`repair_unmergeable_islands`, `repair_non_manifold_edges`,
`repair_position_tolerance`, and `repair_seconds` describe the pass before
overlap fixing. Non-manifold edges are included rather than rejected.
`repair_failed` reports a failure of the grouping pass as a whole; in that
case the original charts proceed to optimization, overlap fixing, and packing.
`final_island_count` and `final_undersized_islands` measure the packed output.
Counts are `None` when repair is disabled. `verbose=True` prints repair timing
and final counts.

Overlap-fixing statistics are available independently of whether small-island
repair is enabled: `fix_islands_before/after` and `fix_undersized_before/after`
count connected surface islands around the single fix. `fix_transferred_faces`
counts boundary moves, `fix_seconds` measures the pass, and `fix_enabled`
records the setting. Island counts are `None` when fixing is disabled or a
check fails. These intermediate counts follow optimization and precede packing;
`final_island_count` still measures the packed output.

After rebuilding and installing the extensions, the UV regression tests can
be run with `python -m unittest discover -s tests -p 'test_uv_*.py' -v`.
`tests/uv_split_test.cpp` also exercises boundary balancing, conflict rejection,
connectivity, and the dense safeguard directly as a standalone C++17 test.
`tests/uv_fix_test.cpp` covers untangling, rollback, alias preservation, distant
obstructions, work budgets, and final conflict separation without Python.

For verbose phase timing, pass `verbose=True`. To trace each native flattening chart and
solver phase, set `CUMESH_UV_TRACE=1` before running the application.

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

## UV unwrapping pipeline

CuMesh's UV pipeline uses the following single path:

* use the CuMesh chart boundaries directly;
* return a chart-colored debug texture after chart parameterization and packing;
* parameterize charts with the native CPU LSCM implementation;
* pack already-parameterized charts through `Atlas.add_uv_mesh`; and
* return optional parameterization and packing statistics.

The functionality is implemented in CuMesh's public modules and native extensions:

| Capability | CuMesh destination | Notes |
| --- | --- | --- |
| CuMesh chart UVs | `cumesh/cumesh.py`, `src/` | CuMesh chart boundaries are parameterized directly and passed to xatlas as UV meshes. |
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
