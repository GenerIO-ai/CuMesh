from typing import *
import math
import os
import warnings
import torch
import numpy as np
from time import perf_counter
from tqdm import tqdm
from .xatlas import Atlas
from . import _C


def _validate_uv_output(atlas, uvs, faces):
    """Validate actual output coordinates; winding is never a quality signal."""
    result = atlas._validate_uv(uvs.contiguous(), faces.contiguous())
    coords = uvs.numpy()
    result["out_of_range_values"] = int(np.count_nonzero((coords < 0.0) | (coords > 1.0)))
    if result["valid"] and result["out_of_range_values"]:
        result["valid"] = False
        result["issue"] = "out_of_range"
    return result


def _project_chart_uvs(vertices):
    """Create a winding-independent planar fallback using PCA."""
    points = np.asarray(vertices, dtype=np.float64)
    centered = points - np.mean(points, axis=0, keepdims=True)
    if len(points) >= 3:
        try:
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            if basis.shape[0] >= 2:
                projected = centered @ basis[:2].T
                if np.isfinite(projected).all() and np.ptp(projected, axis=0).max() > 0:
                    return projected
        except np.linalg.LinAlgError:
            pass

    # This is only an emergency path for malformed/zero-span input. It does
    # not infer or enforce a face winding.
    indices = np.arange(len(points), dtype=np.float64)
    return np.column_stack((indices, (indices % 2) * 1e-3))


def _repair_collapsed_uv_faces(uvs, faces, vmaps=None, threshold=2e-18):
    """Repair numerically collapsed triangles without using face winding.

    A repaired corner receives a duplicate UV vertex. This preserves the
    source-vertex mapping while allowing the problematic face to be made
    non-zero without perturbing its neighbours. The operation is normally a
    no-op and is linear in the number of faces.
    """
    uv_np = np.ascontiguousarray(np.asarray(uvs, dtype=np.float32))
    face_np = np.ascontiguousarray(np.asarray(faces, dtype=np.int32)).copy()
    vmap_np = None if vmaps is None else np.ascontiguousarray(np.asarray(vmaps, dtype=np.int32))
    if uv_np.ndim != 2 or uv_np.shape[1] != 2 or face_np.ndim != 2 or face_np.shape[1] != 3:
        return uv_np, face_np, vmap_np, 0
    if len(face_np) == 0:
        return uv_np, face_np, vmap_np, 0
    if np.any(face_np < 0) or np.any(face_np >= len(uv_np)):
        return uv_np, face_np, vmap_np, 0

    p0 = uv_np[face_np[:, 0]].astype(np.float64)
    p1 = uv_np[face_np[:, 1]].astype(np.float64)
    p2 = uv_np[face_np[:, 2]].astype(np.float64)
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
        (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    bad = ~np.isfinite(cross) | (np.abs(cross) <= threshold)
    bad_faces = np.flatnonzero(bad)
    if len(bad_faces) == 0:
        return uv_np, face_np, vmap_np, 0

    extra_uvs = []
    extra_vmaps = []
    repaired = 0
    for face_index in bad_faces:
        tri = face_np[face_index].copy()
        points = uv_np[tri].astype(np.float64)
        if not np.isfinite(points).all():
            points = np.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)

        edge_pairs = ((0, 1), (1, 2), (2, 0))
        lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in edge_pairs]
        pair_index = int(np.argmax(lengths))
        a, b = edge_pairs[pair_index]
        edge = points[b] - points[a]
        edge_length = lengths[pair_index]
        delta = max(1e-6, 1e-4 * edge_length)

        if edge_length > 0.0 and np.isfinite(edge_length):
            # The sign is arbitrary; it is not used as a winding signal.
            perpendicular = np.array([-edge[1], edge[0]], dtype=np.float64) / edge_length
            replacement = 0.5 * (points[a] + points[b]) + perpendicular * delta
            corner = 3 - a - b
            extra_uvs.append(replacement.astype(np.float32))
            if vmap_np is not None:
                extra_vmaps.append(vmap_np[tri[corner]])
            face_np[face_index, corner] = len(uv_np) + len(extra_uvs) - 1
        else:
            # All three UVs coincide. Give two private corners a tiny
            # non-collinear offset.
            center = np.mean(points, axis=0)
            first = center + np.array([delta, 0.0])
            second = center + np.array([0.0, delta])
            extra_uvs.extend((first.astype(np.float32), second.astype(np.float32)))
            if vmap_np is not None:
                extra_vmaps.extend((vmap_np[tri[1]], vmap_np[tri[2]]))
            face_np[face_index, 1] = len(uv_np) + len(extra_uvs) - 2
            face_np[face_index, 2] = len(uv_np) + len(extra_uvs) - 1
        repaired += 1

    if extra_uvs:
        uv_np = np.concatenate((uv_np, np.asarray(extra_uvs, dtype=np.float32)), axis=0)
        if vmap_np is not None:
            vmap_np = np.concatenate((vmap_np, np.asarray(extra_vmaps, dtype=np.int32)))
    return uv_np, face_np, vmap_np, repaired


def _prepare_uv_chart(vertices, faces, uvs, vmaps, area_scale=None):
    """Condition chart UVs for float32/xatlas and repair zero-area faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    vmap_np = np.ascontiguousarray(np.asarray(vmaps, dtype=np.int32))
    uv_np = np.asarray(uvs, dtype=np.float64)
    projected = False
    if uv_np.ndim != 2 or uv_np.shape != (len(vmap_np), 2) or not np.isfinite(uv_np).all():
        uv_np = _project_chart_uvs(vertices)
        projected = True

    if area_scale is not None and np.isfinite(area_scale) and area_scale > 0:
        uv_np *= float(area_scale)

    lower = np.min(uv_np, axis=0)
    extent = float(np.ptp(uv_np, axis=0).max())
    if not np.isfinite(lower).all() or not np.isfinite(extent) or extent <= 0:
        uv_np = _project_chart_uvs(vertices)
        lower = np.min(uv_np, axis=0)
        projected = True
    # Translation preserves both chart geometry and the relative chart area
    # used by xatlas. Avoiding a second per-chart scale is important: it keeps
    # the existing texel-density behaviour while still removing large UV
    # offsets before the float32 conversion.
    uv_np = uv_np - lower
    uv_np = np.ascontiguousarray(uv_np.astype(np.float32))
    uv_np, faces_np, vmap_np, repaired = _repair_collapsed_uv_faces(
        uv_np, faces_np, vmap_np
    )
    return uv_np, faces_np, vmap_np, repaired, projected


def _chart_texture_color(chart_index):
    """Return a deterministic, well-separated RGBA color for one chart."""
    hue = (0.07 + chart_index * 0.618033988749895) % 1.0
    h = hue * 6.0
    sector = min(5, int(h))
    fraction = h - int(h)
    q = 1.0 - fraction
    t = fraction
    rgb = (
        (1.0, t, 0.0),
        (q, 1.0, 0.0),
        (0.0, 1.0, t),
        (0.0, q, 1.0),
        (t, 0.0, 1.0),
        (1.0, 0.0, q),
    )[sector]
    return np.asarray(
        [round(channel * 255.0) for channel in rgb] + [255],
        dtype=np.uint8,
    )


def _rasterize_chart_texture(uvs, faces, face_chart_ids, width, height):
    """Rasterize packed chart IDs into a small CPU RGBA debug texture.

    The texture uses the same normalized UV domain returned by ``Atlas``.
    Pixels not covered by a packed triangle receive an opaque dark background.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    texture = np.empty((height, width, 4), dtype=np.uint8)
    texture[:, :] = np.array([24, 24, 24, 255], dtype=np.uint8)
    chart_colors = {}

    uv_np = np.asarray(uvs, dtype=np.float64)
    face_np = np.asarray(faces, dtype=np.int64)
    chart_np = np.asarray(face_chart_ids, dtype=np.int64)
    if (
        uv_np.ndim != 2
        or uv_np.shape[1] != 2
        or face_np.ndim != 2
        or face_np.shape[1] != 3
        or len(face_np) != len(chart_np)
    ):
        return torch.from_numpy(texture)

    for triangle, chart_index in zip(face_np, chart_np):
        if np.any(triangle < 0) or np.any(triangle >= len(uv_np)):
            continue
        triangle_uv = uv_np[triangle]
        if not np.isfinite(triangle_uv).all():
            continue

        # Trimesh uses a bottom-left UV origin, while image rows start at the
        # top. Convert V into image-row coordinates before rasterizing.
        x = triangle_uv[:, 0] * width
        y = (1.0 - triangle_uv[:, 1]) * height
        min_x = max(0, int(np.floor(np.min(x))))
        max_x = min(width - 1, int(np.ceil(np.max(x)) - 1))
        min_y = max(0, int(np.floor(np.min(y))))
        max_y = min(height - 1, int(np.ceil(np.max(y)) - 1))
        if min_x > max_x or min_y > max_y:
            continue

        pixel_x, pixel_y = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
            np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
        )
        denominator = (
            (y[1] - y[2]) * (x[0] - x[2])
            + (x[2] - x[1]) * (y[0] - y[2])
        )
        if abs(denominator) <= 1e-12:
            continue

        weight_0 = (
            (y[1] - y[2]) * (pixel_x - x[2])
            + (x[2] - x[1]) * (pixel_y - y[2])
        ) / denominator
        weight_1 = (
            (y[2] - y[0]) * (pixel_x - x[2])
            + (x[0] - x[2]) * (pixel_y - y[2])
        ) / denominator
        weight_2 = 1.0 - weight_0 - weight_1
        covered = (weight_0 >= -1e-6) & (weight_1 >= -1e-6) & (weight_2 >= -1e-6)
        if np.any(covered):
            chart_index = int(chart_index)
            if chart_index not in chart_colors:
                chart_colors[chart_index] = _chart_texture_color(chart_index)
            color = chart_colors[chart_index]
            texture[min_y:max_y + 1, min_x:max_x + 1][covered] = color

    return torch.from_numpy(texture)


class CuMesh:
    def __init__(self):
        self.cu_mesh = _C.CuMesh()

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Initialize the CuMesh with vertices and faces.

        Args:
            vertices: a tensor of shape [V, 3] containing the vertex positions.
            faces: a tensor of shape [F, 3] containing the face indices.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input vertices must be of shape [V, 3]"
        assert faces.ndim == 2 and faces.shape[1] == 3, "Input faces must be of shape [F, 3]"
        assert vertices.is_contiguous() and faces.is_contiguous(), "Input tensors must be contiguous"
        assert vertices.is_cuda and faces.is_cuda and vertices.device == faces.device, "Input tensors must both be on the same CUDA device"
        self.cu_mesh.init(vertices, faces)
        
    @property
    def num_vertices(self) -> int:
        return self.cu_mesh.num_vertices()
    
    @property
    def num_faces(self) -> int:
        return self.cu_mesh.num_faces()
    
    @property
    def num_edges(self) -> int:
        return self.cu_mesh.num_edges()
    
    @property
    def num_boundaries(self) -> int:
        return self.cu_mesh.num_boundaries()
    
    @property
    def num_conneted_components(self) -> int:
        return self.cu_mesh.num_conneted_components()
    
    @property
    def num_boundary_conneted_components(self) -> int:
        return self.cu_mesh.num_boundary_conneted_components()
    
    @property
    def num_boundary_loops(self) -> int:
        return self.cu_mesh.num_boundary_loops()

    def clear_cache(self):
        """
        Clear the cached data.
        """
        self.cu_mesh.clear_cache()

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the current vertices and faces from the CuMesh.

        Returns:
            A tuple of two tensors: the vertex positions and the face indices.
        """
        return self.cu_mesh.read()
    
    def read_face_normals(self) -> torch.Tensor:
        """
        Read the normals of the faces from the CuMesh.

        Returns:
            The face normals as an [F, 3] tensor.
        """
        return self.cu_mesh.read_face_normals()
    
    def read_vertex_normals(self) -> torch.Tensor:
        """
        Read the normals of the vertices from the CuMesh.

        Returns:
            The vertex normals as an [V, 3] tensor.
        """
        return self.cu_mesh.read_vertex_normals()
    
    def read_edges(self) -> torch.Tensor:
        """
        Read the edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [E, 2] containing the edge indices.
        """
        return self.cu_mesh.read_edges()
    
    def read_boundaries(self) -> torch.Tensor:
        """
        Read the boundary edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [B] containing the boundary edge indices.
        """
        return self.cu_mesh.read_boundaries()
    
    
    def read_manifold_face_adjacency(self) -> torch.Tensor:
        """
        Read the manifold face adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold face adjacency.
        """
        return self.cu_mesh.read_manifold_face_adjacency()
    
    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        """
        Read the manifold boundary adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold boundary adjacency.
        """
        return self.cu_mesh.read_manifold_boundary_adjacency()
    
    def read_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each face.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [F] containing the connected component ID for each face.
        """
        return self.cu_mesh.read_connected_components()
    
    def read_boundary_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each boundary edge.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [E] containing the connected component ID for each boundary edge.
        """
        return self.cu_mesh.read_boundary_connected_components()
    
    def read_boundary_loops(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Read the boundary loops of the mesh.

        Returns:
            A tuple of three values:
                - the number of boundary loops
                - a tensor of shape [L] containing the indices of the boundary edges in each loop.
                - a tensor of shape [N_loops + 1] containing the offsets of the boundary edges in each loop.
        """
        return self.cu_mesh.read_boundary_loops()
    
    def read_all_cache(self) -> Dict[str, torch.Tensor]:
        """
        Read all cached data.

        Returns:
            A dictionary of cached data.
        """
        return self.cu_mesh.read_all_cache()
    
    def compute_face_normals(self):
        """
        Compute the normals of the faces.
        """
        self.cu_mesh.compute_face_normals()
    
    def compute_vertex_normals(self):
        """
        Compute the normals of the vertices.
        """
        self.cu_mesh.compute_vertex_normals()
        
    def get_vertex_face_adjacency(self):
        """
        Compute the vertex to face adjacency.
        """
        self.cu_mesh.get_vertex_face_adjacency()
        
    def get_edges(self):
        """
        Compute the edges of the mesh.
        """
        self.cu_mesh.get_edges()
        
    def get_edge_face_adjacency(self):
        """
        Compute the edge to face adjacency.
        """
        self.cu_mesh.get_edge_face_adjacency()
        
    def get_vertex_edge_adjacency(self):
        """
        Compute the vertex to edge adjacency.
        """
        self.cu_mesh.get_vertex_edge_adjacency()
        
    def get_boundary_info(self):
        """
        Compute the boundary information of the mesh.
        """
        self.cu_mesh.get_boundary_info()
        
    def get_vertex_boundary_adjacency(self):
        """
        Compute the vertex to boundary adjacency.
        """
        self.cu_mesh.get_vertex_boundary_adjacency()
        
    def get_manifold_face_adjacency(self):
        """
        Compute the manifold face adjacency.
        """
        self.cu_mesh.get_manifold_face_adjacency()
        
    def get_manifold_boundary_adjacency(self):
        """
        Compute the manifold boundary adjacency.
        """
        self.cu_mesh.get_manifold_boundary_adjacency()
        
    def get_connected_components(self):
        """
        Compute the connected components of the mesh.
        """
        self.cu_mesh.get_connected_components()
        
    def get_boundary_connected_components(self):
        """
        Compute the connected components of the boundary of the mesh.
        """
        self.cu_mesh.get_boundary_connected_components()
        
    def get_boundary_loops(self):
        """
        Compute the boundary loops of the mesh.
        """
        self.cu_mesh.get_boundary_loops()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh.

        Args:
            face_mask: a boolean tensor of shape [F] indicating which faces to remove.
        """
        assert face_mask.ndim == 1 and face_mask.shape[0] == self.num_faces, "face_mask must be a boolean tensor of shape [F]"
        assert face_mask.is_contiguous() and face_mask.is_cuda, "face_mask must be a CUDA tensor"
        assert face_mask.dtype == torch.bool, "face_mask must be a boolean tensor"
        self.cu_mesh.remove_faces(face_mask)
    
    def remove_unreferenced_vertices(self):
        """
        Remove unreferenced vertices from the mesh.
        """
        self.cu_mesh.remove_unreferenced_vertices()
        
    def remove_duplicate_faces(self):
        """
        Remove duplicate faces from the mesh.
        """
        self.cu_mesh.remove_duplicate_faces()
        
    def remove_degenerate_faces(self, abs_thresh: float=1e-24, rel_thresh: float=1e-12):
        """
        Remove degenerate faces from the mesh.

        Args:
            abs_thresh: absolute area threshold below which a face is considered degenerate.
            rel_thresh: relative area to square of the longest edge threshold below which a face is considered degenerate.
                Note that a face is considered degenerate if both the absolute and relative conditions are met.
        """
        self.cu_mesh.remove_degenerate_faces(abs_thresh, rel_thresh)

    def normalize(
        self,
        min_area_abs: float = 1e-24,
        min_area_rel: float = 1e-12,
        iterations: int = 1,
        verbose: bool = False,
    ):
        """Split non-manifold edges once and collapse small interior triangles.

        Args:
            min_area_abs: absolute triangle-area threshold.
            min_area_rel: triangle-area threshold relative to the square of
                the triangle's longest edge.
            iterations: number of GPU small-triangle collapse passes. Stage 1
                is always performed exactly once.
            verbose: reserved for native timing/progress diagnostics.
        """
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if min_area_abs < 0 or min_area_rel < 0:
            raise ValueError("area thresholds must be non-negative")
        self.cu_mesh.normalize(
            float(min_area_abs),
            float(min_area_rel),
            iterations,
            bool(verbose),
        )
        
    def fill_holes(self, max_hole_perimeter: float=3e-2):
        """
        Fill holes in the mesh.

        Args:
            max_hole_perimeter: the maximum perimeter of a hole to fill.
        """
        self.cu_mesh.fill_holes(max_hole_perimeter)
        
    def repair_non_manifold_edges(self):
        """
        Repair Non-manifold edges by splitting vertices.
        This creates duplicate vertices with the same coordinates.
        """
        self.cu_mesh.repair_non_manifold_edges()

    def remove_non_manifold_faces(self):
        """
        Remove faces on non-manifold edges.
        For each non-manifold edge (shared by >2 faces), only keep the first 2 faces.
        This repairs non-manifold edges by deleting faces instead of splitting vertices.
        """
        self.cu_mesh.remove_non_manifold_faces()
        
    def remove_small_connected_components(self, min_area: float):
        """
        Repair Non-manifold edges by splitting edges
        
        Args:
            min_area: the minimum area of a connected component to keep.
        """
        self.cu_mesh.remove_small_connected_components(min_area)
        
    def unify_face_orientations(self):
        """
        Unify the orientations of the faces.
        """
        self.cu_mesh.unify_face_orientations()
    
    def simplify(self, target_num_faces: int, verbose: bool=False, options: dict={}):
        """
        Simplifies the mesh using a fast approximation algorithm with gpu acceleration.

        Args:
            target_num_faces: the target number of faces to simplify to.
            verbose: whether to print the progress of the simplification.
            options: a dictionary of options for the simplification algorithm.
        """
        assert isinstance(target_num_faces, int) and target_num_faces > 0, "target_num_faces must be a positive integer"

        num_face = self.cu_mesh.num_faces()
        if num_face <= target_num_faces:
            return
        
        if verbose:
            pbar = tqdm(total=num_face-target_num_faces, desc="Simplifying", disable=not verbose)

        thresh = options.get('thresh', 1e-8)
        lambda_edge_length = options.get('lambda_edge_length', 1e-2)
        lambda_skinny = options.get('lambda_skinny', 1e-3)
        while True:
            if verbose:
                pbar.set_description(f"Simplifying [thres={thresh:.2e}]")
            
            new_num_vert, new_num_face = self.cu_mesh.simplify_step(lambda_edge_length, lambda_skinny, thresh, False)
            
            if verbose:
                pbar.update(num_face - max(target_num_faces, new_num_face))

            if new_num_face <= target_num_faces:
                break
            
            del_num_face = num_face - new_num_face
            if del_num_face / num_face < 1e-2:
                thresh *= 10
            num_face = new_num_face
            
        if verbose:
            pbar.close()
            
    def compute_charts(
        self,
        threshold_cone_half_angle_rad: float=math.radians(90),
        refine_iterations: int=0,
        global_iterations: int=3,
        smooth_strength: float=1,
        area_penalty_weight: float=0.0,
        perimeter_area_ratio_weight: float=0.0,
    ):
        """
        Compute the atlas charts.

        Args:
            threshold_cone_half_angle_rad: The threshold for the cone half angle in radians.
            refine_iterations: The number of refinement iterations.
            smooth_strength: The strength of chart boundary smoothing.
            area_penalty_weight: Coefficient for chart size penalty. Cost += Area * weight.
                                 Prevents charts from becoming too large if > 0, 
                                 or encourages larger charts if < 0 (though usually used to penalize size variance).
            perimeter_area_ratio_weight: Coefficient for shape irregularity (long-strip) penalty. 
                                         Cost += (Perimeter / Area) * weight.
                                         Higher values penalize long strips and encourage circular/compact shapes.
        """
        self.cu_mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight
        )
        
    def read_atlas_charts(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cu_mesh.read_atlas_charts()

    def merge_micro_charts(
        self,
        min_area_ratio: float = 0.0005,
        min_faces: int = 128,
        min_enclosure: float = 0.60,
        merge_iterations: int = 3,
        max_cone_half_angle_rad: float = 1.5707963,
    ) -> int:
        return self.cu_mesh.merge_micro_charts(
            min_area_ratio,
            min_faces,
            min_enclosure,
            merge_iterations,
            max_cone_half_angle_rad,
        )

    def uv_unwrap(
        self,
        compute_charts_kwargs: Optional[dict] = None,
        xatlas_pack_charts_kwargs: Optional[dict] = None,
        return_vmaps: bool = False,
        verbose: bool = False,
        return_stats: bool = False,
        debug_charts: bool = False,
    ):
        """Generate packed UVs directly from CuMesh charts.

        The pipeline normalizes the mesh, computes CuMesh charts, parameterizes
        each chart with native LSCM, adds the parameterized charts to xatlas,
        packs them, and returns the packed mesh. When debug_charts is true,
        the return value also contains an RGBA texture colored by chart.
        """
        compute_kwargs = dict(compute_charts_kwargs or {})
        compute_kwargs.setdefault("refine_iterations", 0)
        compute_kwargs.setdefault("area_penalty_weight", 0.0)
        compute_kwargs.setdefault("perimeter_area_ratio_weight", 0.0)

        pack_opts = dict(xatlas_pack_charts_kwargs or {})
        pack_opts.setdefault("padding", 4)
        pack_opts.setdefault("verbose", verbose)

        def log_completed(name: str, started_at: float, details: str = ""):
            if verbose:
                suffix = f" ({details})" if details else ""
                print(
                    f"CuMesh UV Unwrap: {name}: "
                    f"completed in {perf_counter() - started_at:.3f}s{suffix}",
                    flush=True,
                )

        trace_lscm_enabled = verbose and os.environ.get("CUMESH_UV_TRACE", "").lower() in {
            "1", "true", "yes", "on"
        }

        phase_started = perf_counter()
        self.normalize(verbose=verbose)
        log_completed(
            "normalize",
            phase_started,
            f"vertices={self.num_vertices}, faces={self.num_faces}",
        )

        phase_started = perf_counter()
        self.compute_charts(**compute_kwargs)
        log_completed("compute_charts", phase_started)

        phase_started = perf_counter()
        new_vertices, new_faces = self.read()
        (
            num_charts,
            _,
            chart_vmap,
            chart_faces,
            chart_vertex_offset,
            chart_face_offset,
        ) = self.read_atlas_charts()
        log_completed(
            "read chart data",
            phase_started,
            f"charts={num_charts}, vertices={new_vertices.shape[0]}, "
            f"faces={new_faces.shape[0]}",
        )

        if num_charts == 0:
            vertices = new_vertices.new_empty((0, 3)).cpu()
            faces = torch.empty((0, 3), dtype=torch.int32)
            uvs = torch.empty((0, 2), dtype=torch.float32)
            out = [vertices, faces, uvs]
            if debug_charts:
                chart_texture = torch.tensor(
                    [24, 24, 24, 255], dtype=torch.uint8
                ).reshape(1, 1, 4)
                out.append(chart_texture)
            if return_vmaps:
                out.append(torch.empty((0,), dtype=torch.int32))
            if return_stats:
                out.append({
                    "chart_count": 0,
                    "debug_charts": bool(debug_charts),
                    "uv_validation_passed": True,
                })
            return tuple(out)

        phase_started = perf_counter()
        chart_vertices = new_vertices[chart_vmap.long()].cpu().numpy()
        chart_faces = chart_faces.cpu().numpy()
        chart_vertex_offset = chart_vertex_offset.cpu().tolist()
        chart_face_offset = chart_face_offset.cpu().tolist()
        chart_vmap = chart_vmap.cpu().numpy()

        charts = []
        for chart_index in range(num_charts):
            vertex_start = chart_vertex_offset[chart_index]
            vertex_end = chart_vertex_offset[chart_index + 1]
            face_start = chart_face_offset[chart_index]
            face_end = chart_face_offset[chart_index + 1]
            charts.append((
                np.ascontiguousarray(
                    chart_vertices[vertex_start:vertex_end], dtype=np.float32
                ),
                np.ascontiguousarray(
                    chart_faces[face_start:face_end] - vertex_start,
                    dtype=np.int32,
                ),
                np.ascontiguousarray(
                    chart_vmap[vertex_start:vertex_end], dtype=np.int32
                ),
            ))
        log_completed(
            "copy chart data to CPU",
            phase_started,
            f"charts={num_charts}",
        )

        atlas = Atlas()
        packed_vertices = np.ascontiguousarray(
            np.concatenate([chart[0] for chart in charts], axis=0),
            dtype=np.float32,
        )
        packed_faces = np.ascontiguousarray(
            np.concatenate([chart[1] for chart in charts], axis=0),
            dtype=np.int32,
        )
        vertex_offsets = [0]
        face_offsets = [0]
        for chart_vertices_i, chart_faces_i, _ in charts:
            vertex_offsets.append(vertex_offsets[-1] + len(chart_vertices_i))
            face_offsets.append(face_offsets[-1] + len(chart_faces_i))

        lscm_progress = tqdm(
            total=num_charts,
            desc="LSCM parameterizing charts",
            disable=not verbose,
        )

        def update_lscm_progress(completed, total):
            if completed > lscm_progress.n:
                lscm_progress.n = completed
                lscm_progress.refresh()
            return True

        def trace_lscm(chart_index, phase, vertex_count, face_count):
            print(
                f"CuMesh UV Trace: LSCM chart {chart_index + 1}/{num_charts}: "
                f"{phase} (vertices={vertex_count}, faces={face_count})",
                flush=True,
            )

        phase_started = perf_counter()
        try:
            (
                batch_uvs,
                batch_faces,
                batch_vmaps,
                batch_vertex_offsets,
                batch_face_offsets,
                batch_success,
                batch_splits,
            ) = atlas._parameterize_lscm_batch(
                torch.from_numpy(packed_vertices),
                torch.from_numpy(packed_faces),
                torch.tensor(vertex_offsets, dtype=torch.int32),
                torch.tensor(face_offsets, dtype=torch.int32),
                update_lscm_progress if verbose else None,
                trace_lscm if trace_lscm_enabled else None,
            )
        finally:
            lscm_progress.close()
        parameterization_seconds = perf_counter() - phase_started
        native_stats = atlas._lscm_stats()

        batch_vertex_offsets = batch_vertex_offsets.tolist()
        batch_face_offsets = batch_face_offsets.tolist()
        batch_success = batch_success.tolist()
        batch_splits = batch_splits.tolist()
        chart_vmaps = []
        python_fallback_charts = 0
        geometry_repairs = 0
        local_splits = 0

        phase_started = perf_counter()
        for chart_index, (chart_vertices_i, chart_faces_i, chart_vmap_i) in enumerate(charts):
            p0 = chart_vertices_i[chart_faces_i[:, 0]]
            p1 = chart_vertices_i[chart_faces_i[:, 1]]
            p2 = chart_vertices_i[chart_faces_i[:, 2]]
            area_3d = 0.5 * np.sum(
                np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            )

            vertex_start = batch_vertex_offsets[chart_index]
            vertex_end = batch_vertex_offsets[chart_index + 1]
            face_start = batch_face_offsets[chart_index]
            face_end = batch_face_offsets[chart_index + 1]
            uvs_source = batch_uvs[vertex_start:vertex_end]
            faces_source = batch_faces[face_start:face_end]
            local_vmap_source = batch_vmaps[vertex_start:vertex_end]
            success_i = bool(batch_success[chart_index])

            if (
                not success_i
                or len(faces_source) != len(chart_faces_i)
                or len(uvs_source) != len(local_vmap_source)
            ):
                python_fallback_charts += 1
                uvs_source = _project_chart_uvs(chart_vertices_i)
                faces_source = chart_faces_i
                local_vmap_np = np.arange(len(chart_vertices_i), dtype=np.int32)
            else:
                local_splits += int(batch_splits[chart_index])
                uvs_source = uvs_source.numpy()
                faces_source = faces_source.numpy()
                local_vmap_np = local_vmap_source.numpy()

            uv_source_np = np.asarray(uvs_source, dtype=np.float64)
            u0 = uv_source_np[faces_source[:, 0]]
            u1 = uv_source_np[faces_source[:, 1]]
            u2 = uv_source_np[faces_source[:, 2]]
            cross = (
                (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1])
                - (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0])
            )
            area_uv = 0.5 * np.sum(np.abs(cross))
            area_scale = None
            if area_uv > 1e-12 and np.isfinite(area_3d) and area_3d > 0:
                area_scale = np.sqrt(area_3d / area_uv)

            uv_np, face_np, local_vmap_np, repaired, projected = _prepare_uv_chart(
                chart_vertices_i,
                faces_source,
                uvs_source,
                local_vmap_np,
                area_scale,
            )
            geometry_repairs += repaired
            if projected and success_i:
                python_fallback_charts += 1

            uvs_i = torch.from_numpy(uv_np)
            faces_i = torch.from_numpy(face_np)
            orig_vmap_i = torch.from_numpy(chart_vmap_i[local_vmap_np])
            chart_vmaps.append(orig_vmap_i)
            atlas.add_uv_mesh(uvs_i, faces_i)

        log_completed(
            "parameterize charts and add UV meshes",
            phase_started,
            f"fallbacks={python_fallback_charts}, repairs={geometry_repairs}",
        )

        packing_started = perf_counter()
        atlas.pack_charts(**pack_opts)
        atlas_info = atlas._atlas_info()
        packing_seconds = perf_counter() - packing_started
        log_completed(
            "pack charts",
            packing_started,
            f"atlases={atlas_info['atlas_count']}, "
            f"size={atlas_info['width']}x{atlas_info['height']}",
        )

        pages = max(1, int(atlas_info["atlas_count"]))
        columns = int(np.ceil(np.sqrt(pages)))
        rows = (pages + columns - 1) // columns
        vmaps = []
        faces = []
        uvs = []
        face_chart_ids = [] if debug_charts else None
        vertex_count = 0
        invalid_pages = False

        phase_started = perf_counter()
        for chart_index in range(num_charts):
            mapping, chart_faces_i, chart_uvs_i = atlas.get_mesh(chart_index)
            page_ids = atlas._mesh_atlas_indices(chart_index).numpy()
            invalid_pages |= bool(np.any((page_ids < 0) | (page_ids >= pages)))
            if pages > 1:
                tiled = chart_uvs_i.numpy().astype(np.float64)
                tiled[:, 0] = (tiled[:, 0] + page_ids % columns) / columns
                tiled[:, 1] = (tiled[:, 1] + page_ids // columns) / rows
                chart_uvs_i = torch.from_numpy(tiled.astype(np.float32))

            vmaps.append(chart_vmaps[chart_index][mapping.long()])
            faces.append(chart_faces_i + vertex_count)
            uvs.append(chart_uvs_i)
            if debug_charts:
                face_chart_ids.append(
                    torch.full(
                        (chart_faces_i.shape[0],),
                        chart_index,
                        dtype=torch.int32,
                    )
                )
            vertex_count += mapping.shape[0]

        vmaps = torch.cat(vmaps, dim=0).contiguous()
        faces = torch.cat(faces, dim=0).contiguous()
        uvs = torch.cat(uvs, dim=0).contiguous()
        if debug_charts:
            face_chart_ids = torch.cat(face_chart_ids, dim=0).contiguous()
        log_completed(
            "gather packed mesh",
            phase_started,
            f"vertices={vmaps.shape[0]}, faces={faces.shape[0]}",
        )

        validation = _validate_uv_output(atlas, uvs, faces)
        if invalid_pages:
            validation.update(valid=False, issue="unassigned_atlas_vertices")
        if not validation["valid"]:
            warnings.warn(
                "CuMesh UV Unwrap produced UVs that did not pass validation "
                f"({validation.get('issue', 'unknown')}).",
                RuntimeWarning,
            )

        chart_texture = None
        if debug_charts:
            texture_width = max(1, int(atlas_info["width"])) * columns
            texture_height = max(1, int(atlas_info["height"])) * rows
            chart_texture = _rasterize_chart_texture(
                uvs,
                faces,
                face_chart_ids,
                texture_width,
                texture_height,
            )

        vertices = new_vertices.cpu()[vmaps.long()]
        stats = {
            "chart_count": num_charts,
            "debug_charts": bool(debug_charts),
            "local_splits": local_splits,
            "python_fallback_charts": python_fallback_charts,
            "geometry_repairs": geometry_repairs,
            "parameterization_seconds": parameterization_seconds,
            "packing_seconds": packing_seconds,
            "xatlas_atlas_count": atlas_info["atlas_count"],
            "uv_validation_passed": bool(validation["valid"]),
            **native_stats,
        }

        out = [vertices, faces, uvs]
        if debug_charts:
            out.append(chart_texture)
        if return_vmaps:
            out.append(vmaps)
        if return_stats:
            out.append(stats)
        return tuple(out)


