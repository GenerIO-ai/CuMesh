from typing import *
import math
import os
import torch
import numpy as np
from time import perf_counter
from tqdm import tqdm
from .xatlas import Atlas
from .flatten_lscm import parameterize_charts as flatten_lscm_charts
from .flatten_pca import project_chart_uvs as project_pca_uvs
from .flatten_project import project_chart_uvs as project_chart_flatten_uvs
from .flatten_tutte import parameterize_charts as flatten_tutte_charts
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


_project_chart_uvs = project_pca_uvs

_FLATTEN_METHODS = ("lscm", "project", "tutte", "pca")


def _normalize_flatten_methods(flatten):
    """Validate and normalize the ordered chart flattening stages."""
    if flatten is None:
        methods = ("project",)
    elif isinstance(flatten, str):
        methods = (flatten,)
    else:
        try:
            methods = tuple(flatten)
        except TypeError as error:
            raise TypeError(
                "flatten must be a string or an ordered sequence of strings"
            ) from error

    normalized = []
    for method in methods:
        if not isinstance(method, str):
            raise TypeError("flatten methods must be strings")
        method = method.strip().lower()
        if method not in _FLATTEN_METHODS:
            raise ValueError(
                f"unknown flatten method {method!r}; "
                f"expected one of {', '.join(_FLATTEN_METHODS)}"
            )
        if method not in normalized:
            normalized.append(method)
    return tuple(normalized)


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


def _prepare_uv_chart(vertices, faces, uvs, vmaps, area_scale=None, allow_fallback=False):
    """Condition chart UVs for float32/xatlas and repair zero-area faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    vmap_np = np.ascontiguousarray(np.asarray(vmaps, dtype=np.int32))
    uv_np = np.asarray(uvs, dtype=np.float64)
    projected = False
    if uv_np.ndim != 2 or uv_np.shape != (len(vmap_np), 2) or not np.isfinite(uv_np).all():
        if not allow_fallback:
            raise RuntimeError(
                "A chart did not produce valid UVs"
            )
        uv_np = _project_chart_uvs(vertices)
        projected = True

    if area_scale is not None and np.isfinite(area_scale) and area_scale > 0:
        uv_np *= float(area_scale)

    lower = np.min(uv_np, axis=0)
    extent = float(np.ptp(uv_np, axis=0).max())
    if not np.isfinite(lower).all() or not np.isfinite(extent) or extent <= 0:
        if not allow_fallback:
            raise RuntimeError(
                "A chart produced collapsed UVs"
            )
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


def _emit_uv_warning(message, affected=0, total=0, critical=False):
    """Print a compact best-effort UV warning with its observed severity."""
    affected = int(max(0, affected))
    total = int(max(0, total))
    ratio = affected / total if total else (1.0 if affected else 0.0)
    level = "ERROR" if critical or ratio >= 0.25 else "WARNING"
    suffix = f" ({affected}/{total}, {100.0 * ratio:.2f}%)" if total else ""
    print(f"CuMesh UV Warning [{level}]: {message}{suffix}", flush=True)
    return level


def _safe_chart_faces(faces, vertex_count):
    """Return valid triangle rows and the number of rows discarded."""
    try:
        face_np = np.asarray(faces, dtype=np.int64)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=np.int32), 0
    if face_np.ndim != 2 or face_np.shape[1] != 3:
        return np.empty((0, 3), dtype=np.int32), int(len(face_np)) if face_np.ndim else 0
    valid = np.isfinite(face_np).all(axis=1)
    valid &= np.all((face_np >= 0) & (face_np < int(vertex_count)), axis=1)
    valid &= ~(
        (face_np[:, 0] == face_np[:, 1])
        | (face_np[:, 0] == face_np[:, 2])
        | (face_np[:, 1] == face_np[:, 2])
    )
    return np.ascontiguousarray(face_np[valid], dtype=np.int32), int(np.count_nonzero(~valid))


def _emergency_project_uvs(vertices):
    """Create finite, normalized planar UVs for the final recovery path."""
    points = np.asarray(vertices, dtype=np.float64)
    projected = np.asarray(_project_chart_uvs(points), dtype=np.float64)
    if projected.ndim != 2 or projected.shape != (len(points), 2) or not np.isfinite(projected).all():
        projected = np.column_stack((
            np.arange(len(points), dtype=np.float64),
            (np.arange(len(points), dtype=np.float64) % 2) * 1e-3,
        ))
    if len(projected) == 0:
        return np.empty((0, 2), dtype=np.float32)
    lower = np.min(projected, axis=0)
    extent = float(np.ptp(projected, axis=0).max())
    if not np.isfinite(lower).all() or not np.isfinite(extent) or extent <= 0.0:
        projected = np.column_stack((
            np.arange(len(points), dtype=np.float64),
            (np.arange(len(points), dtype=np.float64) % 2) * 1e-3,
        ))
        lower = np.min(projected, axis=0)
        extent = float(np.ptp(projected, axis=0).max())
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 1.0
    return np.ascontiguousarray(((projected - lower) / extent).astype(np.float32))


def _fallback_pack_charts(records):
    """Pack prepared chart records into a small non-overlapping grid."""
    count = len(records)
    if count == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    columns = max(1, int(np.ceil(np.sqrt(count))))
    rows = max(1, int(np.ceil(count / columns)))
    cell_width = 1.0 / columns
    cell_height = 1.0 / rows
    margin_x = 0.05 * cell_width
    margin_y = 0.05 * cell_height
    placed_uvs = []
    placed_faces = []
    placed_vmaps = []
    placed_chart_ids = []
    vertex_offset = 0
    for record_index, record in enumerate(records):
        uv_np = np.asarray(record["uvs"], dtype=np.float64)
        face_np = np.asarray(record["faces"], dtype=np.int32)
        vmap_np = np.asarray(record["vmaps"], dtype=np.int32)
        lower = np.min(uv_np, axis=0)
        extent = np.ptp(uv_np, axis=0)
        width = max(float(extent[0]), 1e-12)
        height = max(float(extent[1]), 1e-12)
        scale = min((cell_width - 2.0 * margin_x) / width,
                    (cell_height - 2.0 * margin_y) / height)
        column = record_index % columns
        row = record_index // columns
        placed = np.empty_like(uv_np, dtype=np.float64)
        placed[:, 0] = column * cell_width + margin_x + (uv_np[:, 0] - lower[0]) * scale
        placed[:, 1] = row * cell_height + margin_y + (uv_np[:, 1] - lower[1]) * scale
        placed_uvs.append(np.ascontiguousarray(placed.astype(np.float32)))
        placed_faces.append(np.ascontiguousarray(face_np + vertex_offset, dtype=np.int32))
        placed_vmaps.append(vmap_np)
        placed_chart_ids.append(np.full(len(face_np), int(record["chart_index"]), dtype=np.int32))
        vertex_offset += len(vmap_np)
    return (
        np.concatenate(placed_vmaps, axis=0),
        np.concatenate(placed_faces, axis=0),
        np.concatenate(placed_uvs, axis=0),
        np.concatenate(placed_chart_ids, axis=0),
    )


def _split_uv_chart_record(atlas, record, next_chart_index):
    """Split one prepared chart into native overlap-free subcharts."""
    result = atlas._split_uv_overlaps(
        torch.from_numpy(record["uvs"]),
        torch.from_numpy(record["faces"]),
    )
    (
        groups,
        group_count,
        overlap_faces,
        overlap_pairs,
        candidates,
        valid,
        limited,
    ) = result
    groups = np.asarray(groups.numpy(), dtype=np.int64).reshape(-1)
    group_count = int(group_count)
    if (
        not bool(valid)
        or len(groups) != len(record["faces"])
        or group_count <= 1
        or np.any(groups < 0)
        or np.any(groups >= group_count)
    ):
        return [record], {
            "overlap_faces": int(overlap_faces),
            "overlap_pairs": int(overlap_pairs),
            "candidates": int(candidates),
            "split_charts": 0,
            "limited": bool(limited),
            "failed": not bool(valid),
        }, next_chart_index

    subrecords = []
    if group_count <= 8:
        grouped_faces = (
            (group, np.flatnonzero(groups == group))
            for group in range(group_count)
        )
    else:
        # The dense-graph safeguard can produce many groups. Sort once so
        # this remains O(F log F), rather than scanning the chart once per
        # face-sized group.
        order = np.argsort(groups, kind="stable")
        ordered_groups = groups[order]
        starts = np.flatnonzero(
            np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
        )
        ends = np.r_[starts[1:], len(order)]
        grouped_faces = (
            (int(ordered_groups[start]), order[start:end])
            for start, end in zip(starts, ends)
        )
    for group, face_indices in grouped_faces:
        if len(face_indices) == 0:
            continue
        selected_faces = record["faces"][face_indices]
        used_vertices = np.unique(selected_faces.reshape(-1))
        remap = np.full(len(record["uvs"]), -1, dtype=np.int64)
        remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
        subrecord = {
            "chart_index": (
                int(record["chart_index"])
                if group == 0
                else int(next_chart_index)
            ),
            "source_chart_index": int(record["chart_index"]),
            "uvs": np.ascontiguousarray(record["uvs"][used_vertices], dtype=np.float32),
            "faces": np.ascontiguousarray(remap[selected_faces], dtype=np.int32),
            "vmaps": np.ascontiguousarray(record["vmaps"][used_vertices], dtype=np.int32),
        }
        if "positions" in record:
            subrecord["positions"] = np.ascontiguousarray(
                record["positions"][used_vertices], dtype=np.float32
            )
        if group != 0:
            next_chart_index += 1
        subrecords.append(subrecord)
    if not subrecords:
        subrecords = [record]
    return subrecords, {
        "overlap_faces": int(overlap_faces),
        "overlap_pairs": int(overlap_pairs),
        "candidates": int(candidates),
        "split_charts": max(0, len(subrecords) - 1),
        "limited": bool(limited),
        "failed": False,
    }, next_chart_index


def _emergency_full_mesh_result(
    vertices,
    faces,
    debug_charts,
    return_vmaps,
    return_stats,
    flatten_methods,
    warning_reason,
):
    """Return a complete planar mesh when charting or packing is unavailable."""
    vertices_cpu = vertices.detach().cpu().contiguous()
    vertex_np = np.asarray(vertices_cpu, dtype=np.float64)
    face_np, dropped_faces = _safe_chart_faces(faces, len(vertex_np))
    uv_np = _emergency_project_uvs(vertex_np)
    faces_cpu = torch.from_numpy(face_np)
    uvs_cpu = torch.from_numpy(uv_np)
    vmaps_cpu = torch.arange(len(vertex_np), dtype=torch.int32)
    chart_texture = None
    if debug_charts:
        chart_texture = _rasterize_chart_texture(
            uvs_cpu,
            faces_cpu,
            torch.zeros(len(face_np), dtype=torch.int32),
            1024,
            1024,
        )
    stats = {
        "chart_count": 0,
        "debug_charts": bool(debug_charts),
        "flatten": list(flatten_methods),
        "python_fallback_charts": 0,
        "native_lscm_charts": 0,
        "native_graph_fallback_charts": 0,
        "emergency_fallback_charts": 0,
        "dropped_charts": 0,
        "dropped_faces": dropped_faces,
        "unmapped_vertices": 0,
        "packed_chart_count": 0,
        "self_overlapping_charts": 0,
        "self_overlap_faces": 0,
        "self_overlap_pairs": 0,
        "self_overlap_candidates": 0,
        "overlap_split_charts": 0,
        "overlap_detection_failures": 0,
        "overlap_detection_limited_charts": 0,
        "atlas_complete": False,
        "emergency_full_mesh_fallback": True,
        "uv_warning_level": _emit_uv_warning(
            warning_reason,
            affected=1,
            total=1,
            critical=True,
        ),
        "uv_validation_passed": False,
    }
    out = [vertices_cpu, faces_cpu, uvs_cpu, chart_texture]
    if return_vmaps:
        out.append(vmaps_cpu)
    if return_stats:
        out.append(stats)
    return tuple(out)


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
        flatten: Sequence[str] = ("project",),
    ):
        """Generate packed UVs directly from CuMesh charts.

        The pipeline normalizes the mesh, computes CuMesh charts, parameterizes
        each chart with the flattening stages in ``flatten`` order, adds the
        parameterized charts to xatlas, packs them, and returns the packed
        mesh. The return value always starts with
        ``(vertices, faces, uvs, chart_texture)``; ``chart_texture`` is
        ``None`` unless ``debug_charts`` is true.

        Args:
            flatten: Ordered flattening methods. Supported values are
                ``"lscm"``, ``"project"``, ``"tutte"``, and ``"pca"``.

        The default is ``["project"]``. A chart uses the first method that
        produces valid UVs and falls through to later methods on failure. If
        every requested method fails, a final emergency planar UV is returned.
        """
        flatten_methods = _normalize_flatten_methods(flatten)
        if not flatten_methods:
            _emit_uv_warning(
                "all flattening stages are disabled; emergency planar UVs will be used",
                affected=1,
                total=1,
                critical=True,
            )

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

        trace_flatten_enabled = verbose and os.environ.get("CUMESH_UV_TRACE", "").lower() in {
            "1", "true", "yes", "on"
        }

        def emergency_current_mesh(reason):
            try:
                current_vertices, current_faces = self.read()
            except Exception as error:
                raise RuntimeError(
                    "CuMesh UV Unwrap could not read the mesh for its final "
                    "UV fallback"
                ) from error
            return _emergency_full_mesh_result(
                current_vertices,
                current_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                reason,
            )

        phase_started = perf_counter()
        try:
            self.normalize(verbose=verbose)
        except Exception as error:
            return emergency_current_mesh(
                f"mesh normalization failed ({type(error).__name__}); "
                "returned a planar emergency UV mesh"
            )
        log_completed(
            "normalize",
            phase_started,
            f"vertices={self.num_vertices}, faces={self.num_faces}",
        )

        phase_started = perf_counter()
        try:
            self.compute_charts(**compute_kwargs)
        except Exception as error:
            return emergency_current_mesh(
                f"chart generation failed ({type(error).__name__}); "
                "returned a planar emergency UV mesh"
            )
        log_completed("compute_charts", phase_started)

        phase_started = perf_counter()
        try:
            new_vertices, new_faces = self.read()
            (
                num_charts,
                _,
                chart_vmap,
                chart_faces,
                chart_vertex_offset,
                chart_face_offset,
            ) = self.read_atlas_charts()
        except Exception as error:
            fallback_vertices = locals().get("new_vertices")
            fallback_faces = locals().get("new_faces")
            if fallback_vertices is None or fallback_faces is None:
                fallback_vertices, fallback_faces = self.read()
            return _emergency_full_mesh_result(
                fallback_vertices,
                fallback_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                f"chart data could not be read ({type(error).__name__}); "
                "returned a planar emergency UV mesh",
            )
        log_completed(
            "read chart data",
            phase_started,
            f"charts={num_charts}, vertices={new_vertices.shape[0]}, "
            f"faces={new_faces.shape[0]}",
        )

        if num_charts == 0:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                "no charts were produced; returned a planar emergency UV mesh",
            )

        phase_started = perf_counter()
        try:
            chart_vertices = new_vertices[chart_vmap.long()].cpu().numpy()
            chart_faces = chart_faces.cpu().numpy()
            chart_vertex_offset = chart_vertex_offset.cpu().tolist()
            chart_face_offset = chart_face_offset.cpu().tolist()
            chart_vmap = chart_vmap.cpu().numpy()
        except Exception as error:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                f"chart vertex mapping could not be converted ({type(error).__name__}); "
                "returned a planar emergency UV mesh",
            )

        charts = []
        try:
            if (
                len(chart_vertex_offset) != num_charts + 1
                or len(chart_face_offset) != num_charts + 1
            ):
                raise ValueError("chart offsets do not cover the complete atlas")
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
        except Exception as error:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                f"chart offsets were incomplete ({type(error).__name__}); "
                "returned a planar emergency UV mesh",
            )
        log_completed(
            "copy chart data to CPU",
            phase_started,
            f"charts={num_charts}",
        )

        try:
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
        except Exception as error:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                f"atlas input could not be assembled ({type(error).__name__}); "
                "returned a planar emergency UV mesh",
            )

        def trace_flatten(chart_index, phase, vertex_count, face_count):
            print(
                f"CuMesh UV Trace: flatten chart {chart_index + 1}/{num_charts}: "
                f"{phase} (vertices={vertex_count}, faces={face_count})",
                flush=True,
            )

        # Adjacent native stages can share one native pass: the LSCM binding
        # already knows how to use Tutte for the charts LSCM cannot flatten.
        # If another method sits between them, keep the passes separate so the
        # requested order remains observable per chart.
        native_jobs = []
        method_index = 0
        while method_index < len(flatten_methods):
            method = flatten_methods[method_index]
            if (
                method == "lscm"
                and method_index + 1 < len(flatten_methods)
                and flatten_methods[method_index + 1] == "tutte"
            ):
                native_jobs.append(("lscm", True))
                method_index += 2
            elif method in ("lscm", "tutte"):
                native_jobs.append((method, False))
                method_index += 1
            else:
                method_index += 1

        native_candidates = {}
        native_errors = []
        phase_started = perf_counter()
        flatten_progress = tqdm(
            total=num_charts * len(native_jobs),
            desc="Flattening charts",
            disable=not verbose or not native_jobs,
        )
        progress_offset = [0]

        def update_native_progress(completed, total):
            progress = progress_offset[0] + min(int(completed), num_charts)
            if progress > flatten_progress.n:
                flatten_progress.n = progress
                flatten_progress.refresh()
            return True

        flatten_args = (
            atlas,
            torch.from_numpy(packed_vertices),
            torch.from_numpy(packed_faces),
            torch.tensor(vertex_offsets, dtype=torch.int32),
            torch.tensor(face_offsets, dtype=torch.int32),
            update_native_progress if verbose else None,
            trace_flatten if trace_flatten_enabled else None,
        )
        for method, combined_tutte in native_jobs:
            try:
                if method == "lscm":
                    batch = flatten_lscm_charts(
                        *flatten_args,
                        flatten_tutte=combined_tutte,
                    )
                else:
                    batch = flatten_tutte_charts(*flatten_args)
                (
                    batch_uvs,
                    batch_faces,
                    batch_vmaps,
                    batch_vertex_offsets,
                    batch_face_offsets,
                    batch_success,
                    batch_splits,
                    batch_fallback_counts,
                ) = batch
                batch_success = batch_success.bool()
                batch_fallback_counts = batch_fallback_counts.to(torch.int32)
                native_candidates[method] = {
                    "uvs": batch_uvs,
                    "faces": batch_faces,
                    "vmaps": batch_vmaps,
                    "vertex_offsets": batch_vertex_offsets.tolist(),
                    "face_offsets": batch_face_offsets.tolist(),
                    "success": (
                        batch_success & (batch_fallback_counts == 0)
                        if combined_tutte
                        else batch_success
                    ),
                    "splits": batch_splits.tolist(),
                    "fallback": batch_fallback_counts,
                    "combined": combined_tutte,
                }
                if combined_tutte:
                    native_candidates["tutte"] = {
                        "uvs": batch_uvs,
                        "faces": batch_faces,
                        "vmaps": batch_vmaps,
                        "vertex_offsets": batch_vertex_offsets.tolist(),
                        "face_offsets": batch_face_offsets.tolist(),
                        "success": batch_success & (batch_fallback_counts > 0),
                        "splits": batch_splits.tolist(),
                        "fallback": batch_fallback_counts,
                        "combined": True,
                    }
            except Exception as error:
                native_errors.append((method, error))
            progress_offset[0] += num_charts
        flatten_progress.close()
        parameterization_seconds = perf_counter() - phase_started
        try:
            native_stats = dict(atlas._lscm_stats())
        except Exception:
            native_stats = {}

        for method, error in native_errors:
            _emit_uv_warning(
                f"native {method} chart flattening failed ({type(error).__name__}); "
                "later requested flattening stages will be tried",
                affected=num_charts,
                total=num_charts,
                critical=True,
            )
        if native_errors:
            native_stats["native_batch_errors"] = len(native_errors)

        chart_records = []
        python_fallback_charts = 0
        native_lscm_charts = 0
        native_graph_fallback_charts = 0
        project_charts = 0
        native_failed_chart_count = 0
        emergency_fallback_charts = 0
        dropped_charts = 0
        dropped_faces = 0
        atlas_add_failures = 0
        geometry_repairs = 0
        local_splits = 0
        overlap_charts = 0
        overlap_faces = 0
        overlap_pairs = 0
        overlap_candidates = 0
        overlap_split_charts = 0
        overlap_detection_failures = 0
        overlap_detection_limited_charts = 0
        distortion_optimization_attempts = 0
        distortion_optimized_charts = 0
        distortion_optimization_failures = 0
        distortion_optimization_unavailable = False
        distortion_optimization_seconds = 0.0
        next_chart_index = num_charts

        phase_started = perf_counter()
        for chart_index, (chart_vertices_i, chart_faces_i, chart_vmap_i) in enumerate(charts):
            try:
                chart_faces_clean, invalid_chart_faces = _safe_chart_faces(
                    chart_faces_i, len(chart_vertices_i)
                )
                dropped_faces += invalid_chart_faces
                p0 = chart_vertices_i[chart_faces_clean[:, 0]]
                p1 = chart_vertices_i[chart_faces_clean[:, 1]]
                p2 = chart_vertices_i[chart_faces_clean[:, 2]]
                area_3d = 0.5 * np.sum(
                    np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
                )
            except (IndexError, TypeError, ValueError):
                chart_faces_clean = np.empty((0, 3), dtype=np.int32)
                area_3d = 0.0

            if not len(chart_faces_clean):
                dropped_charts += 1
                continue

            selected = None
            attempted_native = False
            for method in flatten_methods:
                if method in ("lscm", "tutte"):
                    attempted_native = True
                    native = native_candidates.get(method)
                    if native is None or not bool(native["success"][chart_index]):
                        continue
                    vertex_start = native["vertex_offsets"][chart_index]
                    vertex_end = native["vertex_offsets"][chart_index + 1]
                    face_start = native["face_offsets"][chart_index]
                    face_end = native["face_offsets"][chart_index + 1]
                    uvs_source = native["uvs"][vertex_start:vertex_end].numpy()
                    faces_source = native["faces"][face_start:face_end].numpy()
                    local_vmap_np = native["vmaps"][vertex_start:vertex_end].numpy()
                    faces_source, invalid_native_faces = _safe_chart_faces(
                        faces_source, len(local_vmap_np)
                    )
                    dropped_faces += invalid_native_faces
                    native_fallback_i = method == "tutte" and native["combined"]
                    split_count = int(native["splits"][chart_index])
                elif method == "project":
                    try:
                        uvs_source = project_chart_flatten_uvs(
                            chart_vertices_i, chart_faces_clean
                        )
                    except Exception:
                        continue
                    faces_source = chart_faces_clean
                    local_vmap_np = np.arange(len(chart_vertices_i), dtype=np.int32)
                    native_fallback_i = False
                    split_count = 0
                else:  # pca
                    try:
                        uvs_source = _project_chart_uvs(chart_vertices_i)
                    except Exception:
                        continue
                    faces_source = chart_faces_clean
                    local_vmap_np = np.arange(len(chart_vertices_i), dtype=np.int32)
                    native_fallback_i = False
                    split_count = 0

                if len(faces_source) == 0:
                    continue
                try:
                    uv_source_np = np.asarray(uvs_source, dtype=np.float64)
                    u0 = uv_source_np[faces_source[:, 0]]
                    u1 = uv_source_np[faces_source[:, 1]]
                    u2 = uv_source_np[faces_source[:, 2]]
                    cross = (
                        (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1])
                        - (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0])
                    )
                    area_uv = 0.5 * np.sum(np.abs(cross))
                except (IndexError, TypeError, ValueError):
                    continue
                area_scale = None
                if area_uv > 1e-12 and np.isfinite(area_3d) and area_3d > 0:
                    area_scale = np.sqrt(area_3d / area_uv)

                try:
                    prepared = _prepare_uv_chart(
                        chart_vertices_i,
                        faces_source,
                        uvs_source,
                        local_vmap_np,
                        area_scale,
                        allow_fallback=False,
                    )
                except Exception:
                    continue
                selected = (method, prepared, native_fallback_i, split_count)
                break

            if selected is None:
                emergency_fallback_charts += 1
                try:
                    prepared = _prepare_uv_chart(
                        chart_vertices_i,
                        chart_faces_clean,
                        _emergency_project_uvs(chart_vertices_i),
                        np.arange(len(chart_vertices_i), dtype=np.int32),
                        None,
                        allow_fallback=False,
                    )
                except Exception:
                    dropped_charts += 1
                    dropped_faces += len(chart_faces_clean)
                    continue
                selected = ("emergency", prepared, False, 0)

            selected_method, prepared, native_fallback_i, split_count = selected
            uv_np, face_np, local_vmap_np, repaired, projected = prepared
            geometry_repairs += repaired
            local_splits += split_count
            if selected_method == "pca":
                python_fallback_charts += 1
            elif selected_method == "project":
                project_charts += 1
            elif selected_method == "tutte":
                native_graph_fallback_charts += 1
            elif selected_method == "lscm":
                native_lscm_charts += 1
            if attempted_native and selected_method not in ("lscm", "tutte"):
                native_failed_chart_count += 1

            chart_vmap_i = np.asarray(chart_vmap_i, dtype=np.int64)
            local_vmap_np = np.asarray(local_vmap_np, dtype=np.int64)
            face_np, invalid_prepared_faces = _safe_chart_faces(
                face_np, len(local_vmap_np)
            )
            dropped_faces += invalid_prepared_faces
            valid_vmap = (
                (local_vmap_np >= 0)
                & (local_vmap_np < len(chart_vmap_i))
            )
            valid_global_vmap = np.zeros(len(local_vmap_np), dtype=bool)
            valid_global_vmap[valid_vmap] = (
                chart_vmap_i[local_vmap_np[valid_vmap]] >= 0
            ) & (
                chart_vmap_i[local_vmap_np[valid_vmap]] < len(new_vertices)
            )
            face_valid_vmaps = valid_global_vmap[face_np].all(axis=1) if len(face_np) else np.empty((0,), dtype=bool)
            if len(face_np) and not face_valid_vmaps.all():
                dropped_faces += int(np.count_nonzero(~face_valid_vmaps))
                face_np = face_np[face_valid_vmaps]
            if len(face_np) == 0:
                dropped_charts += 1
                continue

            try:
                orig_vmap_np = chart_vmap_i[local_vmap_np]
                chart_positions_np = np.zeros(
                    (len(local_vmap_np), 3), dtype=np.float32
                )
                if np.any(valid_vmap):
                    chart_positions_np[valid_vmap] = chart_vertices_i[
                        local_vmap_np[valid_vmap]
                    ]
                record = {
                    "chart_index": chart_index,
                    "source_chart_index": chart_index,
                    "uvs": np.ascontiguousarray(uv_np, dtype=np.float32),
                    "faces": np.ascontiguousarray(face_np, dtype=np.int32),
                    "vmaps": np.ascontiguousarray(orig_vmap_np, dtype=np.int32),
                    # Keep the matching 3D chart vertices beside the UVs so
                    # the post-flatten optimizer can evaluate the local
                    # metric without another global gather.
                    "positions": chart_positions_np,
                }
                try:
                    records_to_add, overlap_info, next_chart_index = _split_uv_chart_record(
                        atlas, record, next_chart_index
                    )
                except Exception:
                    records_to_add = [record]
                    overlap_info = {
                        "overlap_faces": 0,
                        "overlap_pairs": 0,
                        "candidates": 0,
                        "split_charts": 0,
                        "limited": False,
                        "failed": True,
                    }
                overlap_faces += overlap_info["overlap_faces"]
                overlap_pairs += overlap_info["overlap_pairs"]
                overlap_candidates += overlap_info["candidates"]
                overlap_split_charts += overlap_info["split_charts"]
                overlap_detection_limited_charts += int(overlap_info["limited"])
                overlap_charts += int(overlap_info["overlap_pairs"] > 0)
                overlap_detection_failures += int(overlap_info["failed"])
                chart_records.extend(records_to_add)
            except Exception:
                atlas_add_failures += 1
                dropped_charts += 1
                dropped_faces += len(face_np)

        # Optimize only after the first overlap split.  This keeps all
        # flatteners on the same path and lets the cheap nonlinear pass see
        # the actual final chart boundaries.  A changed chart is checked once
        # more because moving its boundary can create a new self-overlap.
        optimization_started = perf_counter()
        optimizer_available = callable(
            getattr(getattr(atlas, "atlas", None), "_optimize_uv_distortion", None)
        )
        if optimizer_available:
            optimized_records = []
            for subrecord in chart_records:
                if len(subrecord["faces"]) < 2:
                    optimized_records.append(subrecord)
                    continue
                distortion_optimization_attempts += 1
                changed = False
                try:
                    changed = bool(atlas._optimize_uv_distortion(
                        torch.from_numpy(subrecord["positions"]),
                        torch.from_numpy(subrecord["uvs"]),
                        torch.from_numpy(subrecord["faces"]),
                    ))
                except Exception:
                    distortion_optimization_failures += 1

                records_after_optimization = [subrecord]
                if changed:
                    distortion_optimized_charts += 1
                    try:
                        (
                            records_after_optimization,
                            overlap_info,
                            next_chart_index,
                        ) = _split_uv_chart_record(
                            atlas, subrecord, next_chart_index
                        )
                    except Exception:
                        overlap_info = {
                            "overlap_faces": 0,
                            "overlap_pairs": 0,
                            "candidates": 0,
                            "split_charts": 0,
                            "limited": False,
                            "failed": True,
                        }
                    overlap_faces += overlap_info["overlap_faces"]
                    overlap_pairs += overlap_info["overlap_pairs"]
                    overlap_candidates += overlap_info["candidates"]
                    overlap_split_charts += overlap_info["split_charts"]
                    overlap_detection_limited_charts += int(overlap_info["limited"])
                    overlap_charts += int(overlap_info["overlap_pairs"] > 0)
                    overlap_detection_failures += int(overlap_info["failed"])
                optimized_records.extend(records_after_optimization)
            chart_records = optimized_records
        else:
            distortion_optimization_unavailable = True
        distortion_optimization_seconds = perf_counter() - optimization_started
        log_completed(
            "optimize chart distortion",
            optimization_started,
            f"attempts={distortion_optimization_attempts}, "
            f"optimized={distortion_optimized_charts}, "
            f"failures={distortion_optimization_failures}",
        )

        records_added = []
        atlas_index = 0
        for subrecord in chart_records:
            try:
                atlas.add_uv_mesh(
                    torch.from_numpy(subrecord["uvs"]),
                    torch.from_numpy(subrecord["faces"]),
                )
                subrecord["atlas_index"] = atlas_index
                records_added.append(subrecord)
                atlas_index += 1
            except Exception:
                atlas_add_failures += 1
                dropped_charts += 1
                dropped_faces += len(subrecord["faces"])

        chart_records = records_added
        for record in chart_records:
            record.pop("positions", None)

        log_completed(
            "parameterize charts, optimize, and add UV meshes",
            phase_started,
            f"python_fallbacks={python_fallback_charts}, "
            f"project_charts={project_charts}, "
            f"emergency_fallbacks={emergency_fallback_charts}, "
            f"overlap_splits={overlap_split_charts}, "
            f"distortion_optimized={distortion_optimized_charts}, "
            f"dropped_charts={dropped_charts}, dropped_faces={dropped_faces}, "
            f"repairs={geometry_repairs}",
        )

        if not chart_records:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                "no chart could be converted to UVs; returned a planar emergency UV mesh",
            )

        packing_started = perf_counter()
        packing_error = None
        try:
            atlas.pack_charts(**pack_opts)
            atlas_info = dict(atlas._atlas_info())
            if int(atlas_info.get("atlas_count", 0)) <= 0:
                raise RuntimeError("the atlas contains no pages")
        except Exception as error:
            packing_error = error
            atlas_info = {"atlas_count": 0, "width": 0, "height": 0}
        packing_seconds = perf_counter() - packing_started

        packing_used_fallback = packing_error is not None
        if packing_error is not None:
            _emit_uv_warning(
                f"atlas packing failed ({type(packing_error).__name__}); "
                "using a simple fallback grid",
                affected=1,
                total=1,
            )

        pages = max(1, int(atlas_info.get("atlas_count", 0)))
        columns = int(np.ceil(np.sqrt(pages)))
        rows = (pages + columns - 1) // columns
        phase_started = perf_counter()
        if packing_used_fallback:
            vmaps_np, faces_np, uvs_np, face_chart_ids_np = _fallback_pack_charts(chart_records)
            atlas_info = {"atlas_count": 1, "width": 1024, "height": 1024}
            pages = columns = rows = 1
        else:
            gathered_vmaps = []
            gathered_faces = []
            gathered_uvs = []
            gathered_chart_ids = []
            vertex_count = 0
            try:
                for record in chart_records:
                    atlas_index = int(record["atlas_index"])
                    mapping, chart_faces_i, chart_uvs_i = atlas.get_mesh(atlas_index)
                    mapping_np = np.asarray(mapping.numpy(), dtype=np.int64).reshape(-1)
                    chart_faces_np = np.asarray(chart_faces_i.numpy(), dtype=np.int64)
                    chart_uvs_np = np.asarray(chart_uvs_i.numpy(), dtype=np.float64)
                    page_ids = np.asarray(
                        atlas._mesh_atlas_indices(atlas_index).numpy(), dtype=np.int64
                    ).reshape(-1)
                    if (
                        mapping_np.ndim != 1
                        or chart_uvs_np.shape != (len(mapping_np), 2)
                        or chart_faces_np.ndim != 2
                        or chart_faces_np.shape[1] != 3
                        or np.any(mapping_np < 0)
                        or np.any(mapping_np >= len(record["vmaps"]))
                        or np.any(chart_faces_np < 0)
                        or np.any(chart_faces_np >= len(mapping_np))
                        or len(page_ids) != len(mapping_np)
                        or np.any(page_ids < 0)
                        or np.any(page_ids >= pages)
                        or not np.isfinite(chart_uvs_np).all()
                    ):
                        raise ValueError("the packed chart mapping is incomplete or invalid")
                    if pages > 1:
                        chart_uvs_np[:, 0] = (chart_uvs_np[:, 0] + page_ids % columns) / columns
                        chart_uvs_np[:, 1] = (chart_uvs_np[:, 1] + page_ids // columns) / rows
                    gathered_vmaps.append(record["vmaps"][mapping_np].astype(np.int32))
                    gathered_faces.append(
                        np.ascontiguousarray(chart_faces_np.astype(np.int32) + vertex_count)
                    )
                    gathered_uvs.append(np.ascontiguousarray(chart_uvs_np.astype(np.float32)))
                    gathered_chart_ids.append(
                        np.full(len(chart_faces_np), int(record["chart_index"]), dtype=np.int32)
                    )
                    vertex_count += len(mapping_np)
                vmaps_np = np.concatenate(gathered_vmaps, axis=0)
                faces_np = np.concatenate(gathered_faces, axis=0)
                uvs_np = np.concatenate(gathered_uvs, axis=0)
                face_chart_ids_np = np.concatenate(gathered_chart_ids, axis=0)
            except Exception as error:
                packing_used_fallback = True
                _emit_uv_warning(
                    f"packed chart mapping was incomplete ({type(error).__name__}); "
                    "using a simple fallback grid",
                    affected=1,
                    total=1,
                )
                vmaps_np, faces_np, uvs_np, face_chart_ids_np = _fallback_pack_charts(
                    chart_records
                )
                atlas_info = {"atlas_count": 1, "width": 1024, "height": 1024}
                pages = columns = rows = 1

        vmaps = torch.from_numpy(np.ascontiguousarray(vmaps_np, dtype=np.int32))
        faces = torch.from_numpy(np.ascontiguousarray(faces_np, dtype=np.int32))
        uvs = torch.from_numpy(np.ascontiguousarray(uvs_np, dtype=np.float32))
        if debug_charts:
            face_chart_ids = torch.from_numpy(
                np.ascontiguousarray(face_chart_ids_np, dtype=np.int32)
            )
        else:
            face_chart_ids = None
        log_completed(
            "gather packed mesh",
            phase_started,
            f"vertices={vmaps.shape[0]}, faces={faces.shape[0]}",
        )

        try:
            validation = _validate_uv_output(atlas, uvs, faces)
        except Exception as error:
            validation = {
                "valid": False,
                "issue": f"validation_failed:{type(error).__name__}",
                "out_of_range_values": 0,
            }

        used_source_vertex_parts = []
        for chart_vertices_i, chart_faces_i, chart_vmap_i in charts:
            chart_faces_clean, _ = _safe_chart_faces(chart_faces_i, len(chart_vertices_i))
            chart_vmap_i = np.asarray(chart_vmap_i, dtype=np.int64)
            if len(chart_faces_clean) and len(chart_vmap_i):
                used_source_vertex_parts.append(
                    chart_vmap_i[chart_faces_clean].reshape(-1)
                )
        source_vertex_ids = (
            np.unique(np.concatenate(used_source_vertex_parts))
            if used_source_vertex_parts
            else np.empty((0,), dtype=np.int32)
        )
        mapped_vertex_ids = np.unique(vmaps_np) if len(vmaps_np) else np.empty((0,), dtype=np.int32)
        unmapped_vertices = int(np.setdiff1d(source_vertex_ids, mapped_vertex_ids).size)
        expected_face_count = len(packed_faces) // 3
        missing_faces = max(0, expected_face_count - int(faces.shape[0]))
        atlas_complete = (
            missing_faces == 0
            and dropped_charts == 0
            and dropped_faces == 0
            and unmapped_vertices == 0
        )

        warning_levels = []
        if native_failed_chart_count:
            warning_levels.append(_emit_uv_warning(
                "some charts did not produce native UVs; planar fallbacks were used",
                native_failed_chart_count,
                num_charts,
            ))
        if emergency_fallback_charts:
            warning_levels.append(_emit_uv_warning(
                "emergency planar UV fallback was required",
                emergency_fallback_charts,
                num_charts,
            ))
        if dropped_charts:
            warning_levels.append(_emit_uv_warning(
                "complete charts could not be represented and were omitted",
                dropped_charts,
                num_charts,
                critical=True,
            ))
        if dropped_faces or missing_faces:
            warning_levels.append(_emit_uv_warning(
                "some mesh faces could not be assigned UVs",
                max(dropped_faces, missing_faces),
                max(expected_face_count, 1),
                critical=True,
            ))
        if unmapped_vertices:
            warning_levels.append(_emit_uv_warning(
                "some chart vertices were not present in the returned mapping",
                unmapped_vertices,
                max(len(source_vertex_ids), 1),
                critical=True,
            ))
        if atlas_add_failures:
            warning_levels.append(_emit_uv_warning(
                "some charts could not be added to the atlas",
                atlas_add_failures,
                num_charts,
                critical=True,
            ))
        if overlap_charts:
            warning_levels.append(_emit_uv_warning(
                "self-overlapping UV chart regions were split before packing",
                overlap_faces,
                max(expected_face_count, 1),
            ))
        if overlap_detection_limited_charts:
            warning_levels.append(_emit_uv_warning(
                "an overlap graph was too dense; conservative per-face chart splitting was used",
                overlap_detection_limited_charts,
                num_charts,
                critical=True,
            ))
        if overlap_detection_failures:
            warning_levels.append(_emit_uv_warning(
                "some charts could not be checked for self-overlap",
                overlap_detection_failures,
                num_charts,
                critical=True,
            ))
        if distortion_optimization_failures:
            warning_levels.append(_emit_uv_warning(
                "some charts could not be distortion-optimized; their valid UVs were kept",
                distortion_optimization_failures,
                max(len(chart_records), 1),
            ))
        if distortion_optimization_unavailable:
            warning_levels.append(_emit_uv_warning(
                "the native distortion optimizer is unavailable; valid initial UVs were kept",
                1,
                1,
            ))
        if packing_used_fallback:
            warning_levels.append(_emit_uv_warning(
                "the native atlas packer was unavailable; a simple grid pack was used",
                1,
                1,
            ))
        if not validation["valid"]:
            warning_levels.append(_emit_uv_warning(
                f"final UV validation failed ({validation.get('issue', 'unknown')})",
                1,
                1,
                critical=True,
            ))

        chart_texture = None
        if debug_charts:
            try:
                texture_width = max(1, int(atlas_info["width"])) * columns
                texture_height = max(1, int(atlas_info["height"])) * rows
                chart_texture = _rasterize_chart_texture(
                    uvs,
                    faces,
                    face_chart_ids,
                    texture_width,
                    texture_height,
                )
            except Exception as error:
                _emit_uv_warning(
                    f"debug chart texture generation failed ({type(error).__name__}); "
                    "returning a blank debug texture",
                    affected=1,
                    total=1,
                )
                chart_texture = torch.full(
                    (max(1, int(atlas_info.get("height", 1))),
                     max(1, int(atlas_info.get("width", 1))), 4),
                    255,
                    dtype=torch.uint8,
                )

        try:
            vertices = new_vertices.cpu()[vmaps.long()]
        except Exception as error:
            return _emergency_full_mesh_result(
                new_vertices,
                new_faces,
                debug_charts,
                return_vmaps,
                return_stats,
                flatten_methods,
                f"final vertex mapping failed ({type(error).__name__}); "
                "returned a planar emergency UV mesh",
            )
        overview_started = perf_counter()
        native_fallback_attempts = int(native_stats.get("fallback_charts", 0))
        native_failed_charts = int(native_stats.get("failed_charts", 0))
        stats = {
            "chart_count": num_charts,
            "packed_chart_count": len(chart_records),
            "debug_charts": bool(debug_charts),
            "flatten": list(flatten_methods),
            "local_splits": local_splits,
            "python_fallback_charts": python_fallback_charts,
            "project_charts": project_charts,
            "emergency_fallback_charts": emergency_fallback_charts,
            "native_failed_charts": native_failed_chart_count,
            "native_lscm_charts": native_lscm_charts,
            "native_graph_fallback_charts": native_graph_fallback_charts,
            "final_parameterization_methods": {
                "native_lscm": native_lscm_charts,
                "python_project": project_charts,
                "native_graph_fallback": native_graph_fallback_charts,
                "python_pca_fallback": python_fallback_charts,
            },
            "geometry_repairs": geometry_repairs,
            "dropped_charts": dropped_charts,
            "dropped_faces": dropped_faces,
            "missing_faces": missing_faces,
            "unmapped_vertices": unmapped_vertices,
            "atlas_add_failures": atlas_add_failures,
            "self_overlapping_charts": overlap_charts,
            "self_overlap_faces": overlap_faces,
            "self_overlap_pairs": overlap_pairs,
            "self_overlap_candidates": overlap_candidates,
            "overlap_split_charts": overlap_split_charts,
            "overlap_detection_failures": overlap_detection_failures,
            "overlap_detection_limited_charts": overlap_detection_limited_charts,
            "distortion_optimization_attempts": distortion_optimization_attempts,
            "distortion_optimized_charts": distortion_optimized_charts,
            "distortion_optimization_failures": distortion_optimization_failures,
            "distortion_optimization_unavailable": distortion_optimization_unavailable,
            "distortion_optimization_seconds": distortion_optimization_seconds,
            "atlas_complete": atlas_complete,
            "atlas_pack_fallback": packing_used_fallback,
            "uv_warning_level": (
                "ERROR" if "ERROR" in warning_levels
                else ("WARNING" if warning_levels else "OK")
            ),
            "parameterization_seconds": parameterization_seconds,
            "packing_seconds": packing_seconds,
            "xatlas_atlas_count": atlas_info["atlas_count"],
            "uv_validation_passed": bool(validation["valid"]),
            **native_stats,
        }

        log_completed(
            "chart method overview",
            overview_started,
            f"total={num_charts}, native_lscm={native_lscm_charts}, "
            f"native_graph_fallback={native_graph_fallback_charts} "
            f"(attempts={native_fallback_attempts}, failed={native_failed_charts}), "
            f"python_project_charts={project_charts}, "
            f"python_pca_fallback={python_fallback_charts}, "
            f"geometry_repairs={geometry_repairs}, local_splits={local_splits}",
        )

        out = [vertices, faces, uvs, chart_texture]
        if return_vmaps:
            out.append(vmaps)
        if return_stats:
            out.append(stats)
        return tuple(out)


