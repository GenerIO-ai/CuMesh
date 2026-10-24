"""Aggressively group small islands before overlap splitting.

Only the selected flattener is used. A failed group keeps its original UVs;
overlap detection and splitting belong to the following pipeline stage.
"""

import heapq
from time import perf_counter

import numpy as np
import torch


def _edge_pairs(faces):
    """Connect every incidence of an edge, including non-manifold edges.

    A star connects all incident faces in linear space, rather than enumerating
    every pair on edges shared by many faces. Collapsed edges are ignored.
    """
    count = len(faces)
    if not count:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty, 0
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges.astype(np.uint64), axis=1)
    if edges.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError("UV repair exceeds 32-bit vertex indexing")
    keys = (edges[:, 0] << np.uint64(32)) | edges[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    roots = np.repeat(order[starts], lengths)
    usable = (order != roots) & (order % count != roots % count)
    usable &= edges[order, 0] != edges[order, 1]
    first, second = roots[usable], order[usable]
    non_manifold = (lengths > 2) & (edges[order[starts], 0] != edges[order[starts], 1])
    return first % count, second % count, first // count, second // count, int(non_manifold.sum())


def _connected_labels(count, first, second):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(first), dtype=np.uint8), (first, second)), shape=(count, count)).tocsr()
    return connected_components(graph, directed=False, return_labels=True)[1]


def _position_vertices(positions, tolerance=0.0):
    """Canonicalize endpoint positions only for the parameterization mesh."""
    points, inverse = np.unique(positions, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if tolerance <= 0 or len(points) < 2:
        return inverse
    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    if not len(pairs):
        return inverse
    labels = _connected_labels(len(points), pairs[:, 0], pairs[:, 1])
    return labels[inverse]


def _uv_labels(positions, uvs, faces, vertex_records=None, geometry=None):
    """Find islands joined through matching surface endpoints and UV edges."""
    columns = [positions if geometry is None else geometry[:, None], uvs]
    if vertex_records is not None:
        columns.append(vertex_records[:, None])
    _, vertices = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    first, second, _, _, _ = _edge_pairs(vertices.reshape(-1)[faces])
    return _connected_labels(len(faces), first, second)


def count_uv_islands(positions, uvs, faces, minimum, position_tolerance=0.0):
    positions = np.asarray(positions)
    geometry = _position_vertices(positions, position_tolerance) if position_tolerance > 0 else None
    labels = _uv_labels(positions, np.asarray(uvs), np.asarray(faces), geometry=geometry)
    sizes = np.bincount(labels)
    return len(sizes), int(np.count_nonzero(sizes < minimum))


def _ordered_groups(labels):
    order = np.argsort(labels, kind="stable")
    starts = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1]])
    return np.split(order, starts[1:])


def repair_small_islands(atlas, records, minimum, flatten_chart):
    """Merge smallest adjacent groups, then flatten each affected group once.

    ``flatten_chart(positions, faces)`` returns UVs, faces and a vertex map into
    the supplied positions. Overlaps are deliberately accepted here. Failure
    of a merged group's selected flattener retains its original islands.
    """
    started = perf_counter()
    stats = {
        "repair_islands_before": 0,
        "repair_islands_after": 0,
        "repair_undersized_before": 0,
        "repair_undersized_after": 0,
        "repair_merges": 0,
        "repair_reparameterized_groups": 0,
        "repair_failed_groups": 0,
        "repair_unmergeable_islands": 0,
        "repair_non_manifold_edges": 0,
        "repair_position_tolerance": 0.0,
    }
    if not records:
        return [], {**stats, "repair_seconds": perf_counter() - started}
    vertex_counts = np.array([len(record["uvs"]) for record in records], dtype=np.int64)
    face_counts = np.array([len(record["faces"]) for record in records], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(vertex_counts)]
    positions = np.ascontiguousarray(np.concatenate([record["positions"] for record in records]), dtype=np.float32)
    uvs = np.ascontiguousarray(np.concatenate([record["uvs"] for record in records]), dtype=np.float32)
    vmaps = np.concatenate([record["vmaps"] for record in records])
    faces = np.concatenate([record["faces"].astype(np.int64) + offsets[i] for i, record in enumerate(records)])
    if not len(faces) or not np.isfinite(positions).all() or not np.isfinite(uvs).all():
        raise ValueError("UV island repair requires nonempty, finite chart geometry")

    # Match duplicated or almost-identical endpoints without modifying source
    # positions. Both endpoints of an edge must match; point contact is unused.
    tolerance = float(np.linalg.norm(np.ptp(positions.astype(np.float64), axis=0)) * 1e-6)
    geometry = _position_vertices(positions, tolerance)
    _, representatives = np.unique(geometry, return_index=True)
    vertex_records = np.repeat(np.arange(len(records)), vertex_counts)
    face_records = np.repeat(np.arange(len(records)), face_counts)
    labels = _uv_labels(positions, uvs, faces, vertex_records, geometry)
    members = [[group] for group in _ordered_groups(labels)]
    sizes = np.bincount(labels).astype(np.int64)
    before = len(sizes)
    stats["repair_islands_before"] = before
    stats["repair_undersized_before"] = int(np.count_nonzero(sizes < minimum))
    stats["repair_position_tolerance"] = tolerance
    first, second, edge_slots, _, non_manifold = _edge_pairs(geometry[faces])
    stats["repair_non_manifold_edges"] = non_manifold
    edge_start = positions[faces[first, edge_slots]].astype(np.float64)
    edge_end = positions[faces[first, (edge_slots + 1) % 3]].astype(np.float64)
    lengths = np.linalg.norm(edge_end - edge_start, axis=1)

    # Build adjacency once. No normal, coplanarity, winding, overlap or
    # manifoldness exclusions are imposed on shared-edge connections.
    a, b = labels[first], labels[second]
    separate = a != b
    pairs = np.sort(np.column_stack((a[separate], b[separate])), axis=1)
    adjacency = [dict() for _ in sizes]
    if len(pairs):
        keys = pairs[:, 0].astype(np.uint64) << np.uint64(32) | pairs[:, 1].astype(np.uint64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        weights = np.add.reduceat(lengths[separate][order], starts)
        for (left, right), weight in zip(pairs[order[starts]], weights):
            adjacency[int(left)][int(right)] = float(weight)
            adjacency[int(right)][int(left)] = float(weight)

    active = np.ones(before, dtype=bool)
    pending = [(int(size), i) for i, size in enumerate(sizes) if size < minimum]
    heapq.heapify(pending)
    while pending:
        size, source = heapq.heappop(pending)
        if not active[source] or sizes[source] != size or size >= minimum:
            continue
        if not adjacency[source]:
            stats["repair_unmergeable_islands"] += 1
            continue
        target = min(adjacency[source], key=lambda n: (
            sizes[n] >= minimum, int(sizes[n]), -adjacency[source][n], n,
        ))
        active[source] = False
        sizes[target] += sizes[source]
        members[target].extend(members[source])
        members[source] = []
        adjacency[target].pop(source, None)
        for neighbor, weight in tuple(adjacency[source].items()):
            adjacency[neighbor].pop(source, None)
            if neighbor == target:
                continue
            combined = adjacency[target].get(neighbor, 0.0) + weight
            adjacency[target][neighbor] = combined
            adjacency[neighbor][target] = combined
        adjacency[source].clear()
        if sizes[target] < minimum:
            heapq.heappush(pending, (int(sizes[target]), target))

    output = []

    def append_record(record, face_ids):
        source = records[int(face_records[face_ids[0]])]
        record["chart_index"] = len(output)
        record["source_chart_index"] = source["source_chart_index"]
        labels = _uv_labels(record["positions"], record["uvs"], record["faces"],
                            geometry=record["repair_vertices"])
        island_sizes = np.bincount(labels)
        stats["repair_islands_after"] += len(island_sizes)
        stats["repair_undersized_after"] += int(np.count_nonzero(island_sizes < minimum))
        output.append(record)

    def append_original(face_ids):
        used, indices = np.unique(faces[face_ids], return_inverse=True)
        append_record({
            "uvs": np.ascontiguousarray(uvs[used]),
            "faces": np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32),
            "positions": np.ascontiguousarray(positions[used]),
            "vmaps": np.ascontiguousarray(vmaps[used], dtype=np.int32),
            "repair_vertices": geometry[used],
        }, face_ids)

    for group in np.flatnonzero(active):
        if len(members[group]) == 1:
            append_original(members[group][0])
            continue
        face_ids = np.concatenate(members[group])
        corners = faces[face_ids]
        # A temporary welded mesh supplies shared variables to the selected
        # flattener. Restore every original source face corner afterwards.
        used, indices = np.unique(geometry[corners], return_inverse=True)
        points = np.ascontiguousarray(positions[representatives[used]])
        work_faces = np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32)
        try:
            new_uvs, new_faces, mapping = flatten_chart(points, work_faces)
            new_uvs = np.asarray(new_uvs, dtype=np.float32)
            new_faces = np.asarray(new_faces, dtype=np.int64)
            mapping = np.asarray(mapping, dtype=np.int64).reshape(-1)
            if (new_uvs.shape != (len(mapping), 2) or not np.isfinite(new_uvs).all()
                    or new_faces.shape != work_faces.shape
                    or np.any(new_faces < 0) or np.any(new_faces >= len(mapping))
                    or np.any(mapping < 0) or np.any(mapping >= len(points))
                    or not np.array_equal(mapping[new_faces], work_faces)):
                raise ValueError("Reparameterization changed the source face-corner mapping or produced invalid UVs")
            p, q, r = (new_uvs[new_faces[:, i]].astype(np.float64) for i in range(3))
            area = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
            if np.any(area == 0):
                raise ValueError("Reparameterization collapsed a triangle")
            source_corners = corners.reshape(-1)
            keys = np.column_stack((vmaps[source_corners], new_faces.reshape(-1)))
            unique, first_corner, indices = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            record = {
                "uvs": np.ascontiguousarray(new_uvs[unique[:, 1]]),
                "faces": np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32),
                "positions": np.ascontiguousarray(positions[source_corners[first_corner]]),
                "vmaps": np.ascontiguousarray(unique[:, 0], dtype=np.int32),
                "repair_vertices": geometry[source_corners[first_corner]],
            }
        except Exception:
            stats["repair_failed_groups"] += 1
            for original in members[group]:
                append_original(original)
            continue
        append_record(record, face_ids)
        stats["repair_reparameterized_groups"] += 1
        stats["repair_merges"] += len(members[group]) - 1

    stats["repair_seconds"] = perf_counter() - started
    return output, stats


def optimize_repaired_island(atlas, record):
    """Keep shared UV variables together while optimizing distortion."""
    geometry = record.get("repair_vertices")
    columns = [record["positions"] if geometry is None else geometry[:, None], record["uvs"]]
    _, first, inverse = np.unique(np.column_stack(columns), axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    positions = np.ascontiguousarray(record["positions"][first])
    uvs = np.ascontiguousarray(record["uvs"][first]).copy()
    faces = np.ascontiguousarray(inverse[record["faces"]], dtype=np.int32)
    changed = bool(atlas._optimize_uv_distortion(
        torch.from_numpy(positions), torch.from_numpy(uvs), torch.from_numpy(faces),
    ))
    if not changed or not np.isfinite(uvs).all():
        return False
    record["uvs"] = np.ascontiguousarray(uvs[inverse])
    return True
