"""Fast normal-median planar projection for chart flattening."""

import numpy as np


def project_chart_uvs(vertices, faces):
    """Project a chart onto the plane of its median triangle normal.

    Triangle normal signs are aligned to the first valid normal before taking
    the component-wise median. This makes opposite face winding equivalent
    while keeping the operation linear and allocation-light.
    """
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    try:
        face_np = np.asarray(faces, dtype=np.int64)
    except (TypeError, ValueError):
        face_np = np.empty((0, 3), dtype=np.int64)
    if face_np.ndim == 2 and face_np.shape[1] == 3:
        valid = np.all((face_np >= 0) & (face_np < len(points)), axis=1)
        valid &= ~(
            (face_np[:, 0] == face_np[:, 1])
            | (face_np[:, 0] == face_np[:, 2])
            | (face_np[:, 1] == face_np[:, 2])
        )
        face_np = face_np[valid]
    else:
        face_np = np.empty((0, 3), dtype=np.int64)

    normal = None
    if len(face_np):
        edge_a = points[face_np[:, 1]] - points[face_np[:, 0]]
        edge_b = points[face_np[:, 2]] - points[face_np[:, 0]]
        normals = np.cross(edge_a, edge_b)
        lengths = np.sqrt(np.sum(normals * normals, axis=1))
        valid_normals = np.isfinite(lengths) & (lengths > 1e-12)
        if np.any(valid_normals):
            normals = normals[valid_normals]
            normals /= lengths[valid_normals, None]

            # Treat n and -n as the same axis. Aligning to one reference
            # hemisphere is enough for a fast, winding-independent median.
            reference = normals[0].copy()
            signs = np.sum(normals * reference, axis=1) < 0.0
            normals[signs] *= -1.0
            normal = np.median(normals, axis=0)
            normal_length = float(np.linalg.norm(normal))
            if not np.isfinite(normal_length) or normal_length <= 1e-6:
                normal = np.mean(normals, axis=0)
                normal_length = float(np.linalg.norm(normal))
            if np.isfinite(normal_length) and normal_length > 1e-6:
                normal = normal / normal_length
            else:
                normal = None

    if normal is None:
        # Degenerate charts still receive a deterministic finite projection.
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    abs_normal = np.abs(normal)
    reference_axis = int(np.argmin(abs_normal))
    reference_axis_vector = np.zeros(3, dtype=np.float32)
    reference_axis_vector[reference_axis] = 1.0
    tangent = np.cross(reference_axis_vector, normal)
    tangent_length = float(np.linalg.norm(tangent))
    if not np.isfinite(tangent_length) or tangent_length <= 1e-6:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tangent -= normal * np.dot(tangent, normal)
        tangent_length = float(np.linalg.norm(tangent))
    tangent /= max(tangent_length, 1e-12)
    bitangent = np.cross(normal, tangent)

    centered = points - np.mean(points, axis=0, dtype=np.float32)
    projected = np.empty((len(points), 2), dtype=np.float32)
    projected[:, 0] = centered @ tangent
    projected[:, 1] = centered @ bitangent
    if not np.isfinite(projected).all():
        return np.column_stack((
            np.arange(len(points), dtype=np.float32),
            (np.arange(len(points), dtype=np.float32) % 2) * 1e-3,
        ))
    return projected
