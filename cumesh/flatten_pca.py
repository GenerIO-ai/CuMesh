"""PCA-based chart flattening fallback used by :meth:`CuMesh.uv_unwrap`."""

import numpy as np


def project_chart_uvs(vertices):
    """Create a winding-independent planar projection using PCA.

    This is the last-resort flattening method.  The fallback projection for
    malformed or zero-span input is deliberately based only on vertex order;
    it does not infer or enforce a face winding.
    """
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

    indices = np.arange(len(points), dtype=np.float64)
    return np.column_stack((indices, (indices % 2) * 1e-3))
