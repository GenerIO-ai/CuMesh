"""Native LSCM chart flattening stage.

The native implementation can optionally continue into the Tutte graph
fallback for charts for which LSCM does not produce a valid result.  Keeping
that option here makes the default one-pass pipeline retain its existing
behaviour while allowing :func:`cumesh.CuMesh.uv_unwrap` to disable either
stage explicitly.
"""


def parameterize_charts(
    atlas,
    vertices,
    faces,
    vertex_offsets,
    face_offsets,
    progress_callback=None,
    trace_callback=None,
    *,
    flatten_tutte=True,
):
    """Flatten packed charts with LSCM and, optionally, Tutte fallback."""
    return atlas._parameterize_lscm_batch(
        vertices,
        faces,
        vertex_offsets,
        face_offsets,
        progress_callback,
        trace_callback,
        flatten_lscm=True,
        flatten_tutte=flatten_tutte,
    )
