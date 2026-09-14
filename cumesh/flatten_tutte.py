"""Native Tutte graph flattening stage.

The native stage replaces the circular boundary with a projected convex-hull
boundary. The nonlinear distortion improvement pass is kept as an explicit native
switch so the projected-boundary Tutte result can be evaluated on its own.
"""


def parameterize_charts(
    atlas,
    vertices,
    faces,
    vertex_offsets,
    face_offsets,
    progress_callback=None,
    trace_callback=None,
):
    """Flatten packed charts with Tutte, without attempting LSCM."""
    return atlas._parameterize_lscm_batch(
        vertices,
        faces,
        vertex_offsets,
        face_offsets,
        progress_callback,
        trace_callback,
        flatten_lscm=False,
        flatten_tutte=True,
    )
