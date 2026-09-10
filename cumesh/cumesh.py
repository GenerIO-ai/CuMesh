from typing import *
import math
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
from .xatlas import Atlas, parameterize_lscm
from . import _C


def _split_edge_connected_components(cv_np, cf_np, cvmap_np):
    num_f = len(cf_np)
    if num_f <= 1:
        return [(cv_np, cf_np, cvmap_np)]
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(cf_np):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_to_faces[tuple(sorted(e))].append(fi)

    parent = list(range(num_f))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for flist in edge_to_faces.values():
        if len(flist) > 1:
            for f_other in flist[1:]:
                union(flist[0], f_other)

    comp_map = defaultdict(list)
    for fi in range(num_f):
        comp_map[find(fi)].append(fi)

    if len(comp_map) == 1:
        return [(cv_np, cf_np, cvmap_np)]

    out_pieces = []
    for flist in comp_map.values():
        sub_cf = cf_np[flist]
        u_v, inv_v = np.unique(sub_cf, return_inverse=True)
        remapped_cf = inv_v.reshape(sub_cf.shape).astype(np.int32)
        remapped_cv = cv_np[u_v]
        remapped_vmap = cvmap_np[u_v]
        out_pieces.append((remapped_cv, remapped_cf, remapped_vmap))
    return out_pieces


def _detach_chart_filaments(cv_np, cf_np, cvmap_np, core_depth_thresh=6, min_filament_length=15):
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(cf_np):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_to_faces[tuple(sorted(e))].append(fi)
    adj = defaultdict(list)
    bnd_faces = set()
    internal_edges = []
    for e, flist in edge_to_faces.items():
        if len(flist) == 2:
            adj[flist[0]].append(flist[1])
            adj[flist[1]].append(flist[0])
            internal_edges.append((flist[0], flist[1]))
        elif len(flist) == 1:
            bnd_faces.add(flist[0])

    depth = np.full(len(cf_np), -1, dtype=np.int32)
    q = list(bnd_faces)
    for bf in bnd_faces: depth[bf] = 0
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        d = depth[u]
        for v in adj[u]:
            if depth[v] == -1: depth[v] = d + 1; q.append(v)

    core_faces = set(np.where(depth >= core_depth_thresh)[0])
    if len(core_faces) == 0:
        return [(cv_np, cf_np, cvmap_np)]

    dist_from_core = np.full(len(cf_np), -1, dtype=np.int32)
    q_core = list(core_faces)
    for cf_idx in core_faces: dist_from_core[cf_idx] = 0
    head = 0
    while head < len(q_core):
        u = q_core[head]; head += 1
        d = dist_from_core[u]
        for v in adj[u]:
            if dist_from_core[v] == -1: dist_from_core[v] = d + 1; q_core.append(v)

    is_filament = (dist_from_core >= min_filament_length) & (depth <= 2)
    if not np.any(is_filament):
        return [(cv_np, cf_np, cvmap_np)]

    sub_0, sub_1 = [], []
    for f0, f1 in internal_edges:
        if is_filament[f0] == is_filament[f1]:
            sub_0.append(f0); sub_1.append(f1)

    g = sp.csr_matrix((np.ones(len(sub_0)), (sub_0, sub_1)), shape=(len(cf_np), len(cf_np)))
    n_c, labels = sp.csgraph.connected_components(g, directed=False)

    out_pieces = []
    for lbl in range(n_c):
        p_idx = np.where(labels == lbl)[0]
        if len(p_idx) > 0:
            sub_cf = cf_np[p_idx]
            u_v, inv_v = np.unique(sub_cf, return_inverse=True)
            remapped_cf = inv_v.reshape(sub_cf.shape).astype(np.int32)
            remapped_cv = cv_np[u_v]
            remapped_vmap = cvmap_np[u_v]
            out_pieces.append((remapped_cv, remapped_cf, remapped_vmap))
    return out_pieces


def _split_bending_ribbons(cv_np, cf_np, cvmap_np, max_angle_span_deg=100.0):
    p0 = cv_np[cf_np[:, 0]]; p1 = cv_np[cf_np[:, 1]]; p2 = cv_np[cf_np[:, 2]]
    cross_3d = np.cross(p1 - p0, p2 - p0)
    a3d = 0.5 * np.sum(np.linalg.norm(cross_3d, axis=1))

    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(cf_np):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_to_faces[tuple(sorted(e))].append(fi)
    bnd_e = [e for e, flist in edge_to_faces.items() if len(flist) == 1]
    if len(bnd_e) == 0:
        return [(cv_np, cf_np, cvmap_np)]
    bnd_len = np.sum(np.linalg.norm(cv_np[[e[0] for e in bnd_e]] - cv_np[[e[1] for e in bnd_e]], axis=1))
    iq = 4 * np.pi * a3d / (bnd_len**2 + 1e-12)

    if iq >= 0.05:
        return [(cv_np, cf_np, cvmap_np)]

    center_xz = np.mean(cv_np[:, [0, 2]], axis=0)
    angles = np.arctan2(cv_np[:, 2] - center_xz[1], cv_np[:, 0] - center_xz[0])
    ang_span = np.ptp(angles)
    if ang_span <= np.radians(max_angle_span_deg):
        return [(cv_np, cf_np, cvmap_np)]

    num_splits = int(np.ceil(ang_span / np.radians(80)))
    f_centers_xz = np.mean(cv_np[cf_np, :][:, :, [0, 2]], axis=1)
    f_angles = np.arctan2(f_centers_xz[:, 1] - center_xz[1], f_centers_xz[:, 0] - center_xz[0])
    f_angles_norm = (f_angles - np.min(angles)) % (2 * np.pi)
    bin_size = (2 * np.pi) / num_splits
    face_bins = np.clip((f_angles_norm / bin_size).astype(int), 0, num_splits - 1)

    out_pieces = []
    for b in range(num_splits):
        b_idx = np.where(face_bins == b)[0]
        if len(b_idx) > 0:
            sub_cf = cf_np[b_idx]
            u_v, inv_v = np.unique(sub_cf, return_inverse=True)
            remapped_cf = inv_v.reshape(sub_cf.shape).astype(np.int32)
            remapped_cv = cv_np[u_v]
            remapped_vmap = cvmap_np[u_v]
            out_pieces.append((remapped_cv, remapped_cf, remapped_vmap))
    if len(out_pieces) == 0:
        return [(cv_np, cf_np, cvmap_np)]
    return out_pieces


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
        refine_iterations: int=100,
        global_iterations: int=3,
        smooth_strength: float=1,
        area_penalty_weight: float=0.1,
        perimeter_area_ratio_weight: float=0.0001,
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
        compute_charts_kwargs: dict = {},
        xatlas_compute_charts_kwargs: dict = {},
        xatlas_pack_charts_kwargs: dict = {},
        return_vmaps: bool = False,
        verbose: bool = False,
        preserve_cumesh_charts: bool = False,
        micro_chart_cleanup_kwargs: dict = {},
        return_stats: bool = False,
    ):
        xatlas_compute_charts_kwargs['verbose'] = verbose
        xatlas_pack_charts_kwargs['verbose'] = verbose

        self.remove_degenerate_faces()

        if not preserve_cumesh_charts:
            self.compute_charts(**compute_charts_kwargs)
            new_vertices, new_faces = self.read()
            num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.read_atlas_charts()
            chart_vertices = new_vertices[chart_vmap].cpu()
            chart_faces = chart_faces.cpu()
            chart_vertex_offset = chart_vertex_offset.cpu()
            chart_face_offset = chart_face_offset.cpu()
            chart_vmap = chart_vmap.cpu()
            if verbose:
                print(f"Get {num_charts} clusters after fast clustering")

            xatlas = Atlas()
            chart_vmaps = []
            for i in tqdm(range(num_charts), desc="Adding clusters to xatlas", disable=not verbose):
                chart_faces_i = chart_faces[chart_face_offset[i]:chart_face_offset[i+1]] - chart_vertex_offset[i]
                chart_vertices_i = chart_vertices[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
                chart_vmap_i = chart_vmap[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
                chart_vmaps.append(chart_vmap_i)
                xatlas.add_mesh(chart_vertices_i, chart_faces_i)
            xatlas.compute_charts(**xatlas_compute_charts_kwargs)
            xatlas.pack_charts(**xatlas_pack_charts_kwargs)
            vmaps = []
            faces = []
            uvs = []
            cnt = 0
            for i in tqdm(range(num_charts), desc="Gathering results from xatlas", disable=not verbose):
                vmap, x_faces, x_uvs = xatlas.get_mesh(i)
                vmaps.append(chart_vmaps[i][vmap])
                faces.append(x_faces + cnt)
                uvs.append(x_uvs)
                cnt += vmap.shape[0]
            vmaps = torch.cat(vmaps, dim=0)
            vertices = new_vertices.cpu()[vmaps]
            faces = torch.cat(faces, dim=0)
            uvs = torch.cat(uvs, dim=0)

            out = [vertices, faces, uvs]
            if return_vmaps:
                out.append(vmaps)
            if return_stats:
                out.append({})
            return tuple(out)


        compute_kwargs = dict(compute_charts_kwargs)
        compute_kwargs.setdefault("refine_iterations", 0)
        compute_kwargs.setdefault("area_penalty_weight", 0.0)
        compute_kwargs.setdefault("perimeter_area_ratio_weight", 0.0005)

        self.compute_charts(**compute_kwargs)
        initial_num_charts, _, _, _, _, _ = self.read_atlas_charts()

        new_vertices, new_faces = self.read()
        cleanup_kwargs = dict(micro_chart_cleanup_kwargs)
        total_mesh_faces = int(new_faces.shape[0])
        cleanup_kwargs.setdefault("min_faces", min(128, max(4, int(total_mesh_faces * 0.005))))
        merges = self.merge_micro_charts(**cleanup_kwargs)

        new_vertices, new_faces = self.read()
        num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.read_atlas_charts()

        chart_vertices = new_vertices[chart_vmap].cpu().numpy()
        chart_faces = chart_faces.cpu().numpy()
        chart_vertex_offset = chart_vertex_offset.cpu().numpy()
        chart_face_offset = chart_face_offset.cpu().numpy()
        chart_vmap = chart_vmap.cpu().numpy()

        final_charts = []
        for i in range(num_charts):
            cf_i = chart_faces[chart_face_offset[i]:chart_face_offset[i+1]] - chart_vertex_offset[i]
            cv_i = chart_vertices[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            cvmap_i = chart_vmap[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            for p1 in _split_edge_connected_components(cv_i, cf_i, cvmap_i):
                for p2 in _detach_chart_filaments(*p1):
                    for p3 in _split_bending_ribbons(*p2):
                        for p4 in _split_edge_connected_components(*p3):
                            final_charts.append(p4)

        xatlas = Atlas()
        lscm_failures = 0
        local_splits = 0
        chart_vmaps = []

        for i, (cv_np, cf_np, cvmap_np) in enumerate(tqdm(final_charts, desc="LSCM parameterizing charts", disable=not verbose)):
            cv_t = torch.from_numpy(cv_np).float()
            cf_t = torch.from_numpy(cf_np).int()

            p0 = cv_np[cf_np[:, 0]]; p1 = cv_np[cf_np[:, 1]]; p2 = cv_np[cf_np[:, 2]]
            a3d = 0.5 * np.sum(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1))

            uvs_i, faces_i, local_vmap_i, success_i, splits_i = parameterize_lscm(cv_t, cf_t)
            if not success_i or len(faces_i) == 0:
                lscm_failures += 1
                cross_3d = np.cross(p1 - p0, p2 - p0)
                norm = np.sum(cross_3d, axis=0)
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-8:
                    norm = norm / norm_len
                else:
                    norm = np.array([0.0, 0.0, 1.0])
                up = np.array([0.0, 1.0, 0.0]) if abs(norm[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
                tangent = np.cross(norm, up)
                tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                bitangent = np.cross(norm, tangent)
                u_proj = np.dot(cv_np, tangent)
                v_proj = np.dot(cv_np, bitangent)
                uvs_i = torch.from_numpy(np.stack([u_proj, v_proj], axis=1)).float()
                faces_i = torch.from_numpy(cf_np).int()
                local_vmap_i = torch.arange(len(cv_np), dtype=torch.int32)
            else:
                local_splits += splits_i

            u0 = uvs_i[faces_i[:, 0]].numpy()
            u1 = uvs_i[faces_i[:, 1]].numpy()
            u2 = uvs_i[faces_i[:, 2]].numpy()
            cross = (u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) - (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0])
            auv = 0.5 * np.sum(np.abs(cross))

            if auv > 1e-12:
                scale = np.sqrt(a3d / auv)
                uvs_i = uvs_i * scale

            orig_vmap_i = torch.from_numpy(cvmap_np[local_vmap_i.long().numpy()])
            chart_vmaps.append(orig_vmap_i)
            xatlas.add_uv_mesh(uvs_i, faces_i)

        pack_opts = dict(xatlas_pack_charts_kwargs)
        pack_opts.setdefault("padding", 4)
        xatlas.pack_charts(**pack_opts)

        vmaps = []
        faces = []
        uvs = []
        cnt = 0
        for i in range(len(final_charts)):
            vmap_pk, x_faces, x_uvs = xatlas.get_mesh(i)
            vmaps.append(chart_vmaps[i][vmap_pk.long()])
            faces.append(x_faces + cnt)
            uvs.append(x_uvs)
            cnt += vmap_pk.shape[0]

        vmaps = torch.cat(vmaps, dim=0)
        vertices = new_vertices.cpu()[vmaps]
        faces = torch.cat(faces, dim=0)
        uvs = torch.cat(uvs, dim=0)

        stats = {
            "initial_chart_count": initial_num_charts,
            "after_cleanup_chart_count": num_charts,
            "merges": merges,
            "lscm_failures": lscm_failures,
            "local_splits": local_splits,
            "final_island_count": len(final_charts),
        }

        if verbose:
            print(f"CuMesh UV Unwrap Preserved: Initial={initial_num_charts}, AfterCleanup={num_charts}, Merges={merges}, LSCM Failures={lscm_failures}, Splits={local_splits}, FinalIslands={len(final_charts)}")

        out = [vertices, faces, uvs]
        if return_vmaps:
            out.append(vmaps)
        if return_stats:
            out.append(stats)

        return tuple(out)

