#include "cumesh.h"
#include "dtypes.cuh"
#include "shared.h"

#include <cub/cub.cuh>
#include <cmath>
#include <cstdint>
#include <limits>


namespace cumesh {


__device__ inline bool normalize_face_contains(const int3& face, int vertex) {
    return face.x == vertex || face.y == vertex || face.z == vertex;
}


__device__ inline float normalize_face_area(const float3* vertices, const int3& face) {
    Vec3f p0(vertices[face.x]);
    Vec3f p1(vertices[face.y]);
    Vec3f p2(vertices[face.z]);
    return 0.5f * (p1 - p0).cross(p2 - p0).norm();
}


__device__ inline float normalize_longest_edge_squared(
    const float3* vertices,
    const int3& face
) {
    Vec3f p0(vertices[face.x]);
    Vec3f p1(vertices[face.y]);
    Vec3f p2(vertices[face.z]);
    float e0 = (p1 - p0).norm2();
    float e1 = (p2 - p1).norm2();
    float e2 = (p0 - p2).norm2();
    return fmaxf(e0, fmaxf(e1, e2));
}


__global__ void mark_normalize_small_faces_kernel(
    const float3* vertices,
    const int3* faces,
    const int F,
    const float min_area_abs,
    const float min_area_rel,
    uint8_t* small_faces
) {
    const int fid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (fid >= F) return;

    const int3 face = faces[fid];
    small_faces[fid] = 0;

    // Repeated-index faces do not have a valid edge-collapse candidate. They
    // are left for the final validation path instead of being removed here.
    if (face.x == face.y || face.y == face.z || face.z == face.x) return;

    const float area = normalize_face_area(vertices, face);
    const float longest_edge_squared = normalize_longest_edge_squared(vertices, face);
    if (!isfinite(area) || !isfinite(longest_edge_squared)) return;

    if (area < min_area_abs && area < min_area_rel * longest_edge_squared) {
        small_faces[fid] = 1;
    }
}


__global__ void mark_normalize_split_corners_kernel(
    const int3* faces,
    const int3* face2edge,
    const int* edge_face_counts,
    const int F,
    const int E,
    int* split_corners
) {
    const int fid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (fid >= F) return;

    const int3 edge_ids = face2edge[fid];
    const int corner_base = 3 * fid;

    split_corners[corner_base + 0] = 0;
    split_corners[corner_base + 1] = 0;
    split_corners[corner_base + 2] = 0;

    if (edge_ids.x >= 0 && edge_ids.x < E && edge_face_counts[edge_ids.x] > 2) {
        split_corners[corner_base + 0] = 1;
        split_corners[corner_base + 1] = 1;
    }
    if (edge_ids.y >= 0 && edge_ids.y < E && edge_face_counts[edge_ids.y] > 2) {
        split_corners[corner_base + 1] = 1;
        split_corners[corner_base + 2] = 1;
    }
    if (edge_ids.z >= 0 && edge_ids.z < E && edge_face_counts[edge_ids.z] > 2) {
        split_corners[corner_base + 2] = 1;
        split_corners[corner_base + 0] = 1;
    }

}


__global__ void write_normalize_split_vertices_kernel(
    const float3* old_vertices,
    const int3* faces,
    float3* vertices,
    const int old_vertex_count,
    const int corner_count,
    const int* split_corners,
    const int* corner_offsets
) {
    const int corner = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (corner >= corner_count || split_corners[corner] == 0) return;

    const int face_id = corner / 3;
    const int local_corner = corner % 3;
    const int3 face = faces[face_id];
    const int source_vertex = local_corner == 0
        ? face.x
        : (local_corner == 1 ? face.y : face.z);
    vertices[old_vertex_count + corner_offsets[corner]] = old_vertices[source_vertex];
}


__global__ void rewrite_normalize_faces_kernel(
    int3* faces,
    const int old_vertex_count,
    const int F,
    const int* split_corners,
    const int* corner_offsets
) {
    const int fid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (fid >= F) return;

    int3 face = faces[fid];
    const int base = 3 * fid;
    if (split_corners[base + 0]) face.x = old_vertex_count + corner_offsets[base + 0];
    if (split_corners[base + 1]) face.y = old_vertex_count + corner_offsets[base + 1];
    if (split_corners[base + 2]) face.z = old_vertex_count + corner_offsets[base + 2];
    faces[fid] = face;
}


__global__ void initialize_int_buffer_kernel(int* values, const int count, const int value) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid < count) values[tid] = value;
}


__device__ inline bool normalize_replaced_face_is_valid(
    const float3* vertices,
    const int3& face,
    const int replaced_vertex,
    const Vec3f& replacement,
    const float min_area_abs
) {
    Vec3f p0(vertices[face.x]);
    Vec3f p1(vertices[face.y]);
    Vec3f p2(vertices[face.z]);
    Vec3f old_normal = (p1 - p0).cross(p2 - p0);

    if (face.x == replaced_vertex) p0 = replacement;
    if (face.y == replaced_vertex) p1 = replacement;
    if (face.z == replaced_vertex) p2 = replacement;

    Vec3f new_normal = (p1 - p0).cross(p2 - p0);
    const float new_area = 0.5f * new_normal.norm();
    if (!isfinite(new_area) || new_area <= fmaxf(min_area_abs, 1e-24f)) return false;

    const float old_area = 0.5f * old_normal.norm();
    if (isfinite(old_area) && old_area > 1e-24f && old_normal.dot(new_normal) < 0.0f) {
        return false;
    }
    return true;
}


__device__ inline bool normalize_collapse_is_valid(
    const float3* vertices,
    const int3* faces,
    const int* vertex_face_ids,
    const int* vertex_face_offsets,
    const int keep_vertex,
    const int removed_vertex,
    const Vec3f& replacement,
    const float min_area_abs
) {
    for (int i = vertex_face_offsets[keep_vertex]; i < vertex_face_offsets[keep_vertex + 1]; ++i) {
        const int face_id = vertex_face_ids[i];
        const int3 face = faces[face_id];
        if (normalize_face_contains(face, removed_vertex)) continue;
        if (!normalize_replaced_face_is_valid(
                vertices, face, keep_vertex, replacement, min_area_abs)) {
            return false;
        }
    }

    for (int i = vertex_face_offsets[removed_vertex]; i < vertex_face_offsets[removed_vertex + 1]; ++i) {
        const int face_id = vertex_face_ids[i];
        const int3 face = faces[face_id];
        if (normalize_face_contains(face, keep_vertex)) continue;
        if (!normalize_replaced_face_is_valid(
                vertices, face, removed_vertex, replacement, min_area_abs)) {
            return false;
        }
    }
    return true;
}


__device__ inline unsigned long long normalize_edge_priority(
    float score,
    int edge_id
) {
    const unsigned int score_bits = __float_as_uint(score);
    return (static_cast<unsigned long long>(score_bits) << 32) |
           static_cast<unsigned int>(edge_id);
}


__global__ void select_normalize_edges_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vertex_face_ids,
    const int* vertex_face_offsets,
    const uint64_t* edges,
    const int* edge_face_counts,
    const int* edge_face_offsets,
    const int* edge_faces,
    const uint8_t* small_faces,
    const int E,
    const float min_area_abs,
    uint64_t* best_edge_per_vertex
) {
    const int edge_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (edge_id >= E || edge_face_counts[edge_id] != 2) return;

    bool touches_small_face = false;
    const int begin = edge_face_offsets[edge_id];
    const int end = edge_face_offsets[edge_id + 1];
    for (int i = begin; i < end; ++i) {
        if (small_faces[edge_faces[i]]) {
            touches_small_face = true;
            break;
        }
    }
    if (!touches_small_face) return;

    const uint64_t edge = edges[edge_id];
    const int keep_vertex = static_cast<int>(edge >> 32);
    const int removed_vertex = static_cast<int>(edge & 0xffffffffu);
    if (keep_vertex == removed_vertex) return;

    Vec3f p0(vertices[keep_vertex]);
    Vec3f p1(vertices[removed_vertex]);
    const Vec3f replacement = (p0 + p1) * 0.5f;
    if (!normalize_collapse_is_valid(
            vertices,
            faces,
            vertex_face_ids,
            vertex_face_offsets,
            keep_vertex,
            removed_vertex,
            replacement,
            min_area_abs)) {
        return;
    }

    const float score = (p1 - p0).norm2();
    if (!isfinite(score)) return;
    const unsigned long long priority = normalize_edge_priority(score, edge_id);
    atomicMin(
        reinterpret_cast<unsigned long long*>(&best_edge_per_vertex[keep_vertex]),
        priority
    );
    atomicMin(
        reinterpret_cast<unsigned long long*>(&best_edge_per_vertex[removed_vertex]),
        priority
    );
}


__global__ void mark_normalize_selected_edges_kernel(
    const float3* vertices,
    const uint64_t* edges,
    const int* edge_face_counts,
    const int E,
    const uint64_t* best_edge_per_vertex,
    uint8_t* selected_edges
) {
    const int edge_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (edge_id >= E) return;
    selected_edges[edge_id] = 0;
    if (edge_face_counts[edge_id] != 2) return;

    const uint64_t edge = edges[edge_id];
    const int keep_vertex = static_cast<int>(edge >> 32);
    const int removed_vertex = static_cast<int>(edge & 0xffffffffu);

    const float score = (Vec3f(vertices[removed_vertex]) - Vec3f(vertices[keep_vertex])).norm2();
    if (!isfinite(score)) return;
    const unsigned long long priority = normalize_edge_priority(score, edge_id);
    if (best_edge_per_vertex[keep_vertex] == priority &&
        best_edge_per_vertex[removed_vertex] == priority) {
        selected_edges[edge_id] = 1;
    }
}


__global__ void collapse_normalize_edges_kernel(
    float3* vertices,
    int3* faces,
    const uint64_t* edges,
    const int* vertex_face_ids,
    const int* vertex_face_offsets,
    const int E,
    const uint8_t* selected_edges,
    int* vertices_kept,
    int* faces_kept
) {
    const int edge_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (edge_id >= E || selected_edges[edge_id] == 0) return;

    const uint64_t edge = edges[edge_id];
    const int keep_vertex = static_cast<int>(edge >> 32);
    const int removed_vertex = static_cast<int>(edge & 0xffffffffu);

    Vec3f p0(vertices[keep_vertex]);
    Vec3f p1(vertices[removed_vertex]);
    const Vec3f replacement = (p0 + p1) * 0.5f;
    vertices[keep_vertex] = make_float3(replacement.x, replacement.y, replacement.z);
    vertices_kept[removed_vertex] = 0;

    for (int i = vertex_face_offsets[keep_vertex]; i < vertex_face_offsets[keep_vertex + 1]; ++i) {
        const int face_id = vertex_face_ids[i];
        int3 face = faces[face_id];
        if (normalize_face_contains(face, removed_vertex)) {
            faces_kept[face_id] = 0;
        }
    }

    for (int i = vertex_face_offsets[removed_vertex]; i < vertex_face_offsets[removed_vertex + 1]; ++i) {
        const int face_id = vertex_face_ids[i];
        int3 face = faces[face_id];
        if (face.x == removed_vertex) face.x = keep_vertex;
        if (face.y == removed_vertex) face.y = keep_vertex;
        if (face.z == removed_vertex) face.z = keep_vertex;
        faces[face_id] = face;
    }
}


__global__ void compress_normalize_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= V) return;
    const int new_id = vertices_map[tid];
    if (vertices_map[tid + 1] == new_id + 1) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


__global__ void compress_normalize_faces_kernel(
    const int* faces_map,
    const int* vertices_map,
    const int3* old_faces,
    const int F,
    int3* new_faces
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= F) return;
    const int new_id = faces_map[tid];
    if (faces_map[tid + 1] == new_id + 1) {
        const int3 face = old_faces[tid];
        new_faces[new_id] = make_int3(
            vertices_map[face.x],
            vertices_map[face.y],
            vertices_map[face.z]
        );
    }
}


static void split_normalize_non_manifold_edges(CuMesh& mesh) {
    const int F = static_cast<int>(mesh.faces.size);
    if (F == 0) return;

    mesh.get_edges();
    mesh.get_edge_face_adjacency();
    const int E = static_cast<int>(mesh.edges.size);
    if (E == 0) return;

    Buffer<int> split_corners;
    Buffer<int> corner_offsets;
    const int corner_count = 3 * F;
    split_corners.resize(corner_count);
    corner_offsets.resize(corner_count);

    mark_normalize_split_corners_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.faces.ptr,
        mesh.face2edge.ptr,
        mesh.edge2face_cnt.ptr,
        F,
        E,
        split_corners.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        split_corners.ptr,
        corner_offsets.ptr,
        corner_count
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr,
        temp_storage_bytes,
        split_corners.ptr,
        corner_offsets.ptr,
        corner_count
    ));

    int last_offset = 0;
    int last_flag = 0;
    CUDA_CHECK(cudaMemcpy(
        &last_offset,
        corner_offsets.ptr + corner_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &last_flag,
        split_corners.ptr + corner_count - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    const int split_count = last_offset + last_flag;

    if (split_count == 0) {
        split_corners.free();
        corner_offsets.free();
        return;
    }

    const int old_vertex_count = static_cast<int>(mesh.vertices.size);
    mesh.vertices.extend(split_count);
    write_normalize_split_vertices_kernel<<<
        (corner_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        mesh.vertices.ptr,
        mesh.faces.ptr,
        mesh.vertices.ptr,
        old_vertex_count,
        corner_count,
        split_corners.ptr,
        corner_offsets.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    rewrite_normalize_faces_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.faces.ptr,
        old_vertex_count,
        F,
        split_corners.ptr,
        corner_offsets.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    split_corners.free();
    corner_offsets.free();
    mesh.clear_cache();
}


static void compact_after_normalize_collapse(
    CuMesh& mesh,
    int old_vertex_count,
    int old_face_count
) {
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        mesh.vertices_map.ptr,
        old_vertex_count + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr,
        temp_storage_bytes,
        mesh.vertices_map.ptr,
        old_vertex_count + 1
    ));

    int new_vertex_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &new_vertex_count,
        mesh.vertices_map.ptr + old_vertex_count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (new_vertex_count > 0) {
        mesh.temp_storage.resize(new_vertex_count * sizeof(float3));
        compress_normalize_vertices_kernel<<<
            (old_vertex_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            mesh.vertices_map.ptr,
            mesh.vertices.ptr,
            old_vertex_count,
            reinterpret_cast<float3*>(mesh.temp_storage.ptr)
        );
        CUDA_CHECK(cudaGetLastError());
        swap_buffers(mesh.temp_storage, mesh.vertices);
    } else {
        mesh.vertices.resize(0);
    }

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        mesh.faces_map.ptr,
        old_face_count + 1
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        mesh.cub_temp_storage.ptr,
        temp_storage_bytes,
        mesh.faces_map.ptr,
        old_face_count + 1
    ));

    int new_face_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &new_face_count,
        mesh.faces_map.ptr + old_face_count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (new_face_count > 0) {
        mesh.temp_storage.resize(new_face_count * sizeof(int3));
        compress_normalize_faces_kernel<<<
            (old_face_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            mesh.faces_map.ptr,
            mesh.vertices_map.ptr,
            mesh.faces.ptr,
            old_face_count,
            reinterpret_cast<int3*>(mesh.temp_storage.ptr)
        );
        CUDA_CHECK(cudaGetLastError());
        swap_buffers(mesh.temp_storage, mesh.faces);
    } else {
        mesh.faces.resize(0);
    }
}


static bool collapse_one_normalize_batch(
    CuMesh& mesh,
    float min_area_abs,
    float min_area_rel
) {
    const int F = static_cast<int>(mesh.faces.size);
    if (F == 0) return false;

    if (mesh.vert2face.is_empty() || mesh.vert2face_offset.is_empty()) {
        mesh.get_vertex_face_adjacency();
    }
    if (mesh.edges.is_empty() || mesh.edge2face_cnt.is_empty()) {
        mesh.get_edges();
    }
    if (mesh.edge2face.is_empty() || mesh.edge2face_offset.is_empty()) {
        mesh.get_edge_face_adjacency();
    }

    const int V = static_cast<int>(mesh.vertices.size);
    const int E = static_cast<int>(mesh.edges.size);
    if (V == 0 || E == 0) return false;

    Buffer<uint8_t> small_faces;
    Buffer<uint64_t> best_edge_per_vertex;
    Buffer<uint8_t> selected_edges;
    small_faces.resize(F);
    best_edge_per_vertex.resize(V);
    selected_edges.resize(E);

    mark_normalize_small_faces_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.vertices.ptr,
        mesh.faces.ptr,
        F,
        min_area_abs,
        min_area_rel,
        small_faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(
        best_edge_per_vertex.ptr,
        0xff,
        V * sizeof(uint64_t)
    ));

    select_normalize_edges_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        mesh.vertices.ptr,
        mesh.faces.ptr,
        mesh.vert2face.ptr,
        mesh.vert2face_offset.ptr,
        mesh.edges.ptr,
        mesh.edge2face_cnt.ptr,
        mesh.edge2face_offset.ptr,
        mesh.edge2face.ptr,
        small_faces.ptr,
        E,
        min_area_abs,
        best_edge_per_vertex.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    mark_normalize_selected_edges_kernel<<<
        (E + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        mesh.vertices.ptr,
        mesh.edges.ptr,
        mesh.edge2face_cnt.ptr,
        E,
        best_edge_per_vertex.ptr,
        selected_edges.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    int* selected_count_device = nullptr;
    CUDA_CHECK(cudaMalloc(&selected_count_device, sizeof(int)));
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        selected_edges.ptr,
        selected_count_device,
        E
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        mesh.cub_temp_storage.ptr,
        temp_storage_bytes,
        selected_edges.ptr,
        selected_count_device,
        E
    ));

    int selected_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &selected_count,
        selected_count_device,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaFree(selected_count_device));

    if (selected_count == 0) {
        small_faces.free();
        best_edge_per_vertex.free();
        selected_edges.free();
        return false;
    }

    mesh.vertices_map.resize(V + 1);
    mesh.faces_map.resize(F + 1);
    initialize_int_buffer_kernel<<<
        (V + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(mesh.vertices_map.ptr, V + 1, 1);
    initialize_int_buffer_kernel<<<
        (F + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(mesh.faces_map.ptr, F + 1, 1);
    CUDA_CHECK(cudaGetLastError());

    collapse_normalize_edges_kernel<<<
        (E + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        mesh.vertices.ptr,
        mesh.faces.ptr,
        mesh.edges.ptr,
        mesh.vert2face.ptr,
        mesh.vert2face_offset.ptr,
        E,
        selected_edges.ptr,
        mesh.vertices_map.ptr,
        mesh.faces_map.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    compact_after_normalize_collapse(mesh, V, F);

    small_faces.free();
    best_edge_per_vertex.free();
    selected_edges.free();
    mesh.clear_cache();
    return true;
}


void CuMesh::normalize(
    float min_area_abs,
    float min_area_rel,
    int iterations,
    bool verbose
) {
    TORCH_CHECK(min_area_abs >= 0.0f, "min_area_abs must be non-negative");
    TORCH_CHECK(min_area_rel >= 0.0f, "min_area_rel must be non-negative");
    TORCH_CHECK(iterations >= 0, "iterations must be non-negative");

    (void)verbose;
    this->clear_cache();
    if (this->faces.size == 0 || this->vertices.size == 0) return;

    // Stage 1 is intentionally a single global split-and-rebuild operation.
    split_normalize_non_manifold_edges(*this);

    // Stage 2 is optional and defaults to one GPU batch. Each additional
    // iteration rebuilds only the data affected by the preceding collapse.
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (!collapse_one_normalize_batch(*this, min_area_abs, min_area_rel)) {
            break;
        }
    }
}


} // namespace cumesh
