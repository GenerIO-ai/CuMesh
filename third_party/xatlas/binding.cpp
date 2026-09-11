#include <torch/extension.h>
#include "xatlas.h"
#include "safe_uv.h"

#include <cstring>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


namespace cumesh_xatlas {


void check_tensor(const torch::Tensor& tensor, const std::string& name, torch::ScalarType type) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has incorrect data type");
}


bool ProgressCallbackTrampoline(cumesh_xatlas::ProgressCategory category, int progress, void* userData) {
    // userData stores pointer to py::function
    auto* func = static_cast<py::function*>(userData);
    
    py::gil_scoped_acquire gil;
    
    try {
        (*func)(cumesh_xatlas::StringForEnum(category), progress);
        return true; // true: continue
    } catch (py::error_already_set& e) {
        return false; // false: stop
    }
}


bool LscmProgressCallbackTrampoline(uint32_t completed, uint32_t total, void* userData) {
    auto* func = static_cast<py::function*>(userData);

    py::gil_scoped_acquire gil;

    try {
        (*func)(completed, total);
        return true;
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("batched LSCM progress callback");
        return false;
    }
}


void LscmTraceCallbackTrampoline(
    uint32_t chartIndex,
    const char* phase,
    uint32_t vertexCount,
    uint32_t faceCount,
    void* userData
) {
    auto* func = static_cast<py::function*>(userData);

    py::gil_scoped_acquire gil;

    try {
        (*func)(chartIndex, phase, vertexCount, faceCount);
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("batched LSCM trace callback");
    }
}


class XAtlasWrapper {

private:
    cumesh_xatlas::Atlas* m_atlas;
    py::dict m_lscmStats;

public:
    py::dict LscmStats() const { return m_lscmStats; }

    py::dict AtlasInfo() const {
        py::dict result;
        result["width"] = m_atlas->width;
        result["height"] = m_atlas->height;
        result["atlas_count"] = m_atlas->atlasCount;
        result["chart_count"] = m_atlas->chartCount;
        return result;
    }

    torch::Tensor MeshAtlasIndices(uint32_t index) const {
        TORCH_CHECK(index < m_atlas->meshCount, "mesh index out of bounds");
        const auto &mesh = m_atlas->meshes[index];
        auto result = torch::empty({int64_t(mesh.vertexCount)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        auto *ptr = result.data_ptr<int32_t>();
        for (uint32_t v=0; v<mesh.vertexCount; ++v) ptr[v] = mesh.vertexArray[v].atlasIndex;
        return result;
    }

    static py::dict ValidateUv(const torch::Tensor &uv, const torch::Tensor &faces) {
        check_tensor(uv, "uv", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);
        TORCH_CHECK(uv.dim()==2 && uv.size(1)==2, "uv must have shape [V,2]");
        TORCH_CHECK(faces.dim()==2 && faces.size(1)==3, "faces must have shape [F,3]");
        TORCH_CHECK(uv.size(0)<=std::numeric_limits<uint32_t>::max() && faces.size(0)<=std::numeric_limits<uint32_t>::max(), "UV mesh too large");
        safeuv::Validation validation;
        {
            py::gil_scoped_release release;
            validation = safeuv::Validator(uv.data_ptr<float>(), faces.data_ptr<int32_t>()).run(
                uint32_t(uv.size(0)), uint32_t(faces.size(0)));
        }
        const char *names[] = {"valid", "nonfinite", "degenerate", "overlap", "invalid_index"};
        py::dict result;
        result["valid"] = validation.issue == 0;
        result["issue"] = names[validation.issue];
        result["face0"] = validation.face0; result["face1"] = validation.face1;
        result["candidate_pairs"] = validation.candidates;
        result["seconds"] = validation.elapsed;
        return result;
    }

    XAtlasWrapper() {
        m_atlas = cumesh_xatlas::Create();
    }

    ~XAtlasWrapper() {
        cumesh_xatlas::Destroy(m_atlas);
    }

    void AddMesh(
        const torch::Tensor& vertices,
        const torch::Tensor& faces,
        std::optional<const torch::Tensor> normals,
        std::optional<const torch::Tensor> uvs
    ) {
        check_tensor(vertices, "vertices", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);
        
        // 1. Construct mesh declaration
        cumesh_xatlas::MeshDecl meshDecl;
        meshDecl.vertexCount = static_cast<uint32_t>(vertices.size(0));
        meshDecl.vertexPositionData = vertices.data_ptr<float>();
        meshDecl.vertexPositionStride = sizeof(float) * 3;
        meshDecl.indexCount = static_cast<uint32_t>(faces.size(0) * 3);
        meshDecl.indexData = faces.data_ptr<int32_t>();
        meshDecl.indexFormat = cumesh_xatlas::IndexFormat::UInt32;
        if (normals.has_value()) {
            check_tensor(*normals, "normals", torch::kFloat32);
            meshDecl.vertexNormalData = normals->data_ptr<float>();
            meshDecl.vertexNormalStride = sizeof(float) * 3;
        }
        if (uvs.has_value()) {
            check_tensor(*uvs, "uvs", torch::kFloat32);
            meshDecl.vertexUvData = uvs->data_ptr<float>();
            meshDecl.vertexUvStride = sizeof(float) * 2;
        }
        
        // 2. Add mesh to atlas
        cumesh_xatlas::AddMeshError result = cumesh_xatlas::AddMesh(m_atlas, meshDecl);
        if (result != cumesh_xatlas::AddMeshError::Success) {
            throw std::runtime_error("Adding mesh failed: " + std::string(cumesh_xatlas::StringForEnum(result)));
        }
    }

    void ComputeCharts(cumesh_xatlas::ChartOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            cumesh_xatlas::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }
        
        {
            py::gil_scoped_release gil;
            cumesh_xatlas::ComputeCharts(m_atlas, options);
        }
        
        cumesh_xatlas::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    void PackCharts(cumesh_xatlas::PackOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            cumesh_xatlas::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }

        {
            py::gil_scoped_release gil;
            cumesh_xatlas::PackCharts(m_atlas, options);
        }

        cumesh_xatlas::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    py::tuple ParameterizeLscmBatch(
        const torch::Tensor& vertices,
        const torch::Tensor& faces,
        const torch::Tensor& vertexOffsets,
        const torch::Tensor& faceOffsets,
        std::optional<py::function> progressCallback,
        std::optional<py::function> traceCallback
    ) {
        check_tensor(vertices, "vertices", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);
        check_tensor(vertexOffsets, "vertexOffsets", torch::kInt32);
        check_tensor(faceOffsets, "faceOffsets", torch::kInt32);

        TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3,
            "vertices must have shape [vertex_count, 3]");
        TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3,
            "faces must have shape [face_count, 3]");
        TORCH_CHECK(vertexOffsets.dim() == 1 && faceOffsets.dim() == 1,
            "offset tensors must be one-dimensional");
        TORCH_CHECK(vertexOffsets.numel() == faceOffsets.numel(),
            "vertexOffsets and faceOffsets must have the same length");
        TORCH_CHECK(vertexOffsets.numel() > 0,
            "offset tensors must contain at least one entry");
        TORCH_CHECK(vertexOffsets.numel() - 1 <= std::numeric_limits<uint32_t>::max(),
            "too many charts for batched LSCM");
        TORCH_CHECK(vertices.size(0) <= std::numeric_limits<uint32_t>::max(),
            "too many vertices for batched LSCM");
        TORCH_CHECK(faces.size(0) <= std::numeric_limits<uint32_t>::max(),
            "too many faces for batched LSCM");

        const int32_t* vertexOffsetPtr = vertexOffsets.data_ptr<int32_t>();
        const int32_t* faceOffsetPtr = faceOffsets.data_ptr<int32_t>();
        for (int64_t i = 0; i < vertexOffsets.numel(); ++i) {
            TORCH_CHECK(vertexOffsetPtr[i] >= 0 && faceOffsetPtr[i] >= 0,
                "offset tensors must contain non-negative values");
            if (i > 0) {
                TORCH_CHECK(vertexOffsetPtr[i] >= vertexOffsetPtr[i - 1],
                    "vertexOffsets must be non-decreasing");
                TORCH_CHECK(faceOffsetPtr[i] >= faceOffsetPtr[i - 1],
                    "faceOffsets must be non-decreasing");
            }
        }
        TORCH_CHECK(vertexOffsetPtr[vertexOffsets.numel() - 1] == vertices.size(0),
            "last vertex offset must equal the number of packed vertices");
        TORCH_CHECK(faceOffsetPtr[faceOffsets.numel() - 1] == faces.size(0),
            "last face offset must equal the number of packed faces");
        TORCH_CHECK(vertexOffsetPtr[0] == 0 && faceOffsetPtr[0] == 0,
            "first offsets must be zero");

        const uint32_t chartCount = static_cast<uint32_t>(vertexOffsets.numel() - 1);
        const int32_t* facePtr = faces.data_ptr<int32_t>();
        for (uint32_t chart = 0; chart < chartCount; ++chart) {
            const int32_t chartVertexCount = vertexOffsetPtr[chart + 1] - vertexOffsetPtr[chart];
            const int64_t firstFace = faceOffsetPtr[chart];
            const int64_t lastFace = faceOffsetPtr[chart + 1];
            for (int64_t face = firstFace; face < lastFace; ++face) {
                for (int corner = 0; corner < 3; ++corner) {
                    const int32_t index = facePtr[face * 3 + corner];
                    TORCH_CHECK(index >= 0 && index < chartVertexCount,
                        "faces must use zero-based indices local to each chart");
                }
            }
        }

        std::vector<cumesh_xatlas::LscmChartResult> results;
        {
            py::gil_scoped_release gil;
            cumesh_xatlas::ParameterizeLscmBatch(
                m_atlas,
                vertices.data_ptr<float>(),
                static_cast<uint32_t>(vertices.size(0)),
                faces.data_ptr<int32_t>(),
                static_cast<uint32_t>(faces.size(0)),
                vertexOffsetPtr,
                faceOffsetPtr,
                chartCount,
                results,
                progressCallback.has_value() ? &LscmProgressCallbackTrampoline : nullptr,
                progressCallback.has_value() ? static_cast<void*>(&(*progressCallback)) : nullptr,
                traceCallback.has_value() ? &LscmTraceCallbackTrampoline : nullptr,
                traceCallback.has_value() ? static_cast<void*>(&(*traceCallback)) : nullptr
            );
        }

        TORCH_CHECK(results.size() == chartCount,
            "native batched LSCM returned an unexpected number of chart results");

        int fallbacks=0, failures=0, pieces=0, cuts=0;
        uint64_t candidates=0;
        double validationSeconds=0, fallbackSeconds=0, solveSeconds=0;
        std::array<int,8> reasons{};
        py::list slowCharts;
        std::vector<uint32_t> slowOrder(chartCount);
        std::iota(slowOrder.begin(),slowOrder.end(),0);
        std::stable_sort(slowOrder.begin(),slowOrder.end(),[&](uint32_t a,uint32_t b) {
            return results[a].solveSeconds+results[a].fallbackSeconds > results[b].solveSeconds+results[b].fallbackSeconds;
        });
        for(uint32_t i=0;i<chartCount;++i) {
            const auto &r=results[i];
            fallbacks+=r.fallbackCount; failures+=!r.success; pieces+=r.repairPieces; cuts+=r.topologyCuts;
            candidates+=r.validationCandidates;
            validationSeconds+=r.validationSeconds; fallbackSeconds+=r.fallbackSeconds; solveSeconds+=r.solveSeconds;
            if(r.invalidIssue>=0 && r.invalidIssue<8)++reasons[r.invalidIssue];
        }
        for(size_t j=0;j<std::min<size_t>(5,slowOrder.size());++j) {
            uint32_t i=slowOrder[j]; const auto &r=results[i]; py::dict item;
            item["chart"]=i; item["faces"]=faceOffsetPtr[i+1]-faceOffsetPtr[i];
            item["solve_seconds"]=r.solveSeconds; item["fallback_seconds"]=r.fallbackSeconds;
            item["validation_seconds"]=r.validationSeconds; item["pieces"]=r.repairPieces;
            item["reason"]=r.invalidIssue; slowCharts.append(item);
        }
        m_lscmStats=py::dict();
        m_lscmStats["fallback_charts"]=fallbacks; m_lscmStats["failed_charts"]=failures;
        m_lscmStats["repair_pieces"]=pieces; m_lscmStats["topology_cut_edges"]=cuts;
        m_lscmStats["lscm_overlap_charts"]=reasons[3];
        m_lscmStats["lscm_degenerate_charts"]=reasons[2];
        m_lscmStats["lscm_nonfinite_charts"]=reasons[1];
        m_lscmStats["solver_failure_charts"]=reasons[5];
        m_lscmStats["topology_recovery_charts"]=reasons[6];
        m_lscmStats["validation_candidate_pairs"]=candidates;
        // Worker seconds are summed CPU-task durations, not wall-clock time.
        m_lscmStats["validation_worker_seconds"]=validationSeconds;
        m_lscmStats["fallback_worker_seconds"]=fallbackSeconds;
        m_lscmStats["solve_worker_seconds"]=solveSeconds;
        m_lscmStats["slowest_charts"]=slowCharts;

        int64_t totalVertices = 0;
        int64_t totalFaces = 0;
        for (const auto& result : results) {
            TORCH_CHECK(result.uvs.size() == result.vmap.size() * 2,
                "native batched LSCM returned malformed UV data");
            TORCH_CHECK(result.indices.size() % 3 == 0,
                "native batched LSCM returned malformed index data");
            TORCH_CHECK(result.vmap.size() <= std::numeric_limits<int32_t>::max(),
                "a parameterized chart has too many vertices");
            TORCH_CHECK(result.indices.size() / 3 <= std::numeric_limits<int32_t>::max(),
                "a parameterized chart has too many faces");
            totalVertices += static_cast<int64_t>(result.vmap.size());
            totalFaces += static_cast<int64_t>(result.indices.size() / 3);
        }
        TORCH_CHECK(totalVertices <= std::numeric_limits<int32_t>::max(),
            "too many output vertices for batched LSCM");
        TORCH_CHECK(totalFaces <= std::numeric_limits<int32_t>::max(),
            "too many output faces for batched LSCM");

        auto floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto intOptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        auto boolOptions = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
        auto outputUvs = torch::empty({totalVertices, 2}, floatOptions);
        auto outputFaces = torch::empty({totalFaces, 3}, intOptions);
        auto outputVmaps = torch::empty({totalVertices}, intOptions);
        auto outputVertexOffsets = torch::empty({static_cast<int64_t>(chartCount) + 1}, intOptions);
        auto outputFaceOffsets = torch::empty({static_cast<int64_t>(chartCount) + 1}, intOptions);
        auto success = torch::empty({chartCount}, boolOptions);
        auto splitCounts = torch::empty({chartCount}, intOptions);

        int32_t* outputVertexOffsetPtr = outputVertexOffsets.data_ptr<int32_t>();
        int32_t* outputFaceOffsetPtr = outputFaceOffsets.data_ptr<int32_t>();
        bool* successPtr = success.data_ptr<bool>();
        int32_t* splitCountPtr = splitCounts.data_ptr<int32_t>();
        int64_t vertexOffset = 0;
        int64_t faceOffset = 0;
        outputVertexOffsetPtr[0] = 0;
        outputFaceOffsetPtr[0] = 0;
        for (uint32_t i = 0; i < chartCount; ++i) {
            const auto& result = results[i];
            if (!result.uvs.empty()) {
                std::memcpy(outputUvs.data_ptr<float>() + vertexOffset * 2,
                    result.uvs.data(), sizeof(float) * result.uvs.size());
            }
            if (!result.indices.empty()) {
                std::memcpy(outputFaces.data_ptr<int32_t>() + faceOffset * 3,
                    result.indices.data(), sizeof(int32_t) * result.indices.size());
            }
            if (!result.vmap.empty()) {
                std::memcpy(outputVmaps.data_ptr<int32_t>() + vertexOffset,
                    result.vmap.data(), sizeof(int32_t) * result.vmap.size());
            }
            vertexOffset += static_cast<int64_t>(result.vmap.size());
            faceOffset += static_cast<int64_t>(result.indices.size() / 3);
            outputVertexOffsetPtr[i + 1] = static_cast<int32_t>(vertexOffset);
            outputFaceOffsetPtr[i + 1] = static_cast<int32_t>(faceOffset);
            successPtr[i] = result.success;
            splitCountPtr[i] = result.splitCount;
        }

        return py::make_tuple(
            outputUvs,
            outputFaces,
            outputVmaps,
            outputVertexOffsets,
            outputFaceOffsets,
            success,
            splitCounts
        );
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetMesh(uint32_t index) {
        if (index >= m_atlas->meshCount) {
            throw std::out_of_range("Mesh index " + std::to_string(index) + " out of bounds for atlas with " + std::to_string(m_atlas->meshCount) + " meshes.");
        }

        auto const& mesh = m_atlas->meshes[index];

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        auto mapping = torch::empty({(long)mesh.vertexCount}, options_int);
        auto faces = torch::empty({(long)mesh.indexCount / 3, 3}, options_int);
        auto uv = torch::empty({(long)mesh.vertexCount, 2}, options);

        int32_t* mappingPtr = mapping.data_ptr<int32_t>();
        int32_t* facesPtr = faces.data_ptr<int32_t>();
        float* uvPtr = uv.data_ptr<float>();

        float width = (float)m_atlas->width;
        float height = (float)m_atlas->height;

        for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
            const auto& v = mesh.vertexArray[i];
            mappingPtr[i] = (int32_t)v.xref;
            
            if (width > 0 && height > 0) {
                uvPtr[i * 2 + 0] = v.uv[0] / width;
                uvPtr[i * 2 + 1] = v.uv[1] / height;
            } else {
                uvPtr[i * 2 + 0] = 0.0f;
                uvPtr[i * 2 + 1] = 0.0f;
            }
        }

        for (uint32_t i = 0; i < mesh.indexCount; ++i) {
            facesPtr[i] = (int32_t)mesh.indexArray[i];
        }

        return std::make_tuple(mapping, faces, uv);
    }

    void AddUvMesh(
        const torch::Tensor& uvs,
        const torch::Tensor& faces,
        std::optional<const torch::Tensor> faceMaterials
    ) {
        check_tensor(uvs, "uvs", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);

        cumesh_xatlas::UvMeshDecl decl;
        decl.vertexCount = static_cast<uint32_t>(uvs.size(0));
        decl.vertexUvData = uvs.data_ptr<float>();
        decl.vertexStride = sizeof(float) * 2;
        decl.indexCount = static_cast<uint32_t>(faces.size(0) * 3);
        decl.indexData = faces.data_ptr<int32_t>();
        decl.indexFormat = cumesh_xatlas::IndexFormat::UInt32;
        if (faceMaterials.has_value()) {
            check_tensor(*faceMaterials, "faceMaterials", torch::kInt32);
            decl.faceMaterialData = reinterpret_cast<const uint32_t*>(faceMaterials->data_ptr<int32_t>());
        }

        cumesh_xatlas::AddMeshError result = cumesh_xatlas::AddUvMesh(m_atlas, decl);
        if (result != cumesh_xatlas::AddMeshError::Success) {
            throw std::runtime_error("Adding UV mesh failed: " + std::string(cumesh_xatlas::StringForEnum(result)));
        }
    }
};

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool, int> ParameterizeLscmBinding(
    const torch::Tensor& vertices,
    const torch::Tensor& faces
) {
    check_tensor(vertices, "vertices", torch::kFloat32);
    check_tensor(faces, "faces", torch::kInt32);

    uint32_t vertexCount = static_cast<uint32_t>(vertices.size(0));
    uint32_t faceCount = static_cast<uint32_t>(faces.size(0));

    std::vector<float> outUvs;
    std::vector<int32_t> outIndices;
    std::vector<int32_t> outVmap;
    int splitCount = 0;

    bool success = cumesh_xatlas::ParameterizeLscm(
        vertices.data_ptr<float>(),
        vertexCount,
        faces.data_ptr<int32_t>(),
        faceCount,
        outUvs,
        outIndices,
        outVmap,
        splitCount
    );

    uint32_t newVertexCount = static_cast<uint32_t>(outVmap.size());
    auto uvsTensor = torch::empty({(long)newVertexCount, 2}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto facesTensor = torch::empty({(long)outIndices.size() / 3, 3}, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto vmapTensor = torch::empty({(long)newVertexCount}, torch::dtype(torch::kInt32).device(torch::kCPU));

    memcpy(uvsTensor.data_ptr<float>(), outUvs.data(), sizeof(float) * outUvs.size());
    memcpy(facesTensor.data_ptr<int32_t>(), outIndices.data(), sizeof(int32_t) * outIndices.size());
    memcpy(vmapTensor.data_ptr<int32_t>(), outVmap.data(), sizeof(int32_t) * outVmap.size());

    return std::make_tuple(uvsTensor, facesTensor, vmapTensor, success, splitCount);
}
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "xatlas wrapper for PyTorch";

    py::class_<cumesh_xatlas::ChartOptions>(m, "ChartOptions")
        .def(py::init<>())
        .def_readwrite("max_chart_area", &cumesh_xatlas::ChartOptions::maxChartArea)
        .def_readwrite("max_boundary_length", &cumesh_xatlas::ChartOptions::maxBoundaryLength)
        .def_readwrite("normal_deviation_weight", &cumesh_xatlas::ChartOptions::normalDeviationWeight)
        .def_readwrite("roundness_weight", &cumesh_xatlas::ChartOptions::roundnessWeight)
        .def_readwrite("straightness_weight", &cumesh_xatlas::ChartOptions::straightnessWeight)
        .def_readwrite("normal_seam_weight", &cumesh_xatlas::ChartOptions::normalSeamWeight)
        .def_readwrite("texture_seam_weight", &cumesh_xatlas::ChartOptions::textureSeamWeight)
        .def_readwrite("max_cost", &cumesh_xatlas::ChartOptions::maxCost)
        .def_readwrite("max_iterations", &cumesh_xatlas::ChartOptions::maxIterations)
        .def_readwrite("use_input_mesh_uvs", &cumesh_xatlas::ChartOptions::useInputMeshUvs)
        .def_readwrite("fix_winding", &cumesh_xatlas::ChartOptions::fixWinding);

    py::class_<cumesh_xatlas::PackOptions>(m, "PackOptions")
        .def(py::init<>())
        .def_readwrite("max_chart_size", &cumesh_xatlas::PackOptions::maxChartSize)
        .def_readwrite("padding", &cumesh_xatlas::PackOptions::padding)
        .def_readwrite("texels_per_unit", &cumesh_xatlas::PackOptions::texelsPerUnit)
        .def_readwrite("resolution", &cumesh_xatlas::PackOptions::resolution)
        .def_readwrite("bilinear", &cumesh_xatlas::PackOptions::bilinear)
        .def_readwrite("block_align", &cumesh_xatlas::PackOptions::blockAlign)
        .def_readwrite("brute_force", &cumesh_xatlas::PackOptions::bruteForce)
        .def_readwrite("rotate_charts", &cumesh_xatlas::PackOptions::rotateCharts)
        .def_readwrite("rotate_charts_to_axis", &cumesh_xatlas::PackOptions::rotateChartsToAxis);

    py::class_<cumesh_xatlas::XAtlasWrapper>(m, "Atlas")
        .def(py::init<>())
        .def("add_mesh", &cumesh_xatlas::XAtlasWrapper::AddMesh)
        .def("add_uv_mesh", &cumesh_xatlas::XAtlasWrapper::AddUvMesh, py::arg("uvs"), py::arg("faces"), py::arg("face_materials") = py::none())
        .def("compute_charts", &cumesh_xatlas::XAtlasWrapper::ComputeCharts)
        .def("pack_charts", &cumesh_xatlas::XAtlasWrapper::PackCharts)
        .def("_parameterize_lscm_batch", &cumesh_xatlas::XAtlasWrapper::ParameterizeLscmBatch,
            py::arg("vertices"), py::arg("faces"), py::arg("vertex_offsets"),
            py::arg("face_offsets"), py::arg("progress_callback") = py::none(),
            py::arg("trace_callback") = py::none())
        .def("get_mesh", &cumesh_xatlas::XAtlasWrapper::GetMesh)
        .def("_lscm_stats", &cumesh_xatlas::XAtlasWrapper::LscmStats)
        .def("_atlas_info", &cumesh_xatlas::XAtlasWrapper::AtlasInfo)
        .def("_mesh_atlas_indices", &cumesh_xatlas::XAtlasWrapper::MeshAtlasIndices)
        .def_static("_validate_uv", &cumesh_xatlas::XAtlasWrapper::ValidateUv);

    m.def("parameterize_lscm", &cumesh_xatlas::ParameterizeLscmBinding);
}
