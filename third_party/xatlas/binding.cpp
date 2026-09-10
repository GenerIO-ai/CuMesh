#include <torch/extension.h>
#include "xatlas.h"

#include <cstring>
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


class XAtlasWrapper {

private:
    cumesh_xatlas::Atlas* m_atlas;

public:
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
        .def("get_mesh", &cumesh_xatlas::XAtlasWrapper::GetMesh);

    m.def("parameterize_lscm", &cumesh_xatlas::ParameterizeLscmBinding);
}
