"""Build configuration for the CuMesh binary extensions.

The three extensions are deliberately built as part of one distribution so downstream
packages can install a wheel without compiling CUDA/C++ code themselves.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    IS_HIP_EXTENSION,
    include_paths,
)


ROOT = Path(__file__).resolve().parent
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
IS_WINDOWS = platform.system() == "Windows"

if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
elif BUILD_TARGET == "cuda":
    IS_HIP = False
elif BUILD_TARGET == "rocm":
    IS_HIP = True
else:
    raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET!r}")


if IS_WINDOWS:
    cxx_flags = [
        "/O2",
        "/std:c++17",
        "/EHsc",
        "/permissive-",
        "/Zc:__cplusplus",
    ]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-Xcompiler=/EHsc",
        "-Xcompiler=/permissive-",
        "-Xcompiler=/Zc:__cplusplus",
        "-allow-unsupported-compiler",
    ]
else:
    cxx_flags = ["-O3", "-std=c++17"]
    nvcc_flags = ["-O3", "-std=c++17"]

if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags += [f"--offload-arch={arch}" for arch in archs]


def _eigen_include_dirs() -> list[str]:
    """Find vendored Eigen or the copy shipped with PyTorch.

    A source checkout may provide Eigen under ``third_party/cubvh/third_party/eigen``;
    otherwise a builder can point this helper at a system Eigen installation or another
    include tree. Binary wheel consumers do not need Eigen.
    """

    candidates = []
    configured = os.environ.get("EIGEN3_INCLUDE_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "third_party" / "cubvh" / "third_party" / "eigen",
            *(Path(path) / "third_party" / "eigen3" for path in include_paths()),
            *(Path(path) / "third_party" / "eigen" for path in include_paths()),
        ]
    )
    eigen_dirs = [str(path) for path in candidates if (path / "Eigen").is_dir()]
    if not eigen_dirs:
        raise RuntimeError(
            "Eigen 3 headers were not found. Install Eigen 3 or set "
            "EIGEN3_INCLUDE_DIR to the directory containing Eigen/."
        )
    return eigen_dirs


main_sources = [
    "src/hash/hash.cu",
    "src/atlas.cu",
    "src/clean_up.cu",
    "src/cumesh.cu",
    "src/connectivity.cu",
    "src/geometry.cu",
    "src/io.cu",
    "src/simplify.cu",
    "src/shared.cu",
    "src/remesh/simple_dual_contour.cu",
    "src/remesh/svox2vert.cu",
    "src/ext.cpp",
]

cubvh_sources = [
    "third_party/cubvh/src/bvh.cu",
    "third_party/cubvh/src/api_gpu.cu",
    "third_party/cubvh/src/bindings.cpp",
]

ext_modules = [
    CUDAExtension(
        name="cumesh._C",
        sources=main_sources,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ),
    CUDAExtension(
        name="cumesh._cubvh",
        sources=cubvh_sources,
        include_dirs=[
            str(ROOT / "third_party" / "cubvh" / "include"),
            *_eigen_include_dirs(),
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags
            + [
                "--extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    ),
    # xatlas is CPU code, but its binding uses the PyTorch C++ API. Keeping it as a
    # separate C++ extension avoids invoking NVCC for this component.
    CppExtension(
        name="cumesh._cumesh_xatlas",
        sources=[
            "third_party/xatlas/xatlas.cpp",
            "third_party/xatlas/binding.cpp",
        ],
        extra_compile_args={"cxx": cxx_flags},
    ),
]


setup(
    name="cumesh",
    version="0.9.0",
    packages=find_packages(include=["cumesh", "cumesh.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    license_files=[
        "LICENSE",
        "third_party/xatlas/LICENSE",
        "third_party/cubvh/LICENSE",
        "third_party/cubvh/LICENSE_NVIDIA",
    ],
    zip_safe=False,
)
