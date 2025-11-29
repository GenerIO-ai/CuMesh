from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup(
    name="cumesh",
    packages=[
        'cumesh',
    ],
    ext_modules=[
        CUDAExtension(
            name="cumesh._C",
            sources=[
                "src/atlas.cu",
                "src/clean_up.cu",
                "src/cumesh.cu",
                "src/connectivity.cu",
                "src/geometry.cu",
                "src/io.cu",
                "src/simplify.cu",
                "src/shared.cu",
                
                # main
                "src/ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)