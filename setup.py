from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

# todo
ext_modules = []

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_xsparse.batch_diag_cuda',
                      ['cuda/batch_diag_cuda.cpp', 'cuda/batch_diag_kernel.cu']),
    ]

setup(
    name='torch_xsparse',
    version='0.0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages()
)