from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

# todo
ext_modules = []

print('CUDA_HOME', CUDA_HOME)

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_xsparse.batch_diag_cuda',
                      ['cuda/batch_diag.cpp', 'cuda/batch_diag_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_xsparse.unique_cuda',
                      ['cuda/tunique.cpp', 'cuda/tunique_kernel.cu'],
                      extra_compile_args=extra_compile_args),
    ]

setup(
    name='torch_xsparse',
    version='0.0.3',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages()
)
