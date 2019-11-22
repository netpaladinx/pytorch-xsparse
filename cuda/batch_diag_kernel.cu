#include <ATen/ATen.h>

#include "compat.cuh"

template <typename scalar_t>
__global__ void batch_diag_cuda_kernel(
	scalar_t* __restrict__ index, size_t elems_max, size_t col_max, size_t numel) {
  
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    index[i] += index[i];
  }
}

std::tuple<at::Tensor> batch_diag_cuda(at::Tensor index, size_t elems_max, size_t col_max) {
  cudaSetDevice(index.get_device());

  const auto numel = index.numel() / 2;
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(index.scalar_type(), "batch_diag_cuda_kernel", [&] {
    batch_diag_cuda_kernel<scalar_t><<<blocks, threads>>>(
        index.DATA_PTR<scalar_t>(), elems_max, col_max, numel);
  });

  return std::make_tuple(index);
}
