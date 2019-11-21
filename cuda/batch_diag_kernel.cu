#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void batch_diag_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> index,
	torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_index) {
  
  const int idx = 5; //blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < index.size(1)) {
    new_index[0][idx] = index[0][idx];
    new_index[1][idx] = index[1][idx] + index[0][idx];
  }
}

torch::Tensor batch_diag_cuda(torch::Tensor index, size_t elems_max, size_t col_max) {
  auto new_index = torch::clone(index);

  const int threads = 1024;
  const int blocks = (index.size(1) + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(index.type(), "batch_diag_cuda_kernel", ([&] {
    batch_diag_cuda_kernel<scalar_t><<<blocks, threads>>>(
        index.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_index.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return new_index;
}