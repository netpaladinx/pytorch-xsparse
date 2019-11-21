#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor batch_diag_cuda(torch::Tensor index, size_t elems_max, size_t col_max);

torch::Tensor batch_diag(torch::Tensor index, size_t elems_max, size_t col_max) {
  CHECK_INPUT(index);
  return batch_diag_cuda(index, elems_max, col_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("batch_diag", &batch_diag, "Diagonalizes index on batch (CUDA)");
}