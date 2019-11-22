#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

std::tuple<at::Tensor> batch_diag_cuda(at::Tensor index, size_t elems_max, size_t col_max);

std::tuple<at::Tensor> batch_diag(at::Tensor index, size_t elems_max, size_t col_max) {
  CHECK_CUDA(index);
  return batch_diag_cuda(index, elems_max, col_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_diag", &batch_diag, "Batch_diag (CUDA)");
}
