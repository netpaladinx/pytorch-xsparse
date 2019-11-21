import torch

if torch.cuda.is_available():
    import torch_xsparse.batch_diag_cuda


def batch_diag(index, elems_max, col_max):
    return torch_xsparse.batch_diag_cuda.batch_diag(index.contiguous(), elems_max, col_max)
