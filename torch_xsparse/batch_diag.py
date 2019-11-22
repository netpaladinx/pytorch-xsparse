import torch

if torch.cuda.is_available():
    import torch_xsparse.batch_diag_cuda


def batch_diag(index, elems_max, col_max):
    index = index.contiguous().view(-1)

    if index.is_cuda:
        out, = torch_xsparse.batch_diag_cuda.batch_diag(index, elems_max, col_max)
    else:
        out = index

    return out.view(2, -1)
