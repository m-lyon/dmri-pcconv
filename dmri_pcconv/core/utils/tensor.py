'''PyTorch Tensor utilities'''

import torch


def split_tensor(tensor: torch.Tensor, dim: int, len0: int, len1: int):
    '''Splits tensor along a dimension'''
    shape = list(tensor.shape)
    orig_dim = shape[dim]
    if (len0 != -1) and (len1 != -1):
        assert len0 * len1 == orig_dim, 'dim0 * dim1 != orig dim'
    shape[dim] = len0
    shape.insert(dim + 1, len1)
    return tensor.view(shape)


def merge_dims(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    '''Merges `dim` and `dim + 1` together'''
    truedim = list(range(tensor.ndim))[dim]
    shape = list(tensor.shape)
    len0, len1 = shape[truedim : truedim + 2]
    shape[truedim] = len0 * len1
    del shape[truedim + 1]
    return tensor.reshape(shape)
