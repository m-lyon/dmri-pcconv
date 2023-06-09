'''Miscellaenous PyTorch Layers'''

from typing import Tuple

import torch


class RepeatBVector(torch.nn.Module):
    '''Repeats B-vector'''

    def __init__(self, spatial_dims: Tuple[int, ...]) -> None:
        '''RepeatBVector, expands bvec from (B, q, 3) -> (B, q, ..., 3)'''
        super().__init__()
        self.spatial_dims = spatial_dims

    def forward(self, bvec: torch.Tensor) -> torch.Tensor:
        '''Forward pass'''
        bvec = bvec.view(*bvec.shape[:2], *((1,) * len(self.spatial_dims)), 3)
        bvec = bvec.expand(-1, -1, *self.spatial_dims, -1)

        return bvec


class RepeatTensor(torch.nn.Module):
    '''Repeats Tensor'''

    def __init__(self, axis: int, num: int) -> None:
        super().__init__()
        self.axis = axis
        self.num = num

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        '''Forward pass'''
        tensor = tensor.unsqueeze(self.axis)
        expand_list = [
            -1,
        ] * len(tensor.shape)
        expand_list[self.axis] = self.num
        tensor = tensor.expand(*tuple(expand_list))

        return tensor
