'''PCConv Layers'''

from functools import partial
from typing import Optional

import torch


class WeightNet(torch.nn.Module):
    '''Weights HyperNetwork'''

    def __init__(
        self,
        *layer_sizes: int,
        ndims: int = 5,
        lnum: int = 10,
        hlayer_nl: Optional[torch.nn.Module] = None,
        final_nl: Optional[torch.nn.Module] = None
    ) -> None:
        '''Initialises WeightNet

        Positional Args:
            layer_size: Hidden layer size

        Keyword Args:
            ndims: Number of dimensions of the coordinate points
                Default: 5
            lnum: Degree of fourier expansion for each spatial co-ordinate component
            hlayer_nl: Non-linear activation for the hidden
                layers. Default: torch.nn.LeakyReLU with negative_slope=0.1
            final_nl: Non-linear activation for the final
                layer. Default: torch.nn.Identity
        '''
        super().__init__()
        self.ndims = ndims
        self.lnum = lnum
        self.hlayer = self._assign_hlayer(hlayer_nl)
        self.final = torch.nn.Identity if final_nl is None else final_nl
        self.layers = self._init_layers(*layer_sizes)

    @staticmethod
    def _assign_hlayer(hlayer_nl: Optional[torch.nn.Module]) -> torch.nn.Module:
        if hlayer_nl is not None:
            return hlayer_nl
        return partial(torch.nn.LeakyReLU, negative_slope=0.1)

    @property
    def coord_size(self) -> int:
        '''Size of input co-ordinate after fourier features transform'''
        return self.ndims * 2 * self.lnum

    def _init_layers(self, *layer_sizes: int) -> torch.nn.Sequential:
        layer_list = []
        for ldx, layer in enumerate(layer_sizes):
            if ldx > 0:
                layer_list.append(torch.nn.Linear(layer_sizes[ldx - 1], layer))
            else:
                layer_list.append(torch.nn.Linear(self.coord_size, layer))
            if ldx < len(layer_sizes) - 1:
                layer_list.append(self.hlayer())
            else:
                layer_list.append(self.final())

        return torch.nn.Sequential(*tuple(layer_list))

    def _apply_fourier_xfm(self, coords: torch.Tensor) -> torch.Tensor:
        '''Transforms co-ordinates into fourier co-ords

        Args:
            coords: Co-ordinate tensor
                shape -> (B, ..., d)

        Returns:
            fourier_coords: Transformed co-ordinates tensor
                shape -> (B, ..., d*2*lnum)
        '''
        lnums = torch.arange(0, self.lnum).to(coords, non_blocking=True)
        ones = (coords.ndim - 1) * (1,)
        sizes = coords.shape[0 : coords.ndim - 1]
        lnums = lnums.view(*ones, self.lnum, 1).expand(*sizes, self.lnum, self.ndims)
        coords = coords.view(*sizes, 1, -1).expand(*sizes, self.lnum, -1)

        sin = torch.sin((2**lnums) * torch.pi * coords)
        cos = torch.cos((2**lnums) * torch.pi * coords)

        fourier_coords = torch.stack([sin, cos], dim=-2).flatten(start_dim=-3)
        return fourier_coords

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        '''Forward pass

        Args:
            coords: Co-ordinate tensor
                shape -> (B, ..., d)

        Returns:
            weights: shape -> (B, ..., f_in).
        '''
        fourier_coords = self._apply_fourier_xfm(coords)
        return self.layers(fourier_coords)
