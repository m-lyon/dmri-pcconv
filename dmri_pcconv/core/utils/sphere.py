'''Spherical functions'''

import torch


def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''Gets normalised spherical distance

    Args:
        x: input q-space co-ordinate points,
            shape -> (..., m, 3) where last dimension has
                cartesian co-ords (x, y, z)
        y: output q-space co-ordinate points,
            shape -> (..., n, 3) where last dimension has
                cartesian co-ords (x, y, z)

    Returns:
        dists: spherical distances matrix
            shape -> (..., m, n)
    '''
    # pylint: disable=invalid-name
    dotprod = torch.matmul(x, y.transpose(-2, -1))
    dotprod = torch.clamp(dotprod, min=-1.0, max=1.0)
    dists = torch.acos(dotprod)

    return dists
