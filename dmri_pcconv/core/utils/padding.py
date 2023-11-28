'''Padding functions for 3D data.'''

import math
import warnings
from typing import Optional, Tuple

import numpy as np


def get_padding(
    orig_shape: Tuple[int, ...],
    patch_shape: Tuple[int, ...],
    strides: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, int], ...]:
    '''Calculates padding needed to ensure whole 3D volume
        can be sliced into 3D patches of shape `patch_shape`.

    Args:
        orig_shape: e.g in 3D: (i, j, k)
        patch_shape: e.g. in 3D (m, n, o)
        strides: e.g. in 3D: (s1, s2, s3)

    Returns:
        padding: Nested tuple of padding
            length (pad1, pad2) for each spatial dimension
    '''
    if strides is None:
        strides = patch_shape
    else:
        warnings.warn(
            'get_padding will calculate the padding needed such that all '
            'input is seen when slicing into patches. '
            '\nNOTE: This is different from "valid" and "same" padding.'
        )

    padding = ()
    for idx, size in enumerate(orig_shape):
        patch_size = patch_shape[idx]
        stride = strides[idx]

        # If length of dimension is less than or equal to the patch size then the padding
        # required will be the amount needed to bring the length of the data to the
        # patch size.
        if size <= patch_size:
            pad = patch_size - size
        else:
            num = math.ceil((size - patch_size) / stride)
            pad = ((num * stride) + patch_size) - size

        if pad == 0:
            padding += ((0, 0),)
        else:
            total_pad = pad
            if total_pad % 2 == 0:
                padding += ((total_pad // 2, total_pad // 2),)
            else:
                padding += (((total_pad // 2) + 1, total_pad // 2),)

    return padding


def apply_padding_3d(data_array: np.ndarray, padding: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    '''Applies padding to data array.

    Args:
        data_array: Array containing data
            with data_array.ndim >= 3 and first three
            dimensions correspond to (i, j, k)
        padding: Nested tuple of padding
            length (inner, outer) for each spatial dimension

    Returns:
        data_array: modified array
            with dimensions of (i+di, j+dj, k+dk, ...)
    '''
    # Get extra dims padding
    extra_dim_pads = tuple(((0, 0) for _ in data_array.shape[3:]))

    # Apply padding
    data_array = np.pad(data_array, padding + extra_dim_pads)

    return data_array


def remove_padding_3d(data_array: np.ndarray, padding: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    '''Removes padding from data_array

    Args:
        data_array: data to remove padding from
            with dimensions of (i+di, j+dj, k+dk, ...)
        padding: Nested tuple of padding
            length (inner, outer) for each spatial dimension

    Returns:
        data_array: modified array
            shape -> (i, j, k, ...)
    '''
    data_array = data_array[
        padding[0][0] or None : -padding[0][1] or None,
        padding[1][0] or None : -padding[1][1] or None,
        padding[2][0] or None : -padding[2][1] or None,
    ]

    return data_array
