'''Module for processing data'''

from typing import Tuple, Dict, Any

import numpy as np
import einops as ein

from nibabel.affines import apply_affine

from npy_patcher import PatcherFloat

from dmri_pcconv.core.utils.padding import get_padding, apply_padding_3d, remove_padding_3d


def get_position_norms(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    '''Gets minimum and maximum spatial co-ordinates within the brain'''
    brain_pts = pts[mask.astype(bool), :]

    def get_minmax(pts, dim):
        return (np.min(pts[:, dim]), np.max(pts[:, dim]))

    minmax = np.empty((brain_pts.shape[1], 2), dtype=np.float32)
    for dim in range(brain_pts.shape[1]):
        minmax[dim] = get_minmax(brain_pts, dim)

    return minmax


def get_patch_positions(
    mask: np.ndarray, affine: np.ndarray, patch_shape: Tuple[int, ...]
) -> np.ndarray:
    '''Gets centroid patch position array

    Args:
        mask: Mask data. shape -> (i, j, k)
        affine: Affine transform. shape -> (4, 4)
        patch_shape: Patch shape

    Returns:
        centroids: Centre patch spatial co-ordinate
            shape -> (N, 3)
    '''
    padding = get_padding(mask.shape, patch_shape)

    # Generate grid
    def get_linspace(dim, pad):
        start, stop = 0 - pad[0], dim - 1 + pad[1]
        return np.linspace(start, stop, dim + pad[0] + pad[1], dtype=np.float32)

    linspace_tuple = (get_linspace(dim, padding[i]) for i, dim in enumerate(mask.shape))
    grid = np.array(np.meshgrid(*linspace_tuple, indexing='ij'))
    grid = grid.transpose(1, 2, 3, 0)
    pts = apply_affine(affine.astype(np.float32), grid)

    # Get min & max co-ordinates
    minmax = get_position_norms(apply_padding_3d(mask, padding), pts)

    # Rearrange grid to patches
    i, j, k = patch_shape
    pts = ein.rearrange(pts, '(ix i) (jx j) (kx k) v -> (ix jx kx) i j k v', i=i, j=j, k=k)

    # Normalise co-ordinates to between [0, 1]
    normed_pts = (pts - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0])

    # Get centroids
    centroids = np.mean(normed_pts, axis=(1, 2, 3))

    return centroids


class TrainingPatcher:
    '''Wrapper class for C++ patcher object'''

    def __init__(self, patch_shape: Tuple[int, int, int]):
        self._patcher = PatcherFloat()
        self.patch_shape = patch_shape

    def get_patch(
        self,
        dmri_fpath: str,
        qspace_index: np.ndarray,
        pnum: int,
        q_in: int,
        q_out: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Loads patch from disk'''
        patch = self._patcher.get_patch(
            dmri_fpath,
            qspace_index,
            self.patch_shape,
            self.patch_shape,
            pnum,
            padding=(),
        )
        full_patch_shape = (q_in + q_out,) + self.patch_shape
        patch = np.array(patch, dtype=np.float32).reshape(full_patch_shape)

        patch = patch.transpose(1, 2, 3, 0)
        patch_in = patch[..., 0:q_in]
        patch_out = patch[..., q_in:]

        return patch_in, patch_out

    def get_patch_index(self, mask: np.ndarray) -> np.ndarray:
        '''Calculates patch index for entire subject'''
        i, j, k = self.patch_shape
        padding = get_padding(mask.shape, self.patch_shape)
        mask = apply_padding_3d(mask, padding)
        mask = ein.rearrange(mask, '(ix i) (jx j) (kx k) -> (ix jx kx) i j k', i=i, j=j, k=k)
        mask_filter = np.sum(mask, (1, 2, 3), dtype=bool)
        patch_index = np.arange(len(mask), dtype=np.int32)[mask_filter]

        return patch_index


class PredictionPatcher:
    '''Splits data into 3D patches'''

    @staticmethod
    def _combine_data(
        data_array: np.ndarray,
        padding: Tuple[Tuple[int, int], ...],
        orig_shape: Tuple[int, ...],
    ):
        '''Combine data from patches into contiguous 3D volumes

        Args:
            data_array: shape -> (X, m, n, o, fs)
            padding: Nested tuple of padding
                length (inner, outer) for each spatial dimension
            orig_shape: Original spatial
                dimensions (i, j, k)

        Returns:
            data_array: shape -> (i+padi, j+padj, k+padk, fs)
        '''
        nums = []
        for idx, pad in enumerate(padding):
            orig_size = orig_shape[idx]
            patch_size = data_array.shape[idx + 1]

            new_size = orig_size + pad[0] + pad[1]
            assert new_size % patch_size == 0

            nums.append(new_size // patch_size)

        data_array = ein.rearrange(
            data_array,
            '(M N O) m n o fs -> (M m) (N n) (O o) fs',
            M=nums[0],
            N=nums[1],
            O=nums[2],
        )

        return data_array

    @staticmethod
    def _get_mask_filter(
        mask: np.ndarray, patch_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
        # pylint: disable=invalid-name
        m, n, o = patch_shape

        # Get padding
        padding = get_padding(mask.shape, patch_shape)

        # Pad mask
        mask = apply_padding_3d(mask, padding)

        # Rearrange mask
        mask = ein.rearrange(mask, '(mx m) (nx n) (ox o) -> (mx nx ox) m n o', m=m, n=n, o=o)

        # Filter out patches that are not contained within brain mask
        mask_filter = np.sum(mask, (1, 2, 3), dtype=bool)

        return mask_filter, padding

    @staticmethod
    def _slice_dmri(
        dmri: np.ndarray,
        mask_filter: np.ndarray,
        padding: Tuple[Tuple[int, int], ...],
        patch_shape: Tuple[int, ...],
    ) -> np.ndarray:
        # pylint: disable=invalid-name
        m, n, o = patch_shape

        # Pad dMRI
        dmri = apply_padding_3d(dmri, padding)

        # Slice into patches
        dmri = ein.rearrange(
            dmri, '(mx m) (nx n) (ox o) q_in -> (mx nx ox) m n o q_in', n=n, m=m, o=o
        )

        # Apply mask filter
        dmri = dmri[mask_filter, ...]

        return dmri

    @staticmethod
    def _slice_bvec(bvec: np.ndarray, num_patches: int):
        return bvec.T[None, :, :].repeat(num_patches, axis=0)

    @staticmethod
    def _get_filter_order(mask_filter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx_pos = np.arange(len(mask_filter))[mask_filter]
        idx_neg = np.arange(len(mask_filter))[~mask_filter]
        order = np.argsort(np.concatenate([idx_neg, idx_pos]))

        return order, idx_neg

    @classmethod
    def forward(
        cls,
        dataset: Dict[str, Any],
        context: Dict[str, Any],
        patch_shape: Tuple[int, int, int] = (10, 10, 10),
    ):
        '''Slices data into patches of size determined by `patch_shape`

        Args:
             dataset (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
                ...
            context (Dict[str,Any]):
                ...
            patch_shape: Default -> (10, 10, 10)

        Modifies:
            dataset (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                ...
            context (Dict[str,Any]):
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                + 'padding': (Tuple[Tuple[int,int], ...])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                + 'mask_filter': (np.ndarray) -> shape (N,)
                ...
        '''
        print('Slicing data into 3D patches...')
        context['orig_shape'] = dataset['mask'].shape

        context['mask_filter'], context['padding'] = cls._get_mask_filter(
            dataset['mask'], patch_shape
        )
        dataset['dmri_in'] = cls._slice_dmri(
            dataset['dmri_in'],
            context['mask_filter'],
            context['padding'],
            patch_shape,
        )
        dataset['bvec_in'] = cls._slice_bvec(dataset['bvec_in'], dataset['dmri_in'].shape[0])
        dataset['bvec_out'] = cls._slice_bvec(dataset['bvec_out'], dataset['dmri_in'].shape[0])

    @classmethod
    def backward(cls, dataset: Dict[str, Any], context: Dict[str, Any]) -> None:
        '''Combines 3D patches into 3D whole volumes

        Args:
            dataset (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (X, m, n, o, q_out)
                ...
            context (Dict[str,Any]):
                'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                'mask_filter': (np.ndarray) -> shape (N,)
                'orig_shape': (Tuple[int,int,int]) -> i, j, k
                ...

        Modifies:
            dataset (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                ...
            context (Dict[str,Any]):
                - 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                - 'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                - 'mask_filter': (np.ndarray) -> shape (N,)
                ...
        '''
        # pylint: disable=invalid-name
        print('Combining 3D patches into contiguous volumes...')
        orig_shape, padding = context.pop('orig_shape'), context.pop('padding')

        order, idx_neg = cls._get_filter_order(context.pop('mask_filter'))
        m, n, o = dataset['dmri_out'].shape[1:4]
        q_out = dataset['dmri_out'].shape[-1]

        unused_patches = np.full((len(idx_neg), m, n, o, q_out), -1.0, dtype=np.float32)

        # Append real data to unused background
        dataset['dmri_out'] = np.concatenate([unused_patches, dataset['dmri_out']], axis=0)

        # Re-order patches
        dataset['dmri_out'] = dataset['dmri_out'][order, ...]

        # Recombine
        dataset['dmri_out'] = cls._combine_data(dataset['dmri_out'], padding, orig_shape)

        # Remove padding
        dataset['dmri_out'] = remove_padding_3d(dataset['dmri_out'], padding)


class SpatialEncodingPredictionPatcher(PredictionPatcher):
    '''Spatial Encoding Prediction Patcher'''

    @classmethod
    def forward(cls, dataset, context, patch_shape=(10, 10, 10)):
        '''Slices data into patches of size determined by `patch_shape`

        Args:
             dataset (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
                ...
            context (Dict[str,Any]):
                ...
            patch_shape: Default -> (10, 10, 10)

        Modifies:
            dataset (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                + 'centroids': (np.ndarray) -> shape (X, 3)
                ...
            context (Dict[str,Any]):
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                + 'padding': (Tuple[Tuple[int,int], ...])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                + 'mask_filter': (np.ndarray) -> shape (N,)
                ...
        '''
        centroids = get_patch_positions(dataset['mask'], context['affine'], patch_shape)
        super().forward(dataset, context, patch_shape)
        dataset['centroids'] = centroids[context['mask_filter'], :]
