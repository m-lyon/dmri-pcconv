'''Normalisation functionality'''

import json
from pathlib import Path
from typing import Dict, Tuple, Union, List

import numpy as np

from dmri_pcconv.core.qspace import QSpaceInfo
from dmri_pcconv.core.qspace import get_shell_filter


XMAX_RATIOS = {
    '1000_to_1000': 1.0,
    '1000_to_2000': 1.66,
    '1000_to_3000': 2.19,
    '2000_to_2000': 1.0,
    '2000_to_3000': 1.32,
    '3000_to_3000': 1.0,
}
XMAX_RATIOS['2000_to_1000'] = 1 / XMAX_RATIOS['1000_to_2000']
XMAX_RATIOS['3000_to_1000'] = 1 / XMAX_RATIOS['1000_to_3000']
XMAX_RATIOS['3000_to_2000'] = 1 / XMAX_RATIOS['2000_to_3000']


class TrainingNormaliser:
    '''Normalising class'''

    @classmethod
    def save_xmax(cls, json_fpath: Union[str, Path], xmax_dict: Dict[int, np.floating]) -> None:
        '''Saves an xmax dict to disk'''
        with open(json_fpath, 'w', encoding='utf-8') as fobj:
            json.dump(xmax_dict, fobj)

    @classmethod
    def load_xmax(cls, json_fpath: Union[str, Path]) -> Dict[int, float]:
        '''Loads xmax dict from disk'''
        with open(json_fpath, 'r', encoding='utf-8') as fobj:
            data: Dict[str, str] = json.load(fobj)
        xmax = {int(key): float(val) for key, val in data.items()}

        return xmax

    @classmethod
    def get_norm_array(cls, xmax_dict, qinfo: QSpaceInfo) -> np.ndarray:
        '''Gets norm array from xmax_dict'''
        norm_array = np.empty(qinfo.total, dtype=np.float32)
        for i, shell in enumerate(qinfo.shells_in):
            norm_array[i : i + qinfo.q_in] = xmax_dict[shell]
        for i, shell in enumerate(qinfo.shells_out):
            j = i + qinfo.in_total
            norm_array[j : j + qinfo.q_out] = xmax_dict[shell]
        return norm_array

    @staticmethod
    def apply(data: np.ndarray, norm: Union[np.floating, np.ndarray]) -> np.ndarray:
        '''Applies normalisation to data'''
        return np.divide(data * 2, norm) - 1

    @classmethod
    def normalise_patches(
        cls, patch_in: np.ndarray, patch_out: np.ndarray, norms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Normalises data such that distribution lies between [-1, 1] approximately

        Args:
            patch_in: shape -> (i, j, k, q_in)
            patch_out: shape -> (i, j, k, q_out)
            norms: shape -> (q_in + q_out,)
        '''
        norms = np.expand_dims(norms, (0, 1, 2))
        patch_in = cls.apply(patch_in, norms[..., 0 : patch_in.shape[-1]])
        patch_out = cls.apply(patch_out, norms[..., patch_in.shape[-1] :])

        return patch_in, patch_out

    @staticmethod
    def calculate_xmax(
        dmri: np.ndarray,
        bval: np.ndarray,
        mask: np.ndarray,
        pcent: Union[int, float] = 99,
        shells: Tuple[int, ...] = (1000, 2000, 3000),
        shell_var: float = 30.0,
    ) -> Dict[int, float]:
        '''Calculates xmax for each shell'''
        xmax_dict = {}
        for shell in shells:
            shell_filter = get_shell_filter(bval, shell, shell_var)
            dmri_shell = dmri[..., shell_filter]
            shell_max = np.percentile(dmri_shell[mask.astype(bool)], pcent)
            xmax_dict[shell] = shell_max
        return xmax_dict


class PredictionNormaliser:
    '''Prediction Normalising class'''

    @staticmethod
    def get_shells_present(bval: np.ndarray, shell_var: float) -> List[int]:
        '''Gets unique shells from bval within a shell variance'''
        shells = []
        for shell in (1000, 2000, 3000):
            # Check if any element in the array is within the specified tolerance of shell_var
            if np.any(np.abs(bval - shell) <= shell_var):
                shells.append(shell)
        return shells

    @staticmethod
    def _get_xmax_from_ratio(context, in_shells, shell):
        xmax_in = context['shell_vars'][in_shells[0]]['xmax']
        try:
            ratio = XMAX_RATIOS[f'{in_shells[0]}_to_{shell}']
        except KeyError as err:
            raise KeyError(f'No xmax ratio found for {in_shells[0]} to {shell}') from err
        return xmax_in / ratio

    @staticmethod
    def apply(data: np.ndarray, norm: Union[np.floating, np.ndarray]) -> np.ndarray:
        '''Applies normalisation to data'''
        return np.divide(data * 2, norm) - 1

    @staticmethod
    def reverse(data: np.ndarray, norm: float) -> np.ndarray:
        '''Reverses normalisation to data'''
        return ((data + 1.0) * norm) / 2.0

    @classmethod
    def forward(cls, datasets, context, pcent: Union[int, float] = 99):
        '''Rescales dMRI data to approximate range [-1,1] independently in each shell
            using percentile of dataset to normalise. Applies b-value modulation
            to b-vectors.

        Args:
            datasets (Dict[str,Any]):
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bval_in': (np.ndarray) -> shape (q_in,)
                'bvec_out': (np.ndarray) -> shape (3, out)
                'bval_out': (np.ndarray) -> shape (q_out,)
                ...
            context (Dict[str,Any]):
                ...
            pcent: Percentage of data to get xmax from


        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (3, q_in)
                ~ 'bvec_out': (np.ndarray) -> shape (3, q_out)
                ...
            context (Dict[str,Any]):
                'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        + 'xmin': (float)
                ...
        '''
        print('Normalizing dMRI data...')

        mask: np.ndarray = datasets['mask']
        in_shells = cls.get_shells_present(datasets['bval_in'], 30.0)
        out_shells = cls.get_shells_present(datasets['bval_out'], 30.0)
        context['shell_vars'] = {shell: {} for shell in sorted(set(in_shells + out_shells))}

        for shell in in_shells:
            # Get shell variables
            shell_idx = get_shell_filter(datasets['bval_in'], shell, 30.0)
            shell_dmri = datasets['dmri_in'][..., shell_idx]
            shell_bvec = datasets['bvec_in'][:, shell_idx]
            shell_bval = datasets['bval_in'][shell_idx]
            # Calculate xmax
            xmax = np.percentile(shell_dmri[mask.astype(bool), :], pcent)
            # Apply rescaling
            datasets['dmri_in'][..., shell_idx] = cls.apply(shell_dmri, xmax)
            # Save scale unit
            context['shell_vars'][shell]['xmax'] = xmax
            # Include bvals in bvector, then normalize
            datasets['bvec_in'][:, shell_idx] = shell_bvec * (shell_bval[None, :] / 1000.0)

        for shell in out_shells:
            # Get shell variables
            shell_idx = get_shell_filter(datasets['bval_out'], shell, 30.0)
            shell_bvec = datasets['bvec_out'][:, shell_idx]
            shell_bval = datasets['bval_out'][shell_idx]
            if shell not in in_shells:
                context['shell_vars'][shell]['xmax'] = cls._get_xmax_from_ratio(
                    context, in_shells, shell
                )
            # Include bvals in bvector, then normalize
            datasets['bvec_out'][:, shell_idx] = shell_bvec * (shell_bval[None, :] / 1000.0)

    @classmethod
    def backward(cls, datasets, context):
        '''Rescales dMRI data back to original intensity range

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                'bval_out': (np.ndarray) -> shape (q_out,)
                ...
            context (Dict[str,Any]):
                'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        'xmin': (float)
                ...

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                ...
            context (Dict[str,Any]):
                'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        - 'xmax': (float)
                ...
        '''
        print('Rescaling dMRI data to original intensity range...')
        out_shells = cls.get_shells_present(datasets['bval_out'], 30.0)

        for shell in out_shells:
            shell_idx = get_shell_filter(datasets['bval_out'], shell, 30.0)
            shell_dmri = datasets['dmri_out'][..., shell_idx]
            # Get xmax
            xmax = context['shell_vars'][shell].pop('xmax')
            # Apply rescaling
            datasets['dmri_out'][..., shell_idx] = cls.reverse(shell_dmri, xmax)
