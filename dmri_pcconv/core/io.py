'''dMRI I/O functionality'''

from pathlib import Path
from typing import Union, Type, Tuple, Optional

import numpy as np
import nibabel as nib


def load_bvec(fpath: Union[str, Path]) -> np.ndarray:
    '''Loads bvec into numpy array

    Args:
        fpath: path to bvec file

    Returns:
        bvec: b-vector array, shape -> (3, b)
    '''
    bvec = np.genfromtxt(fpath, dtype=np.float32)
    if bvec.shape[1] == 3:
        bvec = bvec.T

    return bvec


def save_bvec(bvec: np.ndarray, fpath: Union[str, Path]) -> None:
    '''Saves bvec to file in shape (3, b)

    Args:
        bvec: bvec array, accepts shapes
            (3, b) or (b, 3).
        fpath: filepath to save bvec to.
    '''
    if bvec.shape[1] == 3:
        bvec = bvec.T

    np.savetxt(fpath, bvec, fmt='%1.6f')


def load_bval(fpath: Union[str, Path]) -> np.ndarray:
    '''Loads bval into numpy array

    Args:
        fpath: path to bvec file

    Returns:
        bval: bval array, shape -> (b,)
    '''
    return np.genfromtxt(fpath, dtype=np.float32)


def save_bval(bval: np.ndarray, fpath: Union[str, Path]) -> None:
    '''Saves bval to file

    Args:
        bval: bval array shape -> (b,)
        fpath: filepath to save bval to
    '''
    np.savetxt(fpath, bval, newline=' ', fmt='%g')


def load_nifti(
    nifti_fpath: Union[str, Path], dtype: Type = np.float32, force_ras: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads NIfTI image into memory

    Args:
        nifti_fpath: Filepath to nifti image
        dtype: Datatype to load array with.
            Default: `np.float32`
        force_ras: Forces data into RAS data ordering scheme.
            Default: `False`.

    Returns:
        data (np.ndarray): image data
        affine (np.ndarray): affine transformation -> shape (4, 4)
    '''
    img: nib.nifti2.Nifti2Image = nib.load(nifti_fpath)  # type: ignore
    if force_ras:
        if nib.orientations.aff2axcodes(img.affine) != ('R', 'A', 'S'):
            print(f'Converting {img.get_filename()} to RAS co-ords')
            img = nib.funcs.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=dtype)

    return data, img.affine


def save_nifti(
    data: np.ndarray, affine: np.ndarray, fpath: Union[str, Path], descrip: Optional[str] = None
) -> None:
    '''Saves NIfTI image to disk.

    Args:
        data: Data array
        affine: affine transformation -> shape (4, 4)
        fpath: Filepath to save to.
        descrip: Additional info to add to header description
    '''
    img = nib.nifti2.Nifti2Image(data, affine)

    if descrip is not None:
        img.header['descrip'] = descrip  # type: ignore

    nib.save(img, fpath)  # type: ignore
