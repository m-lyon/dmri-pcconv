'''Dataset classes for PCCNN'''

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Union, Optional

import numpy as np

from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl

from dmri_pcconv.core.io import load_bvec, load_bval, load_nifti
from dmri_pcconv.core.patcher import TrainingPatcher, get_patch_positions
from dmri_pcconv.core.qspace import QSpaceSampler, QSpaceInfo
from dmri_pcconv.core.qspace import spherical_distances, get_shell_filter
from dmri_pcconv.core.normalisation import TrainingNormaliser


@dataclass
class Subject:
    '''Subject Dataclass'''

    dmri_fpath: Union[str, Path]
    bvec_fpath: Union[str, Path]
    bval_fpath: Union[str, Path]
    mask_fpath: Union[str, Path]
    xmax_fpath: Union[str, Path]

    def check_data(self):
        '''Checks all filepaths exist'''
        for fpath in (
            self.dmri_fpath,
            self.bvec_fpath,
            self.bval_fpath,
            self.mask_fpath,
            self.xmax_fpath,
        ):
            if not Path(fpath).is_file():
                raise OSError(f'{fpath} does not exist.')


class PCCNNDataModule(pl.LightningDataModule):
    '''Patcher PyTorch Lightning Data Module'''

    def __init__(
        self,
        train_subjects: Union[List[Subject], Tuple[Subject, ...]],
        val_subjects: Union[List[Subject], Tuple[Subject, ...]],
        qinfo: QSpaceInfo,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.qinfo = qinfo
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._train_dataloader = None
        self._val_dataloader = None
        self.val_dataset: PCCNNTrainingDataset
        self.train_dataset: PCCNNTrainingDataset

    @property
    def dataset_class(self):
        '''Dataset class'''
        return PCCNNTrainingDataset

    def setup(self, stage=None) -> None:
        '''This is run on each GPU'''
        if stage in (None, 'fit'):
            self.train_dataset = self.dataset_class(
                self.train_subjects, self.qinfo, random=True, seed=self.seed
            )
            self._train_dataloader = DataLoader(
                self.train_dataset,
                self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            self.val_dataset = self.dataset_class(self.val_subjects, self.qinfo, random=False)
            self._val_dataloader = DataLoader(
                self.val_dataset,
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


class PCCNNBvDataModule(PCCNNDataModule):
    '''PatcherDataModuleV4 with bvecs modification'''


class PCCNNSpDataModule(PCCNNDataModule):
    '''PatcherDataModuleV4 with spatial encoding'''

    @property
    def dataset_class(self):
        '''Dataset class'''
        return PCCNNSpTrainingDataset


class PCCNNBvSpDataModule(PCCNNSpDataModule):
    '''PatcherDataModuleV4 with bvecs modification and spatial encoding'''


class PCCNNTrainingDataset(Dataset):
    '''Torch Dataset for PCCNN Training'''

    def __init__(
        self,
        subjects: Union[List[Subject], Tuple[Subject, ...]],
        qinfo: QSpaceInfo,
        random=True,
        seed=None,
    ):
        super().__init__()
        self.patch_shape_in = (10, 10, 10)
        self.subjects = subjects
        self.qinfo = qinfo
        self.random = random
        self.seed = seed
        self.patcher = TrainingPatcher(self.patch_shape_in)
        self.qspace_sampler = QSpaceSampler(self.qinfo, self.random, self.seed)
        self.normaliser = TrainingNormaliser
        self._total_len = 0
        self._total_index: np.ndarray
        self._init_dataset()

    def _init_dataset(self):
        '''Initialises dataset'''
        self.subject_data = []
        for subject in self.subjects:
            subject.check_data()
        self._load_subject_data()
        self._filter_shells()
        self._calculate_patches()
        self._calculate_sph_dists()
        self._calculate_total_length()
        self._set_total_index()

    def _load_subject_data(self):
        '''Loads bvecs, bvals, & mask data'''
        for subject in self.subjects:
            mask, affine = load_nifti(subject.mask_fpath)
            self.subject_data.append(
                {
                    'dmri': str(subject.dmri_fpath),
                    'bvec': load_bvec(subject.bvec_fpath),
                    'bval': load_bval(subject.bval_fpath),
                    'mask': mask,
                    'xmax': self.normaliser.load_xmax(subject.xmax_fpath),
                    'affine': affine,
                }
            )

    def _filter_shells(self):
        for data in self.subject_data:
            index = np.arange(len(data['bval']), dtype=np.int32)
            data['bvec_shells'], data['shell_index'] = {}, {}
            for shell in self.qinfo.candidate_shells:
                shell_filter = get_shell_filter(data['bval'], shell, 30.0)
                data['bvec_shells'][shell] = data['bvec'][:, shell_filter]
                data['bvec'][:, shell_filter] = np.multiply(
                    data['bvec_shells'][shell], data['bval'][shell_filter] / 1000.0
                )
                data['shell_index'][shell] = index[shell_filter]
            data['bvec_normed'] = data.pop('bvec')

    def _calculate_patches(self):
        for data in self.subject_data:
            data['patch_index'] = self.patcher.get_patch_index(data.pop('mask'))

    def _calculate_sph_dists(self):
        for data in self.subject_data:
            data['sph_dist_shells'] = {}
            for shell in self.qinfo.candidate_shells:
                data['sph_dist_shells'][shell] = spherical_distances(data['bvec_shells'][shell])

    def _calculate_total_length(self):
        for data in self.subject_data:
            self._total_len += len(data['patch_index'])

    def _set_total_index(self):
        total_list = []
        for i, data in enumerate(self.subject_data):
            local_patch_idx = np.arange(len(data['patch_index']), dtype=int)
            subject_idx = np.full(len(local_patch_idx), i, dtype=int)
            total_list.append(np.stack([subject_idx, local_patch_idx], axis=1))
        self._total_index = np.concatenate(total_list, axis=0)

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        local_index = self._total_index[idx]
        subject = self.subject_data[local_index[0]]
        return self._get_patch_data(subject, local_index)

    def _get_patch_data(
        self, subject: Dict[str, Any], local_index: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        '''Loads patch data into memory and returns data objects'''
        if self.random:
            self.qinfo.random_sample()
        else:
            self.qinfo.deterministic_sample(local_index[1])
        dmri_fpath, bvec, norms = self._get_subject_idx_data(subject)
        pnum = subject['patch_index'][local_index[1]]
        qspace_index = self._get_qspace_index(subject)
        patch_in, patch_out = self.patcher.get_patch(
            dmri_fpath, qspace_index, pnum, self.qinfo.in_total, self.qinfo.out_total
        )
        patch_in, patch_out = self.normaliser.normalise_patches(patch_in, patch_out, norms)
        bvec_in, bvec_out = self._get_bvec(bvec, qspace_index)

        patch_in, bvec_in = self._apply_qspace_padding(patch_in, bvec_in)
        return (patch_in, bvec_in, bvec_out), patch_out

    def _get_subject_idx_data(self, subject) -> Tuple[str, np.ndarray, np.ndarray]:
        norms = self.normaliser.get_norm_array(subject['xmax'], self.qinfo)
        dmri_fpath = subject['dmri']
        bvec = subject['bvec_normed']
        return dmri_fpath, bvec, norms

    def _get_qspace_index(self, subject):
        return self.qspace_sampler.get_subject_sample(
            subject['bvec_shells'], subject['shell_index'], subject['sph_dist_shells']
        )

    def _get_bvec(
        self, bvec: np.ndarray, qspace_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        bvec_use = bvec[:, qspace_index]
        bvec_in = bvec_use[:, 0 : self.qinfo.in_total].T
        bvec_out = bvec_use[:, self.qinfo.in_total :].T

        return bvec_in, bvec_out

    def _apply_qspace_padding(
        self, patch_in: np.ndarray, bvec_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Pads qspace input to q_in_max size'''
        if (diff := self.qinfo.q_in_max - self.qinfo.q_in) != 0:
            patch_in_zeros = np.zeros(self.patch_shape_in + (diff,), dtype=np.float32)
            patch_in = np.concatenate([patch_in, patch_in_zeros], axis=-1)
            bvec_in_zeros = np.zeros((diff, 3), dtype=np.float32)
            bvec_in = np.concatenate([bvec_in, bvec_in_zeros], axis=0)
        return patch_in, bvec_in


class PCCNNSpTrainingDataset(PCCNNTrainingDataset):
    '''Torch Dataset for PCCNN Training with spatial encoding'''

    def _get_patch_data(
        self, subject: Dict[str, Any], local_index: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        (patch_in, bvec_in, bvec_out), patch_out = super()._get_patch_data(subject, local_index)
        pnum = subject['patch_index'][local_index[1]]
        patch_pos = subject['patch_pos'][pnum, :]
        return (patch_in, bvec_in, bvec_out, patch_pos), patch_out

    def _calculate_patches(self):
        for data in self.subject_data:
            mask, affine = data.pop('mask'), data.pop('affine')
            data['patch_index'] = self.patcher.get_patch_index(mask)
            data['patch_pos'] = get_patch_positions(mask, affine, self.patch_shape_in)
