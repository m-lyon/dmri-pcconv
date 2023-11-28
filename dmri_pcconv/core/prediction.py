'''Prediction module'''

from typing import Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl

from dmri_pcconv.core.io import load_bval, load_bvec, load_nifti, save_nifti
from dmri_pcconv.core.normalisation import PredictionNormaliser
from dmri_pcconv.core.patcher import PredictionPatcher, SpatialEncodingPredictionPatcher
from dmri_pcconv.core.utils.tensor import split_tensor, merge_dims


class PCCNNPredictDataset(Dataset):
    '''Torch Dataset for PCCNN Prediction'''

    def __init__(
        self,
        dmri_in: np.ndarray,
        bvec_in: np.ndarray,
        bvec_out: np.ndarray,
        out_num: int,
    ):
        '''Initialises dataset

        Args:
            dmri_in: shape -> (X, m, n, o, in_num_total)
            bvec_in: shape -> (X, in_num_total, 3)
            bvec_out: shape -> (X, out_num_total, 3)
            out_num: Batch out_num.
        '''
        super().__init__()
        self.dmri_in = dmri_in
        self.bvec_in = bvec_in
        self.bvec_out = bvec_out
        self.out_num = out_num
        self._assert_valid_out_num()
        self._list = []
        self._qdx_list = []

    def _assert_valid_out_num(self) -> None:
        if self.bvec_out.shape[1] % self.out_num != 0:
            err_msg = 'Mismatch of bvec_out size and out_num'
            err_msg += f'\n{self.bvec_out.shape[1]} % {self.out_num} != 0'
            raise NotImplementedError(err_msg)
        self.num_qspace_batches = self.bvec_out.shape[1] // self.out_num

    def __len__(self) -> int:
        return len(self.dmri_in) * self.num_qspace_batches

    @staticmethod
    def get_rel_idx(idx: int, num_qspace_batches: int, out_num: int) -> Tuple[int, int, int]:
        '''Gets relative patch and qspace indexes'''
        patch_idx = idx // num_qspace_batches
        qspace_idx = idx % num_qspace_batches
        start_qdx = qspace_idx * out_num
        end_qdx = (qspace_idx + 1) * out_num
        return patch_idx, start_qdx, end_qdx

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._list.append(idx)
        patch_idx, start_qdx, end_qdx = self.get_rel_idx(idx, self.num_qspace_batches, self.out_num)
        self._qdx_list.append((patch_idx, start_qdx, end_qdx))

        dmri_in, bvec_in = self.dmri_in[patch_idx, ...], self.bvec_in[patch_idx, ...]
        bvec_out = self.bvec_out[patch_idx, start_qdx:end_qdx, :]

        return dmri_in, bvec_in, bvec_out


class PCCNNSpPredictDataset(PCCNNPredictDataset):
    '''CombinedShell Predict Dataset with Spatial Encoding'''

    def __init__(self, dmri_in, bvec_in, bvec_out, centroid, out_num):
        self.centroid = centroid
        super().__init__(dmri_in, bvec_in, bvec_out, out_num)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_tup = super().__getitem__(idx)
        patch_idx, _, _ = self.get_rel_idx(idx, self.num_qspace_batches, self.out_num)
        patch_pos = self.centroid[patch_idx, :]
        return data_tup + (patch_pos,)  # type: ignore


class PCCNNPredictionProcessor:
    '''PCCNN Prediction Processor'''

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 8,
        accelerator: str = 'gpu',
    ):
        '''Initializes processor object

        Args:
            batch_size (int): Batch size for prediction
            num_workers (int): Number of CPU workers for dataloader
            accelerator (str): Accelerator to use for prediction, either 'gpu' or 'cpu'.
        '''
        self.batch_size = batch_size
        self.workers = num_workers
        self.accelerator = accelerator
        # Hyperparameters used by the model. If changed,
        # the model training & architecture must reflect this.
        self.patch_shape = (10, 10, 10)
        self.pcent = 99
        self.out_num = 10

    def load_dataset(
        self,
        dmri_in: Union[str, Path],
        bvec_in: Union[str, Path],
        bval_in: Union[str, Path],
        bvec_out: Union[str, Path],
        bval_out: Union[str, Path],
        mask: Union[str, Path],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        '''Loads dataset into data dict

        Args:
            dmri_in: Path to NIfTI dMRI file
            bvec_in: Path to Input bvec file
            bval_in: Path to Input bval file
            bvec_out: Path to Target bvec file
            bval_out: Path to Target bval file
            mask: Path to NIfTI brain mask file

        Returns:
            dataset:

        '''
        dmri_data, affine = load_nifti(dmri_in, dtype=np.float32)
        mask_data, _ = load_nifti(mask, dtype=np.int8)
        bvec_in_data, bvec_out_data = load_bvec(bvec_in), load_bvec(bvec_out)
        bval_in_data, bval_out_data = load_bval(bval_in), load_bval(bval_out)

        dataset = {
            'dmri_in': dmri_data,
            'bvec_in': bvec_in_data,
            'bval_in': bval_in_data,
            'bvec_out': bvec_out_data,
            'bval_out': bval_out_data,
            'mask': mask_data,
        }
        context = {'affine': affine}
        return dataset, context

    @property
    def normaliser(self):
        '''Returns normaliser class'''
        return PredictionNormaliser

    @property
    def patcher(self):
        '''Returns patcher class'''
        return PredictionPatcher

    def preprocess(self, dataset: Dict[str, Any], context: Dict[str, Any]) -> None:
        '''Preprocesses data.

        Args:
            dataset (Dict[str,Any]):
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bval_in': (np.ndarray) -> shape (q_in,)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
                'bval_out': (np.ndarray) -> shape (q_out,)
                'mask': (np.ndarray) -> shape (i, j, k)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)

        Modifies:
            dataset (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                ~ 'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                ...
            context (Dict[str,Any]):
                + 'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        'xmin': (float)
                        'xmax': (float)
                        'index': (np.ndarray) -> shape (fs,)
                + 'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                + 'mask_filter': (np.ndarray) -> shape (N,)
        '''
        self.normaliser.forward(dataset, context, self.pcent)
        self.patcher.forward(dataset, context, self.patch_shape)

    def _get_predict_dataset(self, dataset: Dict[str, Any]) -> PCCNNPredictDataset:
        '''Gets predict dataset'''
        return PCCNNPredictDataset(
            dataset['dmri_in'], dataset['bvec_in'], dataset['bvec_out'], self.out_num
        )

    def run_model(self, dataset, model):
        '''Runs model through inference to produce dMRI outputs

        Args:
            dataset (Dict[str,Any]):
                'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                ...

        Modifies:
            dataset (Dict[str,Any]):
                + 'dmri_out': (np.ndarray) -> shape (X, m, n, o, q_out)
                ...
        '''
        print('Running inference on data...')
        predict_dataset = self._get_predict_dataset(dataset)
        dataloader = DataLoader(predict_dataset, self.batch_size, pin_memory=True, num_workers=8)
        trainer = pl.Trainer(accelerator=self.accelerator, devices=1, logger=False)
        dmri_out = trainer.predict(model, dataloaders=dataloader)
        dmri_out = torch.cat(dmri_out, dim=0)  # type: ignore
        dmri_out = split_tensor(dmri_out, 0, -1, predict_dataset.num_qspace_batches)
        dataset['dmri_out'] = merge_dims(dmri_out.movedim(1, -2), -2)

    def postprocess(self, dataset: Dict[str, Any], context: Dict[str, Any]) -> None:
        '''Postprocesses data.

        Args:
            dataset (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (X, m, n, o, q_out)
                ...
            context (Dict[str,Any]):
                'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        'xmin': (float)
                        'xmax': (float)
                        'index': (np.ndarray) -> shape (fs,)
                'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                'orig_shape': (Tuple[int,int,int]) -> i, j, k
                'mask_filter': (np.ndarray) -> shape (N,)

        Modifies:
            dataset (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                ...
            context (Dict[str,Any]):
                - 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                - 'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                - 'mask_filter': (np.ndarray) -> shape (N,)
                'shell_vars': (Dict[float,Dict])
                    `shell`: (Dict[str,Any])
                        - 'xmax': (float)
        '''
        self.patcher.backward(dataset, context)
        self.normaliser.backward(dataset, context)

    def save_dataset(
        self, dataset: Dict[str, Any], context: Dict[str, Any], out_fpath: Union[str, Path]
    ) -> None:
        '''Saves dataset to disk

        Args:
            dataset (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                ...
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
            out_fpath: Path to save output NIfTI file to
        '''
        print('Saving output to disk...')
        save_nifti(dataset['dmri_out'], context['affine'], out_fpath)

    def run_subject(
        self,
        model: pl.LightningModule,
        dmri_in: Union[str, Path],
        bvec_in: Union[str, Path],
        bval_in: Union[str, Path],
        bvec_out: Union[str, Path],
        bval_out: Union[str, Path],
        mask: Union[str, Path],
        out_fpath: Union[str, Path],
    ):
        '''Runs subject through preprocessing, model inference, and postprocessing.

        Args:
            model: Initialised PyTorch Lightning model
            dmri_in: Path to input NIfTI dMRI file
            bvec_in: Path to input bvec file
            bval_in: Path to input bval file
            bvec_out: Path to target bvec file
            bval_out: Path to tatget bval file
            mask: Path to NIfTI brain mask file
            out_fpath: Path to save output NIfTI file to
            tmp_dir: Path to temporary directory to save FOD file during
                processing, this should be using an SSD if possible. Defaults to using `tempfile`
                module to create a temporary directory.
        '''
        # Load data
        dataset, context = self.load_dataset(dmri_in, bvec_in, bval_in, bvec_out, bval_out, mask)
        # Preprocess
        self.preprocess(dataset, context)
        # Run the model
        self.run_model(dataset, model)
        # Postprocessing
        self.postprocess(dataset, context)
        # Save to disk
        self.save_dataset(dataset, context, out_fpath)


class PCCNNBvPredictionProcessor(PCCNNPredictionProcessor):
    '''PCCNN-Bv Prediction Processor'''


class PCCNNSpPredictionProcessor(PCCNNPredictionProcessor):
    '''PCCNN-Sp Prediction Processor'''

    @property
    def patcher(self):
        return SpatialEncodingPredictionPatcher


class PCCNNBvSpPredictionProcessor(PCCNNPredictionProcessor):
    '''PCCNN-Bv-Sp Prediction Processor'''

    @property
    def patcher(self):
        return SpatialEncodingPredictionPatcher
