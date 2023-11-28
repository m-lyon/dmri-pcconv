'''PCCNN-Bv Model'''

from typing import Type

import torch

from dmri_pcconv.core.model.pccnn import PCCNN, PCConvBlock
from dmri_pcconv.core.model.lightning import BaseLightningModule
from dmri_pcconv.core.model.layers.pcconv_bv import PCConvBv, PCConvBvFactorised


class PCConvBvBlock(PCConvBlock):
    '''PCConv Block with spatially pointwise and factorised length 3 kernels
    and bval modification
    '''

    @property
    def pcconv_class(self) -> Type[PCConvBv]:
        return PCConvBv

    @property
    def pcconv_factorised_class(self) -> Type[PCConvBvFactorised]:
        return PCConvBvFactorised


class PCCNNBv(PCCNN):
    '''Parametric Continuous Convolutional Neural Network with bval modification'''

    @property
    def pcconv_class(self) -> Type[PCConvBv]:
        return PCConvBv

    @property
    def pcconv_block_class(self) -> Type[PCConvBvBlock]:
        return PCConvBvBlock


class PCCNNBvLightningModel(BaseLightningModule):
    '''PCCNN-Bv Lightning Model'''

    def __init__(self) -> None:
        super().__init__()
        self.pccnn = PCCNNBv()

    def forward(
        self, dmri_in: torch.Tensor, bvec_in: torch.Tensor, bvec_out: torch.Tensor
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.pccnn(dmri_in, bvec_in, bvec_out)
