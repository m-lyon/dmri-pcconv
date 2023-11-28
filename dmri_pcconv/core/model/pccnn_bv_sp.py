'''PCCNN-Bv-Sp Model'''

from typing import Type

import torch

from dmri_pcconv.core.model.lightning import BaseLightningModule
from dmri_pcconv.core.model.pccnn_sp import PCCNNSp, PCConvSpBlock
from dmri_pcconv.core.model.layers.pcconv_bv_sp import PCConvBvSp, PCConvBvSpFactorised


class PCConvBvSpBlock(PCConvSpBlock):
    '''PCConv Block with spatially pointwise and factorised length 3 kernels
    with bval modification and mean spatial encoding
    '''

    @property
    def pcconv_class(self) -> Type[PCConvBvSp]:
        return PCConvBvSp

    @property
    def pcconv_factorised_class(self) -> Type[PCConvBvSpFactorised]:
        return PCConvBvSpFactorised


class PCCNNBvSp(PCCNNSp):
    '''Parametric Continuous Convolutional Neural Network
    with bval modification and mean spatial encoding
    '''

    @property
    def pcconv_class(self) -> Type[PCConvBvSp]:
        return PCConvBvSp

    @property
    def pcconv_block_class(self) -> Type[PCConvBvSpBlock]:
        return PCConvBvSpBlock


class PCCNNBvSpLightningModel(BaseLightningModule):
    '''PCCNN-Bv-Sp Lightning Model'''

    def __init__(self) -> None:
        super().__init__()
        self.pccnn = PCCNNBvSp()

    def forward(
        self,
        dmri_in: torch.Tensor,
        bvec_in: torch.Tensor,
        bvec_out: torch.Tensor,
        patch_pos: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.pccnn(dmri_in, bvec_in, bvec_out, patch_pos)
