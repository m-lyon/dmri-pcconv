'''PCCNN-Bv-Sp Model'''

from typing import Type

from dmri_pcconv.core.model.layers.pcconv_bv_sp import PCConvBvSp, PCConvBvSpFactorised
from dmri_pcconv.core.model.pccnn_sp import PCCNNSp, PCConvSpBlock


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
