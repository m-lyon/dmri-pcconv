'''PCCNN-Bv Model'''

from typing import Type

from dmri_pcconv.core.model.layers.pcconv_bv import PCConvBv, PCConvBvFactorised
from dmri_pcconv.core.model.pccnn import PCCNN, PCConvBlock


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
