'''Parametric continuous convolution with bval modification and mean spatial encoding'''

from typing import Type

from dmri_pcconv.core.model.layers.pcconv_sp import PCConvSp, PCConvSpFactorised
from dmri_pcconv.core.model.layers.pcconv_bv import AngularKernelBv


class PCConvBvSp(PCConvSp):
    '''Parametric Continuous Convolution with bval modification and mean spatial encoding'''

    @property
    def nsphdims(self) -> int:
        return 3

    def _get_angular_kernel_class(self) -> AngularKernelBv:
        return AngularKernelBv()


class PCConvBvSpFactorised(PCConvSpFactorised):
    '''Factorised Parametric Continuous Convolution
    with bval modification and mean spatial encoding
    '''

    @property
    def nsphdims(self) -> int:
        return 3

    @property
    def pcconv_class(self) -> Type[PCConvBvSp]:
        return PCConvBvSp
