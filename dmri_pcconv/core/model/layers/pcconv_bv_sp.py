'''Parametric continuous convolution with bval modification and mean spatial encoding'''

from dmri_pcconv.core.model.layers.pcconv_sp import PCConvSp, PCConvSpFactorised
from dmri_pcconv.core.model.layers.pcconv_bv import AngularKernelBv


class PCConvBvSp(PCConvSp):
    '''Parametric Continuous Convolution with bval modification and mean spatial encoding'''

    nsphdims = 3

    def _get_angular_kernel_class(self):
        return AngularKernelBv()


class PCConvBvSpFactorised(PCConvSpFactorised):
    '''Factorised Parametric Continuous Convolution
    with bval modification and mean spatial encoding
    '''

    nsphdims = 3

    @property
    def pcconv_class(self):
        return PCConvBvSp
