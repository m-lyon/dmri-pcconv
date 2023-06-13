'''PCConv-Bv variant'''

import torch

from dmri_pcconv.core.model.layers.pcconv import AngularKernel, PCConv, PCConvFactorised


class AngularKernelBv(AngularKernel):
    '''Angular kernel with input and output bval components'''

    @staticmethod
    def _get_bval_component(norm_ang_in: torch.Tensor, norm_ang_out: torch.Tensor) -> torch.Tensor:
        '''Gets normalised input and output bvalues

        Args:
            norm_c_in: Normalising tensor used to normalise coord.
                shape -> (B, q_in, 1)
            norm_c_out: Normalising tensor used to normalise coord.
                shape -> (B, q_out, 1)

        Returns:
            bval_diff: bvalue differences (normalised by dividing by 1000)
                shape -> (B, q_in, q_out, 2)
        '''
        norm_ang_in = norm_ang_in.expand(-1, -1, norm_ang_out.size(1))  # (B, q_in, q_out)
        norm_ang_out = norm_ang_out.permute(0, 2, 1).expand(-1, norm_ang_in.size(1), -1)

        return torch.stack([norm_ang_in, norm_ang_out], dim=-1)


class PCConvBv(PCConv):
    '''Parametric Continuous Convolution with bval modification'''

    nsphdims = 3

    def _get_angular_kernel_class(self):
        return AngularKernelBv()


class PCConvBvFactorised(PCConvFactorised):
    '''Factorised Parametric Continuous Convolution with bval modification'''

    nsphdims = 3

    @property
    def pcconv_class(self):
        return PCConvBv
