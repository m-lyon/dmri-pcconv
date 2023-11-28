'''PCCNN-Sp Model'''

from typing import Type

import torch

from dmri_pcconv.core.model.pccnn import PCCNN, PCConvBlock
from dmri_pcconv.core.model.lightning import BaseLightningModule
from dmri_pcconv.core.model.layers.pcconv_sp import PCConvSp, PCConvSpFactorised


class PCConvSpBlock(PCConvBlock):
    '''PCConv Block with spatially pointwise and factorised length 3 kernels
    and mean spatial encoding
    '''

    @property
    def pcconv_class(self) -> Type[PCConvSp]:
        return PCConvSp

    @property
    def pcconv_factorised_class(self) -> Type[PCConvSpFactorised]:
        return PCConvSpFactorised

    def forward(
        self,
        ang_in: torch.Tensor,
        ang_out: torch.Tensor,
        f_in: torch.Tensor,
        patch_pos: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward pass PCConvBlock

        Args:
            ang_in: Input angular co-ordinates,
                shape -> (B, q_in, 3) where last dimension has
                    cartesian co-ords (x, y, z)
            ang_out: Output angular co-ordinates,
                shape -> (B, q_out, 3) where last dimension has
                    cartesian co-ords (x, y, z)
            f_in: Input feature map
                shape -> (B, N, q_in, C_in), where N are the `d` input spatial
                dimensions (Typically 3 though any are supported)
            patch_pos: Mean spatial position
                shape -> (B, d)

        Returns:
            f_out: Output feature map, shape -> (B, M, q_out, S),
                where M are the `d` output spatial dimensions.
        '''
        # pylint: disable=arguments-differ
        f_out_1 = self.pcconv1(ang_in, ang_out, f_in, patch_pos)
        f_out_1 = self.relu(f_out_1)
        f_out_3 = self.pcconv3(ang_in, ang_out, f_in, patch_pos)
        f_out_3 = self.relu(f_out_3)
        f_out = torch.concat([f_out_1, f_out_3], dim=-1)
        if self.residual:
            return self.relu(f_out + f_in)
        return self.relu(f_out)


class PCCNNSp(PCCNN):
    '''Parametric Continuous Convolutional Neural Network with mean spatial encoding'''

    @property
    def pcconv_class(self) -> Type[PCConvSp]:
        return PCConvSp

    @property
    def pcconv_block_class(self) -> Type[PCConvSpBlock]:
        return PCConvSpBlock

    def forward(
        self,
        dmri_in: torch.Tensor,
        bvec_in: torch.Tensor,
        bvec_out: torch.Tensor,
        patch_pos: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward pass of the PCCNN-Sp model

        Args:
            dmri_in: Input dMRI data
                shape -> (B, N, q_in), where N are the `d` spatial
                dimensions (Typically 3 though any are supported)
            bvec_in: Input b-vectors associated with `dmri_in`,
                shape -> (B, q_in, 3) where last dimension has
                cartesian co-ords (x, y, z)
            bvec_out: Target b-vectors that `dmri_out` should refer to,
                shape -> (B, q_out, 3) where last dimension has
                cartesian co-ords (x, y, z)
            patch_pos: Mean spatial position
                shape -> (B, d)

        Returns:
            dmri_out: Target dMRI data, shape -> (B, N, q_out),
        '''
        # pylint: disable=arguments-differ
        dmri_in = dmri_in.unsqueeze(-1)

        a_out = self.pcconv_in(bvec_in, bvec_in, dmri_in, patch_pos)
        a_out = self.relu(a_out)

        for in_block in self.in_blocks:
            a_out = in_block(bvec_in, bvec_in, a_out, patch_pos)

        a_out = self.pcconv_mid(bvec_in, bvec_in, a_out, patch_pos)
        a_out = self.relu(a_out)

        for i, out_block in enumerate(self.out_blocks):
            if i == 0:
                a_out = out_block(bvec_in, bvec_out, a_out, patch_pos)
            else:
                a_out = out_block(bvec_out, bvec_out, a_out, patch_pos)

        a_out = self.pcconv_out1(bvec_out, bvec_out, a_out, patch_pos)
        a_out = self.relu(a_out)

        a_out = self.pcconv_out2(bvec_out, bvec_out, a_out, patch_pos)
        dmri_out = a_out.squeeze(-1)

        return dmri_out


class PCCNNSpLightningModel(BaseLightningModule):
    '''PCCNN-Sp Lightning Model'''

    def __init__(self) -> None:
        super().__init__()
        self.pccnn = PCCNNSp()

    def forward(
        self,
        dmri_in: torch.Tensor,
        bvec_in: torch.Tensor,
        bvec_out: torch.Tensor,
        patch_pos: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.pccnn(dmri_in, bvec_in, bvec_out, patch_pos)
