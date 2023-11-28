'''PCCNN Model'''

from typing import Type, Tuple

import torch

from dmri_pcconv.core.model.lightning import BaseLightningModule
from dmri_pcconv.core.model.layers.pcconv import PCConv, PCConvFactorised


MODEL_HPARAMS = {
    'in_blocks': {'num': 3, 'out1': 128, 'out3': 128, 'hlayer1': 128, 'hlayer2': 32, 'lnum': 5},
    'out_blocks': {'num': 5, 'out1': 128, 'out3': 32, 'hlayer1': 16, 'hlayer2': 16, 'lnum': 15},
    'pcconv_in': {'hlayer1': 64, 'hlayer2': 64, 'lnum': 15},
    'pcconv_mid': {'hlayer1': 32, 'hlayer2': 256, 'lnum': 15},
    'pcconv_out1': {'out': 64, 'hlayer1': 32, 'hlayer2': 64, 'lnum': 15},
    'pcconv_out2': {'hlayer1': 256, 'hlayer2': 256, 'lnum': 10},
}


class PCConvBlock(torch.nn.Module):
    '''PCConv Block with spatially pointwise and factorised length 3 kernels'''

    @property
    def pcconv_class(self) -> Type[PCConv]:
        '''Parametric Continuous Convolution class'''
        return PCConv

    @property
    def pcconv_factorised_class(self) -> Type[PCConvFactorised]:
        '''Factorised Parametric Continuous Convolution class'''
        return PCConvFactorised

    def __init__(
        self,
        conv_in: int,
        conv1_out: int,
        conv3_out: int,
        hidden_layers: Tuple[int, ...],
        lnum: int,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.relu = torch.nn.ReLU()
        self.pcconv1 = self.pcconv_class(
            in_channels=conv_in,
            out_channels=conv1_out,
            hidden_layers=hidden_layers,
            lnum=lnum,
            spatial_kernel=(1, 1, 1),
        )
        self.pcconv3 = self.pcconv_factorised_class(
            in_channels=conv_in,
            out_channels=conv3_out,
            hidden_layers=hidden_layers,
            lnum=lnum,
            kernel_len=3,
        )

    def forward(
        self, ang_in: torch.Tensor, ang_out: torch.Tensor, f_in: torch.Tensor
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
                shape -> (B, N, q_in, C), where N are the `d` input spatial
                dimensions (Typically 3 though any are supported)

        Returns:
            f_out: Output feature map, shape -> (B, N, q_out, K),
        '''
        f_out_1 = self.pcconv1(ang_in, ang_out, f_in)
        f_out_1 = self.relu(f_out_1)
        f_out_3 = self.pcconv3(ang_in, ang_out, f_in)
        f_out_3 = self.relu(f_out_3)
        f_out = torch.concat([f_out_1, f_out_3], dim=-1)
        if self.residual:
            return self.relu(f_out + f_in)
        return self.relu(f_out)


class PCCNN(torch.nn.Module):
    '''Parametric Continuous Convolutional Neural Network'''

    @property
    def pcconv_class(self) -> Type[PCConv]:
        '''Parametric Continuous Convolution class'''
        return PCConv

    @property
    def pcconv_block_class(self) -> Type[PCConvBlock]:
        '''PCConvBlock class'''
        return PCConvBlock

    def __init__(self):
        super().__init__()
        config = MODEL_HPARAMS
        self.relu = torch.nn.ReLU()
        self.pcconv_in = self.pcconv_class(
            in_channels=1,
            out_channels=config['in_blocks']['out1'] + config['in_blocks']['out3'],
            hidden_layers=(config['pcconv_in']['hlayer1'], config['pcconv_in']['hlayer2']),
            lnum=config['pcconv_in']['lnum'],
            spatial_kernel=(1, 1, 1),
        )

        self.in_blocks = torch.nn.ModuleList()
        for nblocks in range(config['in_blocks']['num']):
            if nblocks == 0:
                conv_in = config['in_blocks']['out1'] + config['in_blocks']['out3']
            else:
                conv_in = config['in_blocks']['out1'] + config['in_blocks']['out3']
            self.in_blocks.append(
                self.pcconv_block_class(
                    conv_in=conv_in,
                    conv1_out=config['in_blocks']['out1'],
                    conv3_out=config['in_blocks']['out3'],
                    hidden_layers=(config['in_blocks']['hlayer1'], config['in_blocks']['hlayer2']),
                    lnum=config['in_blocks']['lnum'],
                )
            )

        self.pcconv_mid = self.pcconv_class(
            in_channels=config['in_blocks']['out1'] + config['in_blocks']['out3'],
            out_channels=config['out_blocks']['out1'] + config['out_blocks']['out3'],
            hidden_layers=(config['pcconv_mid']['hlayer1'], config['pcconv_mid']['hlayer2']),
            lnum=config['pcconv_mid']['lnum'],
            spatial_kernel=(1, 1, 1),
        )

        self.out_blocks = torch.nn.ModuleList()
        for nblocks in range(config['out_blocks']['num']):
            if nblocks == 0:
                conv_in = config['out_blocks']['out1'] + config['out_blocks']['out3']
            else:
                conv_in = config['out_blocks']['out1'] + config['out_blocks']['out3']
            self.out_blocks.append(
                self.pcconv_block_class(
                    conv_in=conv_in,
                    conv1_out=config['out_blocks']['out1'],
                    conv3_out=config['out_blocks']['out3'],
                    hidden_layers=(
                        config['out_blocks']['hlayer1'],
                        config['out_blocks']['hlayer2'],
                    ),
                    lnum=config['out_blocks']['lnum'],
                    residual=(nblocks != 0),
                )
            )

        self.pcconv_out1 = self.pcconv_class(
            in_channels=config['out_blocks']['out1'] + config['out_blocks']['out3'],
            out_channels=config['pcconv_out1']['out'],
            hidden_layers=(config['pcconv_out1']['hlayer1'], config['pcconv_out1']['hlayer2']),
            lnum=config['pcconv_out1']['lnum'],
            spatial_kernel=(1, 1, 1),
        )
        self.pcconv_out2 = self.pcconv_class(
            in_channels=config['pcconv_out1']['out'],
            out_channels=1,
            hidden_layers=(config['pcconv_out2']['hlayer1'], config['pcconv_out2']['hlayer2']),
            lnum=config['pcconv_out2']['lnum'],
            spatial_kernel=(1, 1, 1),
        )

    def forward(
        self, dmri_in: torch.Tensor, bvec_in: torch.Tensor, bvec_out: torch.Tensor
    ) -> torch.Tensor:
        '''Forward pass of the PCCNN model

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

        Returns:
            dmri_out: Target dMRI data, shape -> (B, N, q_out),
        '''
        dmri_in = dmri_in.unsqueeze(-1)

        a_out = self.pcconv_in(bvec_in, bvec_in, dmri_in)
        a_out = self.relu(a_out)

        for in_block in self.in_blocks:
            a_out = in_block(bvec_in, bvec_in, a_out)

        a_out = self.pcconv_mid(bvec_in, bvec_in, a_out)
        a_out = self.relu(a_out)

        for i, out_block in enumerate(self.out_blocks):
            if i == 0:
                a_out = out_block(bvec_in, bvec_out, a_out)
            else:
                a_out = out_block(bvec_out, bvec_out, a_out)

        a_out = self.pcconv_out1(bvec_out, bvec_out, a_out)
        a_out = self.relu(a_out)

        a_out = self.pcconv_out2(bvec_out, bvec_out, a_out)
        dmri_out = a_out.squeeze(-1)

        return dmri_out


class PCCNNLightningModel(BaseLightningModule):
    '''PCCNN Lightning Model'''

    def __init__(self) -> None:
        super().__init__()
        self.pccnn = PCCNN()

    def forward(
        self, dmri_in: torch.Tensor, bvec_in: torch.Tensor, bvec_out: torch.Tensor
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.pccnn(dmri_in, bvec_in, bvec_out)
