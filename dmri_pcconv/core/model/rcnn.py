'''3D RCNN Model implementation'''

from typing import Tuple, Optional

import torch

from dmri_pcconv.core.model.layers.misc import RepeatBVector, RepeatTensor
from dmri_pcconv.core.model.layers.conv3dlstm import Conv3DLSTM
from dmri_pcconv.core.model.layers.distributed import DistributedConv3D


class Encoder3D(torch.nn.Module):
    '''3D RCNN Decoder'''

    def __init__(
        self, q_in: int, in_shape: Tuple[int, int, int] = (10, 10, 10), lstm_size: int = 48
    ) -> None:
        super().__init__()
        self.q_in = q_in
        self.lstm_size = lstm_size

        self.repeat_bvec = RepeatBVector(in_shape)

        self.conv1 = DistributedConv3D(4, 200, 1, instance_norm=True)

        self.conv21 = DistributedConv3D(200, 104, 1, instance_norm=True)
        self.conv22 = DistributedConv3D(200, 200, 2, instance_norm=True)
        self.conv23 = DistributedConv3D(200, 72, 3, instance_norm=True)

        self.conv31 = DistributedConv3D(380, 280, 1, batch_norm=True)
        self.conv32 = DistributedConv3D(380, 240, 2, batch_norm=True)
        self.conv33 = DistributedConv3D(380, 144, 3, batch_norm=True)

        self.conv4 = DistributedConv3D(668, 32, 1, batch_norm=True)
        self.conv5 = DistributedConv3D(32, 88, 1, batch_norm=True)

        self.conv_lstm = Conv3DLSTM(in_shape, 88, self.lstm_size, 1)

    def forward(self, dmri_in: torch.Tensor, bvec_in: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass'''
        # Duplicate B-vector so each pixel has B-vector
        bvec_in = self.repeat_bvec(bvec_in)

        # Reshape q-space dimension next to batch
        dmri_in = dmri_in.permute(0, 4, 1, 2, 3, 5)

        # Concatenate B-Vector with input image
        dmri_in = torch.concat([dmri_in, bvec_in], dim=-1)

        # Reshape channels to infront of spatial dims
        dmri_in = dmri_in.permute(0, 1, 5, 2, 3, 4)

        # Initial B-vector convolution
        conv_tensor = self.conv1(dmri_in)

        # Subsequent convolutional layers to extract features
        conv1_tensor = self.conv21(conv_tensor)
        conv2_tensor = self.conv22(conv_tensor)
        conv3_tensor = self.conv23(conv_tensor)
        conv_tensor = torch.concat([conv1_tensor, conv2_tensor, conv3_tensor, dmri_in], dim=2)

        # Second convolutional block
        conv1_tensor = self.conv31(conv_tensor)
        conv2_tensor = self.conv32(conv_tensor)
        conv3_tensor = self.conv33(conv_tensor)
        conv_tensor = torch.concat([conv1_tensor, conv2_tensor, conv3_tensor, dmri_in], dim=2)

        # Compress to latent space
        conv_tensor = self.conv4(conv_tensor)
        conv_tensor = self.conv5(conv_tensor)

        # ConvLSTM
        conv_tensor = self.conv_lstm(conv_tensor)

        return conv_tensor


class Decoder3D(torch.nn.Module):
    '''3D RCNN Decoder'''

    def __init__(
        self,
        q_out: int,
        out_shape: Tuple[int, int, int] = (10, 10, 10),
        lstm_size: int = 48,
        final_nl: Optional[str] = 'relu',
    ) -> None:
        super().__init__()
        self.repeat_tensor = RepeatTensor(1, q_out)
        self.repeat_bvec = RepeatBVector(out_shape)
        self.lstm_size = lstm_size

        self.conv1 = DistributedConv3D(self.lstm_size + 3, 176, 1, batch_norm=True)
        self.conv2 = DistributedConv3D(176, 224, 1, batch_norm=True)

        self.conv31 = DistributedConv3D(227, 240, 1, batch_norm=True)
        self.conv32 = DistributedConv3D(227, 256, 2, batch_norm=True)
        self.conv33 = DistributedConv3D(227, 136, 3, batch_norm=True)

        self.conv41 = DistributedConv3D(635, 176, 1, batch_norm=True)
        self.conv42 = DistributedConv3D(635, 136, 2, batch_norm=True)
        self.conv43 = DistributedConv3D(635, 88, 3, batch_norm=True)

        self.conv5 = DistributedConv3D(403, 16, 1)
        self.conv6 = DistributedConv3D(16, 1, 1, activation=final_nl)

    def forward(self, latent_tensor: torch.Tensor, bvec_out: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass'''
        latent_tensor = self.repeat_tensor(latent_tensor)
        bvec_out = self.repeat_bvec(bvec_out)
        # Reshape channels to infront of spatial dims
        bvec_out = bvec_out.permute(0, 1, 5, 2, 3, 4)

        latent_tensor = torch.concat([latent_tensor, bvec_out], dim=2)

        # Initial B-vector convolution
        latent_tensor = self.conv1(latent_tensor)
        latent_tensor = self.conv2(latent_tensor)
        latent_tensor = torch.concat([latent_tensor, bvec_out], dim=2)

        # Subsequent convolutional layers to extract features
        conv1_tensor = self.conv31(latent_tensor)
        conv2_tensor = self.conv32(latent_tensor)
        conv3_tensor = self.conv33(latent_tensor)
        latent_tensor = torch.concat([conv1_tensor, conv2_tensor, conv3_tensor, bvec_out], dim=2)

        # Second convolutional block
        conv1_tensor = self.conv41(latent_tensor)
        conv2_tensor = self.conv42(latent_tensor)
        conv3_tensor = self.conv43(latent_tensor)
        latent_tensor = torch.concat([conv1_tensor, conv2_tensor, conv3_tensor, bvec_out], dim=2)

        latent_tensor = self.conv5(latent_tensor)
        latent_tensor = self.conv6(latent_tensor)

        # Reshape channels & q-space to end of spatial dims
        out = latent_tensor.permute(0, 3, 4, 5, 1, 2)

        return out


class RCNN(torch.nn.Module):
    '''RCNN Model'''

    def __init__(
        self,
        q_in: int,
        q_out: int,
        shape: Tuple[int, int, int] = (10, 10, 10),
        lstm_size: int = 48,
        final_nl: Optional[str] = 'relu',
    ) -> None:
        super().__init__()
        self.encoder = Encoder3D(q_in, shape, lstm_size)
        self.decoder = Decoder3D(q_out, shape, lstm_size, final_nl)

    def forward(
        self, dmri_in: torch.Tensor, bvec_in: torch.Tensor, bvec_out: torch.Tensor
    ) -> torch.Tensor:
        '''Runs forward pass'''

        # Add channel dimension to input dMRI images
        dmri_in = dmri_in.unsqueeze(-1)

        latent_tensor = self.encoder(dmri_in, bvec_in)
        dmri_out = self.decoder(latent_tensor, bvec_out)

        dmri_out = dmri_out.squeeze(-1)

        return dmri_out
