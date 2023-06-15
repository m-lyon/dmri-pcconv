'''TimeDistributed classes'''

from typing import Optional, Union, Tuple

import torch


class TimeDistributed(torch.nn.Module):
    '''TimeDistributed Container module'''

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass'''
        # Squash samples and timesteps into a single axis
        x_reshape = x.view(-1, *x.shape[2:])  # (samples * timesteps, ...)
        y = self.module(x_reshape)

        # reshape Y
        y = y.view(*x.shape[:2], *y.shape[1:])  # (samples, timesteps, ...)
        return y


class DistributedConv3D(torch.nn.Module):
    '''TimeDistributed Conv3D'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        instance_norm: bool = False,
        batch_norm: bool = False,
        activation: Optional[str] = 'swish',
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.instance_norm = instance_norm
        self.batch_norm = batch_norm
        self.activation = activation

        # Initialise layers
        conv_layer = self._get_conv_layer()
        activation_layer = self._get_activation_layer()
        norm_layer = self._get_norm_layer()
        sequential = torch.nn.Sequential(conv_layer)
        if activation_layer is not None:
            sequential.add_module(str(len(sequential)), activation_layer)
        if norm_layer is not None:
            sequential.add_module(str(len(sequential)), norm_layer)

        self.layers = TimeDistributed(sequential)

    def _get_conv_layer(self) -> torch.nn.Module:
        return torch.nn.Conv3d(
            self.in_channels, self.out_channels, self.kernel_size, padding='same'
        )

    def _get_activation_layer(self) -> Union[None, torch.nn.Module]:
        if self.activation is None:
            return None
        act_layers = {
            'swish': torch.nn.SiLU,
            'relu': torch.nn.ReLU,
        }
        assert self.activation in act_layers, f'{self.activation} not in allowed activations.'
        return act_layers[self.activation]()

    def _get_norm_layer(self) -> Union[None, torch.nn.Module]:
        if self.instance_norm:
            return torch.nn.InstanceNorm3d(self.out_channels)
        if self.batch_norm:
            return torch.nn.BatchNorm3d(self.out_channels)
        return None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass'''
        return self.layers(tensor)
