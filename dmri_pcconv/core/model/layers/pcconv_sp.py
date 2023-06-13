'''Parametric continuous convolution with mean spatial encoding'''
from typing import Optional

import torch

from dmri_pcconv.core.model.layers.weightnet import WeightNet
from dmri_pcconv.core.model.layers.pcconv import PCConv, PCConvFactorised


class PCConvSp(PCConv):
    '''PCConv with mean spatial encoding'''

    def _get_weightnet(
        self, weightnet: Optional[WeightNet], weightnet_final_nl: Optional[torch.nn.Module]
    ) -> WeightNet:
        if weightnet is not None:
            return weightnet
        final_nl = torch.nn.Identity if weightnet_final_nl is None else weightnet_final_nl

        return WeightNet(
            *self.hidden_layers,
            self.in_channels,
            lnum=self.lnum,
            ndims=(self.nsdims * 2) + self.nsphdims,
            final_nl=final_nl,
        )

    def _expand_for_mean_spatial(self, r_out: torch.Tensor, sp_in: torch.Tensor) -> torch.Tensor:
        '''Expands feature weights

        Args:
            r_out: shape -> (B, q_out, K, q_in, d+2)
            sp_in: Mean spatial co-ordinate of input
                shape ->  (B, d)

        Returns:
            r_out: shape -> (B, q_out, K, q_in, (d*2)+2)
        '''
        ones = (1,) * (self.nsdims + 2)
        sp_in = sp_in.view(-1, *ones, self.nsdims)
        sp_out = sp_in.expand(-1, self.q_out, *self.kernel, self.q_in, -1)

        r_out = torch.concat([r_out, sp_out], dim=-1)
        return r_out

    def forward(self, ang_in, ang_out, f_in, sp_in):
        '''Forward pass of parametric continuous convolution

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
            sp_in: Mean spatial encoding.
                shape -> (B, d)

        Returns:
            f_out: Output feature map, shape -> (B, M, q_out, K),
                where M are the `d` output spatial dimensions.
        '''
        # pylint: disable=arguments-differ
        self._set_runtime_shapes(ang_in, ang_out, f_in)

        p_ang, mask = self.ang_kernel.get_p_ang(ang_in, ang_out, self.d_max, self.k_max)
        p_ang = self._expand_p_ang(p_ang)
        p_spatial = self._get_p_spatial(p_ang)
        p = torch.cat((p_spatial, p_ang), dim=-1)

        p = self._expand_for_mean_spatial(p, sp_in)

        w_in = self.weight_net(p)

        f_in = self._reshape_input_features(f_in)
        w_in = self._expand_conv_weights(w_in)
        f_in = self._apply_and_sum(f_in, w_in, mask)
        f_out = self._apply_feature_layer(f_in)

        return f_out


class PCConvSpFactorised(PCConvFactorised):
    '''Factorised PCConv with mean spatial encoding'''

    @property
    def pcconv_class(self):
        return PCConvSp

    def _get_weightnet(self):
        return WeightNet(
            *self.hidden_layers,
            self.in_channels,
            lnum=self.lnum,
            ndims=(self.nsdims * 2) + self.nsphdims,
        )

    def forward(self, ang_in, ang_out, f_in, sp_in):
        '''Forward pass of factorised parametric continuous convolution

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
            sp_in: Mean spatial encoding.
                shape -> (B, d)

        Returns:
            f_out: Output feature map, shape -> (B, M, q_out, K),
                where M are the `d` output spatial dimensions.
        '''
        # pylint: disable=arguments-differ
        a_out = self.pcconv1(ang_in, ang_in, f_in, sp_in)
        a_out = self.pcconv2(ang_in, ang_in, a_out, sp_in)
        a_out = self.pcconv3(ang_in, ang_in, a_out, sp_in)
        a_out = self.pcconv4(ang_in, ang_out, a_out, sp_in)

        return a_out
