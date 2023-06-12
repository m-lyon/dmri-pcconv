'''PCConv Layers'''

import math

from typing import Union, Tuple, Optional

import torch

from dmri_pcconv.core.utils.sphere import spherical_distance
from dmri_pcconv.core.model.layers.weightnet import WeightNet


class AngularKernel:
    '''Angular kernel class containing methods to calculate angular kernel'''

    def get_p_ang(
        self, ang_in: torch.Tensor, ang_out: torch.Tensor, d_max: float, k_max: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Gets kernel index and mask array.
            spherical distance is calculated independent of radius
            in `ang_in` & `ang_out`, therefore if `ang_in` is multishell,
            (i.e. multiple r-values), all are used in kernel.

        Args:
            ang_in: Input q-space co-ordinate points,
                shape -> (B, q_in, 3) where last dimension has
                    cartesian co-ords (x, y, z)
            ang_out: Output q-space co-ordinate points,
                shape -> (B, q_out, 3) where last dimension has
                    cartesian co-ords (x, y, z)

        Returns:
            p_ang: angular components of vector `p`
                shape -> (B, q_in, q_out, 2).
                Last dimension [0]: angular distance,
                                [1]: difference in bvalues.
            mask: Combined d_max & k_max mask.
                shape -> (B, q_in, q_out)
        '''
        # Normalise points to unit sphere
        ang_in, norm_ang_in = self._normalise_coord(ang_in)
        ang_out, norm_ang_out = self._normalise_coord(ang_out)

        # Get spherical distances
        dists, _ = self._get_spherical_distances(ang_in, ang_out)

        # Get distance mask to minimum distances from q_out
        mask = self._get_min_dists_mask(dists, d_max, k_max)

        # Get bvalue differences
        bval_diff = self._get_bval_diff(norm_ang_in, norm_ang_out)

        # Get r_out
        p_ang = torch.cat([dists.unsqueeze(-1), bval_diff], dim=-1)

        return p_ang, mask

    @staticmethod
    def _normalise_coord(coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Normalise angular co-ordinate

        Args:
            coord: input q-space co-ordinate points,
                shape -> (B, q, 3) where last dimension has
                    cartesian co-ords (x, y, z)

        Returns:
            coord_normed: Normalised coord tensor
                shape -> (B, q, 3)
            norm: Normalising tensor used to normalise coord.
                shape -> (B, q, 1)
        '''
        norm = torch.norm(coord, p=2, dim=-1, keepdim=True)
        coord_normed = coord.div(norm).nan_to_num()

        return coord_normed, norm

    @staticmethod
    def _get_spherical_distances(
        ang_in: torch.Tensor, ang_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Calculates spherical distances, including antipodal points on the sphere

        Args:
            ang_in: Normalised ang_in coord tensor
                shape -> (B, q_in, 3)
            ang_out: Normalised ang_out coord tensor
                shape -> (B, q_out, 3)

        Returns:
            dists: Spherical distances tensor
                shape -> (B, q_in, q_out)
            ang_in_idx: Ordering index of `ang_in`
        '''
        q_in, q_out = ang_in.size(1), ang_out.size(1)
        # Get whole sphere, therefore include antipodal coords
        ang_in = torch.cat([ang_in, -ang_in], dim=1)  # (B, 2*q_in, 3)
        # Get unit sphere spherical distances
        dists = spherical_distance(ang_in, ang_out)  # (B, 2*q_in, q_out)
        # Get minimum distance on either hemispheres
        dists = dists.reshape(ang_in.size(0), 2, q_in, q_out)  # (B, 2, q_in, q_out)
        dists, ang_in_idx = torch.min(dists, dim=1)

        return dists, ang_in_idx

    @staticmethod
    def _get_bval_diff(norm_ang_in: torch.Tensor, norm_ang_out: torch.Tensor) -> torch.Tensor:
        '''Gets relative difference in bvalues / 1000

        Args:
            norm_ang_in: Normalising tensor used to normalise coord.
                shape -> (B, q_in, 1)
            norm_ang_out: Normalising tensor used to normalise coord.
                shape -> (B, q_out, 1)

        Returns:
            bval_diff: bvalue differences (normalised by dividing by 1000)
                shape -> (B, q_in, q_out, 1)
        '''
        norm_ang_in = norm_ang_in.expand(-1, -1, norm_ang_out.size(1))  # (B, q_in, q_out)
        bval_diff = norm_ang_out.permute(0, 2, 1) - norm_ang_in  # (B, q_in, q_out)

        return bval_diff.unsqueeze(-1)

    @staticmethod
    def _get_min_dists_mask(dists: torch.Tensor, d_max: float, k_max: int) -> torch.Tensor:
        '''Gets minimum spherical distances

        Args:
            dists: Spherical distances tensor
                shape -> (B, q_in, q_out)

        Returns:
            mask: Indexing mask for k_max and distance threshold
                shape -> (B, q_in, q_out)
        '''
        # Sort tensor to minimum distances from q_out
        sorted_dists: torch.Tensor
        indexer_mask: torch.Tensor
        sorted_dists, indexer_mask = dists.sort(dim=1)
        inv_mask = indexer_mask.argsort(dim=1)

        # Get distance mask
        dist_mask = (sorted_dists <= d_max).int()
        dist_mask = dist_mask.gather(dim=1, index=inv_mask)

        # Get k_max mask
        kmax_mask = (inv_mask < k_max).int()

        # combine masks
        mask = dist_mask.mul(kmax_mask).float()

        return mask


class PCConv(torch.nn.Module):
    '''Parametric Continuous Convolution

    Args:
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: Tuple[int, ...],
        k_max: Optional[int] = None,
        hidden_layers: Tuple[int, ...] = (32, 64),
        d_max: float = 1.0,
        stride: Union[int, Tuple[int, ...], None] = None,
        padding: Union[str, int, Tuple[int, ...]] = 'same',
        return_coords: bool = False,
        bias: bool = True,
        weightnet: Optional[WeightNet] = None,
        weightnet_final_nl: Optional[torch.nn.Module] = None,
        feature_layer: Optional[torch.nn.Module] = None,
        lnum: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = spatial_kernel
        self.nsdims = len(self.kernel)
        self.nsphdims = 2
        self.k_max = k_max
        self.hidden_layers = hidden_layers
        self.d_max = d_max
        assert self.d_max <= math.pi / 2.0
        self.strides = self._get_strides(stride)
        self.padding = self._get_padding(padding)
        self.return_coords = return_coords
        self._shapes_set = False
        self.q_in, self.q_out, self.out_shape = 0, 0, (0, 0, 0)
        self.weight_net = self._get_weightnet(weightnet, weightnet_final_nl)
        self.feature_layer = self._get_feature_layer(feature_layer)
        self.bias = self._get_bias(bias)
        self.ang_kernel = self._get_angular_kernel_gen()
        self.lnum = lnum

    @staticmethod
    def conv_out_length(input_len: int, filter_len: int, lpad: int, rpad: int, stride: int) -> int:
        '''Determines output length of a convolution'''
        return int(((input_len - filter_len + lpad + rpad) / stride) + 1)

    def _get_strides(self, stride: Union[int, Tuple[int, ...], None]) -> Tuple[int, ...]:
        if stride is None:
            return (1, 1, 1)
        if isinstance(stride, int):
            return (stride,) * self.nsdims
        if isinstance(stride, tuple):
            if self.nsdims == len(stride):
                return stride
            raise ValueError(f'Stride ({len(stride)}D) != kernel_shape ({self.nsdims}D)')
        raise ValueError(f'Invalid stride type: {type(stride)}')

    def _get_feature_layer(self, feature_layer: Optional[torch.nn.Module]) -> torch.nn.Module:
        if feature_layer is not None:
            return feature_layer
        return torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

    def _get_padding(self, padding: Union[str, int, Tuple[int, ...]]) -> Tuple[int, ...]:
        # pylint: disable=protected-access
        if isinstance(padding, str):
            tar_padding = [0, 0] * self.nsdims
            if padding == 'valid':
                return tuple(tar_padding)
            if padding == 'same':
                if self.strides != (1,) * self.nsdims:
                    raise ValueError('Strides greater than 1 not supported with same padding.')
                for k, i in zip(self.kernel, range(self.nsdims - 1, -1, -1)):
                    total_padding = k - 1
                    left_pad = total_padding // 2
                    tar_padding[2 * i] = left_pad
                    tar_padding[2 * i + 1] = total_padding - left_pad
                return tuple(tar_padding)
            raise ValueError(f'Unrecognised padding: {padding}')
        if isinstance(padding, tuple):
            if len(padding) != self.nsdims:
                raise ValueError('padding and kernel_shape dimensions do not match.')
            return torch.nn.modules.utils._reverse_repeat_tuple(padding, 2)
        if isinstance(padding, int):
            return torch.nn.modules.utils._reverse_repeat_tuple((padding,) * self.nsdims, 2)
        raise ValueError(f'Unrecognised padding type: {type(padding)}')

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
            ndims=self.nsdims + self.nsphdims,
            final_nl=final_nl,
        )

    def _get_bias(self, bias: Union[torch.nn.Parameter, bool]) -> Optional[torch.nn.Parameter]:
        if not isinstance(bias, bool):
            return bias
        if bias:
            return torch.nn.parameter.Parameter(torch.zeros(self.out_channels))
        return None

    def _get_angular_kernel_gen(self):
        return AngularKernel(self.kernel, self.nsdims)

    def _set_runtime_shapes(
        self, ang_in: torch.Tensor, ang_out: torch.Tensor, f_in: torch.Tensor
    ) -> None:
        '''Sets shapes required for operation with tensors provided at runtime.
            Only sets once.

        Args:
            ang_in: Input angular co-ordinate points,
                shape -> (B, q_in, 3) where last dimension has
                    cartesian co-ords (x, y, z)
            ang_out: Output angular co-ordinate points,
                shape -> (B, q_out, 3)
            f_in: Input feature map.
                shape -> (B, N, q_in, C), where N are the `d` input spatial
        '''
        if self._shapes_set:
            return
        self.q_in, self.q_out = ang_in.size(1), ang_out.size(1)
        if self.k_max is None:
            self.k_max = ang_in.size(1)
        self.out_shape = self._get_output_shape(f_in)
        self._shapes_set = True

    def _get_output_shape(self, f_in: torch.Tensor) -> Tuple[int, ...]:
        '''Caluclates output shape given input shape, padding, stride & kernel size'''
        out_shape = []
        for i, filter_len in enumerate(self.kernel):
            length = self.conv_out_length(
                f_in.size(i + 1),
                filter_len,
                self.padding[-(i * 2) - 1],
                self.padding[-(i * 2) - 2],
                self.strides[i],
            )
            out_shape.append(length)
        return tuple(out_shape)

    def _expand_p_ang(self, p_ang: torch.Tensor) -> torch.Tensor:
        '''Expands p_ang to correct dimensionality to be concatenaged with p_spatial

        Args:
            p_ang: Relative output q-space co-ordinate points
                shape -> (B, q_in, q_out, nsphdims).

        Returns:
            p_ang: Reshaped p_ang to include spatial kernel dimensions
                shape -> (B, q_out, S, q_in, nsphdims)
                    where S are the `d` spatial kernel dimensions.
        '''
        _, q_in, q_out, nsphdims = p_ang.shape
        p_ang = p_ang.transpose(1, 2)  # (B, q_out, q_in, 2)
        p_ang = p_ang.reshape(-1, q_out, *(1,) * self.nsdims, q_in, nsphdims)
        exp_tuple = (-1, -1) + self.kernel + (-1, -1)
        p_ang = p_ang.expand(*exp_tuple)

        return p_ang

    def _get_p_spatial(self, p_ang: torch.Tensor) -> torch.Tensor:
        '''Creates spatial co-ordinate tensor to be appended with r_out

        Args:
            p_ang: Relative output q-space co-ordinate points
                shape -> (B, q_out, S, k_max, 2)
                    where S are the `d` spatial kernel dimensions.

        Return:
            p_spatial: Relative spatial co-ordinate points
                shape -> (B, q_out, S, k_max, d)
        '''

        p_spatial = torch.cartesian_prod(
            *(
                torch.linspace(0, 1, size).to(p_ang, non_blocking=True)
                if size != 1
                else torch.zeros(1).to(p_ang, non_blocking=True)
                for size in self.kernel
            )
        )
        p_spatial = p_spatial.view(1, 1, *self.kernel, 1, self.nsdims)  # (1, 1, S, 1, d)
        exp_tuple = (p_ang.size(0), self.q_out) + (-1,) * self.nsdims + (self.k_max, -1)
        p_spatial = p_spatial.expand(*exp_tuple)

        return p_spatial

    def _reshape_mask(self, mask: torch.Tensor) -> torch.Tensor:
        '''Reshapes indexer to then be applied to f_in

        Args:
            mask: Combined d_max & k_max mask.
                shape -> (B, q_in, q_out)

        Returns:
            mask: Combined d_max and k_max mask
                shape -> (B, M, q_out, S, q_in, C)
                    where S are the `d` spatial kernel dimensions.
        '''
        mask = mask.transpose(1, 2)
        ones = (1,) * self.nsdims
        mask = mask.view(-1, *ones, self.q_out, *ones, self.q_in, 1)
        exp_tuple = (-1,) + self.out_shape + (-1,) + self.kernel + (-1, self.in_channels)
        mask = mask.expand(*exp_tuple)

        return mask

    def _reshape_input_into_kernels(self, f_in: torch.Tensor) -> torch.Tensor:
        '''Reshapes f_in to kernel sizes at end of tensor dimensions

        Args:
            f_in: Input feature map
                shape -> (B, N+p, q_in, C) where N are the `d` spatial
                    dimensions and p is the "same" padding applied.

        Returns:
            f_in: Reshaped feature map
                shape -> (B, M, q_in, q_out, C, S)
                    where M are the `d` output spatial dimensions
        '''
        # Unfold the kernel shape
        for idx, size in enumerate(self.kernel):
            f_in = f_in.unfold(idx + 1, size, self.strides[idx])
        # Expand a_in
        exp_tuple = (-1,) * (2 + self.nsdims) + (self.q_out,) + (-1,) * (1 + self.nsdims)
        f_in = f_in.unsqueeze(2 + self.nsdims)
        f_in = f_in.expand(exp_tuple)

        return f_in

    def _permute_f_in(self, f_in: torch.Tensor) -> torch.Tensor:
        '''Permute f_in

        Args:
            f_in: Input features.
                shape -> (B, M, q_in, q_out, C, S)

        Returns:
            f_in: Indexed input features.
                shape -> (B, M, q_out, S, q_in, C)
        '''
        perm = [0, *range(1, self.nsdims + 1), self.nsdims + 2]
        perm.extend(range(f_in.ndim - self.nsdims, f_in.ndim))
        perm.extend((1 + self.nsdims, f_in.ndim - 1 - self.nsdims))
        f_in = f_in.permute(*tuple(perm))

        return f_in

    def _reshape_input_features(self, f_in: torch.Tensor) -> torch.Tensor:
        '''Reshapes `f_in` ready to be convolved with weights. This function
            works with varying spatial dimensionality.

        Args:
            f_in: Input feature map
                shape -> (B, N, q_in, C), where N are the `d` spatial
                dimensions (typically 3 though any is supported)

        Returns:
            f_in: Feature map to be convolved with weights.
                shape -> (B, M, q_out, S, q_in, C)
                    where M are the `d` output spatial dimensions
        '''

        # Pad input feature map
        f_in = torch.nn.functional.pad(f_in, (0,) * 4 + self.padding)  # type: ignore

        # Reshape for kernels
        f_in = self._reshape_input_into_kernels(f_in)
        f_in = self._permute_f_in(f_in)

        return f_in

    def _expand_conv_weights(self, weights: torch.Tensor) -> torch.Tensor:
        '''Expands weights provided by WeightNet to same shape as `f_in`

        Args:
            weights: Weights tensor provided by WeightNet
                shape -> (B, q_out, S, q_in, C)

        Returns:
            weights: Expanded weights
                shape -> (B, M, q_out, S, q_in, C)
        '''
        for _ in self.out_shape:
            weights = weights.unsqueeze(1)
        weights = weights.expand(-1, *self.out_shape, -1, *(-1,) * len(self.out_shape), -1, -1)

        return weights

    def _apply_and_sum(
        self, f_in: torch.Tensor, w_in: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        '''Applies matrix multiplication inplace, then sums over kernel dimensions

        Args:
            f_in: Features to be convolved.
                shape -> (B, M, q_out, S, q_in, C)
                    where M are the `d` output spatial dimensions.
            w_in: Weights to convolve with features.
                shape -> (B, M, q_out, S, q_in, C)
                    where M dimension is singleton, not actual M size.
            mask: Combined d_max and k_max mask
                shape -> (B, M, q_out, S, q_in, C)

        Returns:
            f_in: output features. shape -> (B, M, q_out, C)
        '''
        f_in = f_in.mul(w_in).mul(mask).sum(dim=tuple(range(-self.nsdims - 2, -1)))
        return f_in

    def _apply_feature_layer(self, f_in: torch.Tensor) -> torch.Tensor:
        '''Applies feature layer to go from in_channels -> out_channels

        Args:
            f_in: shape -> (B, M, q_out, C)

        Returns:
            f_out: shape -> (B, M, q_out, K)
        '''
        f_out = self.feature_layer(f_in)
        if self.bias is not None:
            f_out = f_out.add(self.bias)
        return f_out

    def forward(
        self, ang_in: torch.Tensor, ang_out: torch.Tensor, f_in: torch.Tensor
    ) -> torch.Tensor:
        '''Forward pass of parametric continuous convolution

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

        Returns:
            f_out: Output feature map, shape -> (B, M, q_out, S),
                where M are the `d` output spatial dimensions.
        '''
        self._set_runtime_shapes(ang_in, ang_out, f_in)

        p_ang, mask = self.ang_kernel.get_p_ang(ang_in, ang_out, self.d_max, self.k_max)
        p_ang = self._expand_p_ang(p_ang)
        p_spatial = self._get_p_spatial(p_ang)
        p = torch.cat((p_spatial, p_ang), dim=-1)

        mask = self._reshape_mask(mask)

        w_in = self.weight_net(p)

        f_in = self._reshape_input_features(f_in)
        w_in = self._expand_conv_weights(w_in)
        f_in = self._apply_and_sum(f_in, w_in, mask)
        f_out = self._apply_feature_layer(f_in)

        return f_out


class FactorisedPCConv(torch.nn.Module):
    '''Factorised Parametric Continuous Convolution'''
