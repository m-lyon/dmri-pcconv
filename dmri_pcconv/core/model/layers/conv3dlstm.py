'''3D Convolution LSTM'''

from typing import Union, Tuple, Iterable

import torch


class Conv3DLSTMCell(torch.nn.Module):
    '''Conv3D LSTM Cell'''

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]],
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: str = 'same',
        stride: Union[int, Tuple[int, int, int]] = 1,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        '''Conv3DLSTM cell.

        Args:
            img_size: Input 3D image size
            input_dim: Input number of channels
            hidden_dim: Hidden/output number of channels
            kernel_size: Size of the convolutional kernel for both cnn and rnn.
            padding: "same" or "valid" padding
            stride: Stride of convolution kernel.
            cnn_dropout: CNN dropout rate. Default: 0.0
            rnn_dropout: RNN dropout rate. Default: 0.0
            bias: Whether or not to add the bias. Default: True
        '''
        super().__init__()
        self.input_shape = self.normalise_tuple(img_size, 3, 'img_shape')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = self.normalise_tuple(kernel_size, 3, 'kernel_size')
        self.padding = padding
        self.stride = self.normalise_tuple(stride, 3, 'stride')
        self.bias = bias

        self.out_height = self.conv_output_length(
            self.input_shape[0], self.kernel_size[0], self.padding, self.stride[0]
        )
        self.out_width = self.conv_output_length(
            self.input_shape[1], self.kernel_size[1], self.padding, self.stride[1]
        )
        self.out_depth = self.conv_output_length(
            self.input_shape[2], self.kernel_size[2], self.padding, self.stride[2]
        )

        self.input_conv = torch.nn.Conv3d(
            self.input_dim,
            4 * self.hidden_dim,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias,
        )
        self.rnn_conv = torch.nn.Conv3d(
            self.hidden_dim,
            4 * self.hidden_dim,
            self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.cnn_dropout = torch.nn.Dropout(cnn_dropout)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout)

    @staticmethod
    def conv_output_length(
        input_length: int, filter_size: int, padding: str, stride: int, dilation: int = 1
    ) -> int:
        '''Determines output length of a convolution given input length.

        Args:
            input_length: Length of input.
            filter_size: Kernel filter size .
            padding: Padding choices, one of "same", "valid", "full", "causal".
            stride: Stride for kernel.
            dilation: dilation rate of kernel.

        Returns:
            The output length.
        '''
        assert padding in {'same', 'valid', 'full', 'causal'}
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if padding in ['same', 'causal']:
            output_length = input_length
        elif padding == 'valid':
            output_length = input_length - dilated_filter_size + 1
        elif padding == 'full':
            output_length = input_length + dilated_filter_size - 1
        else:
            raise AttributeError('Unknown padding type')
        return (output_length + stride - 1) // stride

    @staticmethod
    def normalise_tuple(value: Iterable[int], n: int, name: str) -> Tuple[int, ...]:
        '''Transforms a single integer or iterable of integers into an integer tuple.

        Args:
            value: The value to validate and convert. Could an int, or any iterable of
                ints.
            n: The size of the tuple to be returned.
            name: The name of the argument being validated, e.g. "strides" or
            "kernel_size". This is only used to format error messages.

        Returns:
            A tuple of n integers.

        Raises:
            ValueError: If something else than an int/long or iterable thereof was
            passed.
        '''
        # pylint: disable=raise-missing-from
        if isinstance(value, int):
            return (value,) * n
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(
                f'The `{name}` argument must be a tuple of {n} integers. Received: {value}'
            )
        if len(value_tuple) != n:
            raise ValueError(
                f'The `{name}` argument must be a tuple of {n} integers. Received: {value}'
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f'The `{name}` argument must be a tuple of {n} integers. Received: {value} '
                    + f'including element {single_value} of type {type(single_value)}'
                )
        return value_tuple

    def forward(
        self, input_tensor: torch.Tensor, cur_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Runs forward pass'''
        # pylint: disable=invalid-name
        h_cur, c_cur = cur_state

        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)

        # separate i, f, c, o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)

        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)

        # separate i, f, c, o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)

        f = torch.sigmoid((x_f + h_f))
        i = torch.sigmoid((x_i + h_i))
        g = torch.tanh((x_c + h_c))
        o = torch.sigmoid((x_o + h_o))

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Initialises hidden state'''
        # pylint: disable=invalid-name
        h, w, d = self.out_height, self.out_width, self.out_depth

        hidden = torch.zeros(
            batch_size, self.hidden_dim, h, w, d, device=self.input_conv.weight.device
        )
        cell = torch.zeros(
            batch_size, self.hidden_dim, h, w, d, device=self.input_conv.weight.device
        )
        return (hidden, cell)


class Conv3DLSTM(torch.nn.Module):
    '''Conv3D LSTM Layer'''

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]],
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: str = 'same',
        stride: Union[int, Tuple[int, int, int]] = 1,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        bias: bool = True,
        return_sequence: bool = False,
    ) -> None:
        '''Conv3D LSTM Layer

        Args:
            img_size: Input 3D image size
            input_dim: Input number of channels
            hidden_dim: Hidden/output number of channels
            kernel_size: Size of the convolutional kernel for both cnn and rnn.
            padding: "same" or "valid" padding
            stride: Stride of convolution kernel.
            cnn_dropout: CNN dropout rate. Default: 0.0
            rnn_dropout: RNN dropout rate. Default: 0.0
            bias: Whether or not to add the bias. Default: True
            return_sequence: return output sequence or final output only
        '''
        super().__init__()
        self.return_sequence = return_sequence
        self.cell = Conv3DLSTMCell(
            img_size,
            input_dim,
            hidden_dim,
            kernel_size,
            padding,
            stride,
            cnn_dropout,
            rnn_dropout,
            bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''Runs forward pass
        Args:
            input_tensor: 6D Tensor. shape -> (b, t, c, h, w, d)

        Returns:
            output: Hidden state, shape -> (b, t, c, h, w, d).
                t = 1 unless return_sequence is True.
        '''
        batch, seq_len = input_tensor.shape[0:2]

        # Since the init is done in forward. Can send image size here
        hidden_state, cell_state = self.cell.init_hidden(batch)

        ## LSTM forward
        output_inner = []
        for time in range(seq_len):
            hidden_state, cell_state = self.cell(
                input_tensor[:, time, :, :, :], [hidden_state, cell_state]
            )
            if self.return_sequence:
                output_inner.append(hidden_state)

        if self.return_sequence:
            return torch.stack((output_inner), dim=1)
        return hidden_state
