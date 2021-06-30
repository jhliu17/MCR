# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.modules.utils import _single


class ConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)
    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.

    conv_tbc is the same as torch.nn.Conv1d, but accepts a different input shape.

    The input shape for nn.Conv1d is batch x channels x time (BCT), which would 
    require a transpose since the rest of the network operates with time x batch x channel (TBC) tensors. 
    conv_tbc takes time x batch x channel (TBC) input directly.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)

        self.weight = torch.nn.Parameter(
            torch.Tensor(self.kernel_size[0], in_channels, out_channels)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(
            input.contiguous(), self.weight, self.bias, self.padding[0]
        )

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)
    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.

    conv_tbc is the same as torch.nn.Conv1d, but accepts a different input shape.

    The input shape for nn.Conv1d is batch x channels x time (BCT), which would 
    require a transpose since the rest of the network operates with time x batch x channel (TBC) tensors. 
    conv_tbc takes time x batch x channel (TBC) input directly.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.conv(input)
        input = input.transpose(1, 2)
        return input.contiguous()

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
