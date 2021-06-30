import typing

import math
import torch
import torch.nn as nn

from matchzoo.modules.conv_tbc import ConvTBC, Conv
from matchzoo.modules.utils import generate_seq_mask


class ConvEncoder(nn.Module):
    """Convolutional 1D Encoder
    """

    def __init__(
            self,
            input_size: int,
            kernel_size: typing.List[int],
            kernel_num: typing.List[int],
            activation: str = 'ReLU',
            padding_index: int = 0):
        super().__init__()

        self.conv_layer = nn.ModuleList()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.activation = activation
        activation_class: nn.Module = getattr(nn, activation, None)

        for ks in self.kernel_size:
            modules: typing.Tuple[nn.Module] = (
                Conv(input_size, kernel_num, ks, padding=math.floor(ks / 2)),
            )
            if activation_class:
                modules = modules + (activation_class(),)
            self.conv_layer.append(
                nn.Sequential(*modules)
            )
    
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor]:
        """Forward N-gram Conv 1D

        Args:
            input (torch.Tensor): the input sequence tensor with [B, T, C]
            input_length (torch.Tensor): the input sequence tensor length with [B, 1]

        Returns:
            typing.List[torch.Tensor]: the n-gram results
        """
        # mask
        unpadding_mask = generate_seq_mask(input_length, max_length=input.size(1))

        # input = input.transpose(0, 1)
        mask = unpadding_mask.unsqueeze(-1)
        input_convs = []
        for layer in self.conv_layer:
            convs = layer(input)
            convs = convs * mask
            input_convs.append(convs)

        return input_convs, unpadding_mask
