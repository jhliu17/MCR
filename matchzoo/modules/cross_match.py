import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMatchLayer(nn.Module):
    """Cross-match layer operation

    A cross match operation to generate similarity matrix. Refer to
    Conv-KNRM.
    """

    def __init__(
            self,
            do_normalize: bool = True):
        super().__init__()
        self.do_normalize = do_normalize

    def forward(self,
                input_left: typing.List[torch.Tensor],
                input_left_unpadding_mask: torch.Tensor,
                input_right: typing.List[torch.Tensor],
                input_right_unpadding_mask: torch.Tensor) -> torch.Tensor:
        """cross-match layer forward

        Args:
            input_left (typing.List[torch.Tensor]): left sequence with [B, T, C]
            input_left_unpadding_mask (torch.Tensor): left sequence length with int value [B, T]
            input_right (typing.List[torch.Tensor]): right sequence with [B, T, C]
            input_right_unpadding_mask (torch.Tensor): right sequence length with int value [B, T]

        Returns:
            torch.Tensor: cross-match result with [B, C, left_seq_len, right_seq_len]
        """
        # do normalize
        if self.do_normalize:
            self._do_normalize(input_left)
            self._do_normalize(input_right)
        
        # masking
        mask = torch.matmul(
            input_left_unpadding_mask.unsqueeze(-1).float(), 
            input_right_unpadding_mask.unsqueeze(1).float())  # [B, T_L, T_R]

        # cross-match
        cross_match_sim = []
        for left in input_left:
            for right in input_right:
                sim = torch.matmul(left, right.transpose(1, 2))  # sim: [B, T_L, T_R]
                cross_match_sim.append(sim)
        
        # stack similarity matrix and mask padding
        cross_match = torch.stack(cross_match_sim, dim=1)
        cross_match = cross_match * mask.unsqueeze(1)
        return cross_match

    def _do_normalize(self, input: typing.List[torch.Tensor]):
        for idx, inp in enumerate(input):
            input[idx] = F.normalize(inp, p=2, dim=-1)
        return input
