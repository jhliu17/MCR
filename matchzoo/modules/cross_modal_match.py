import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.modules.attention import BidirectionalAttention


class CrossModalMatchLayer(nn.Module):
    """Cross-Modal-match layer operation

    A cross modal match operation to generate similarity matrix. 
    Refer to SCAN.
    """

    def __init__(
            self,
            left_dim: int,
            right_dim: int,
            hidden_dim: int,
            do_normalize: bool = True):
        super().__init__()
        self.left_dim = left_dim
        self.right_dim = right_dim
        self.hidden_dim = hidden_dim
        self.do_normalize = do_normalize

        self.left_fc = nn.Sequential(
            nn.Linear(left_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.right_fc = nn.Sequential(
            nn.Linear(right_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.cross_attention = BidirectionalAttention()

    def forward(self,
                input_left: torch.Tensor,
                input_left_unpadding_mask: torch.Tensor,
                input_right: torch.Tensor,
                input_right_unpadding_mask: torch.Tensor) -> torch.Tensor:
        # mapping
        input_left = self.left_fc(input_left)
        input_right = self.right_fc(input_right)

        # cross-modal-match
        match_repr = self._cross_embedding(
            input_left, input_right,
            input_left_unpadding_mask,
            input_right_unpadding_mask)
        match_repr = match_repr.unsqueeze(1)  # [B, 1, T_L]
        return match_repr

    def _cross_embedding(self,
                         input_left: torch.Tensor,
                         input_right: torch.Tensor,
                         input_left_mask: torch.Tensor,
                         input_right_mask: torch.Tensor):
        left_attn_emb, _ = self.cross_attention(
            input_left,
            input_left_mask,
            input_right,
            input_right_mask)

        # do normalize
        if self.do_normalize:
            input_left = F.normalize(input_left, p=2, dim=-1)
            left_attn_emb = F.normalize(left_attn_emb, p=2, dim=-1)

        left_match = torch.sum(input_left * left_attn_emb,
                               dim=-1) * input_left_mask  # [B, T_L]
        return left_match
