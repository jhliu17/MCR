import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU
from matchzoo.modules.gated_tanh import GatedTanh


class SelfAttentionLayer(nn.Module):
    def __init__(self, nhid):
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ReLU(True),
            Linear(64, 1)
        )

    def forward(self, inputs, index, claims, padding_mask):
        tmp = None
        nins = inputs.size(1)
        if index > -1:
            idx = torch.LongTensor([index]).to(inputs.device)
            own = torch.index_select(inputs, 1, idx)
            own = own.repeat(1, nins, 1)
            tmp = torch.cat((own, inputs), 2)
        else:
            claims = claims.unsqueeze(1)
            claims = claims.repeat(1, nins, 1)
            tmp = torch.cat((claims, inputs), 2)

        # before
        attention = self.project(tmp)
        weights = F.softmax(
            attention.squeeze(-1).masked_fill(padding_mask, float('-inf')), dim=1)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs


class BroadcastSelfAttentionLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.nhid = nhid
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ReLU(True),
            Linear(64, 1)
        )

    def forward(self, inputs, padding_mask, relation=None, claims=None):
        """the reasoning attenion modeule

        Args:
            inputs (torch.Tensor): input [B, T, F]
            relation (torch.Tensor): constructed relation [B, T_left, T_right, 2*F]
            padding_mask (torch.Tensoe): the input sequence mask
            claims (torch.Tensor, optional): the input attention claim to reduce the second. Defaults to None.

        Returns:
            torch.Tensor: Shape with [B, T_left, F] or [B, F]
        """
        tmp = None
        # construct relation
        if claims is not None:
            T = inputs.size(1)
            claims = claims.unsqueeze(1).expand(-1, T, -1)  # [B, T, F]
            tmp = torch.cat((claims, inputs), dim=-1)  # [B, T, 2*F]
        else:
            tmp = relation  # [B, T, T, 2*F]

        # before
        attention = self.project(tmp)  # [B, T, T, 1] or [B, T, 1]

        if attention.dim() == 3:
            weights = F.softmax(
                attention.squeeze(-1).masked_fill(padding_mask, float('-inf')), dim=-1)
            outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        elif attention.dim() == 4:
            T = padding_mask.size(1)
            padding_mask = padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
            weights = F.softmax(
                attention.squeeze(-1).masked_fill(padding_mask, float('-inf')), dim=-1)
            outputs = (inputs.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=2)
        else:
            raise NotImplementedError
        return outputs


class AttentionLayer(nn.Module):
    def __init__(self, nhid):
        super(AttentionLayer, self).__init__()
        self.attentions = SelfAttentionLayer(nhid=nhid * 2)

    def forward(self, inputs, padding_mask):
        T = inputs.size(1)
        outputs = torch.cat([self.attentions(inputs, i, None, padding_mask)
                             for i in range(T)], dim=1)
        outputs = outputs.view(inputs.shape)
        return outputs


class BroadcastAttentionLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.attentions = BroadcastSelfAttentionLayer(nhid=nhid * 2)

    def forward(self, inputs, padding_mask):
        T = inputs.size(1)
        inputs_left = inputs.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, F]
        inputs_right = inputs.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, F]
        inputs_cast = torch.cat((inputs_left, inputs_right), dim=-1)  # [B, T, T, 2*F]
        outputs = self.attentions(inputs, padding_mask, relation=inputs_cast)  # [B, T, F]
        return outputs


class GraphReasoning(nn.Module):
    def __init__(self, nfeat, nlayer, pool: str = "att"):
        super().__init__()

        self.nlayer = nlayer
        # self.nins = nins

        # resoning graph
        self.attentions = nn.ModuleList(
            [BroadcastAttentionLayer(nfeat) for _ in range(nlayer)])
        # self.batch_norms = [BatchNorm1d(nins) for _ in range(nlayer)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        self.pool = pool
        if pool:
            if pool == 'att':
                self.aggregate = BroadcastSelfAttentionLayer(nfeat * 2)
            self.index = torch.LongTensor([0])

    def forward(self, inputs, claims, padding_mask):
        for i in range(self.nlayer):
            inputs = self.attentions[i](inputs, padding_mask)

        if self.pool:
            if self.pool == 'att':
                inputs = self.aggregate(inputs, padding_mask, claims=claims)
            if self.pool == 'max':
                inputs = torch.max(inputs, dim=1)[0]
            if self.pool == 'mean':
                T = inputs.size(1)
                inputs = torch.sum(inputs, dim=1) / (T - torch.sum(padding_mask, dim=-1, keepdim=True))
            if self.pool == 'top':
                inputs = torch.index_select(
                    inputs, 1, self.index.to(inputs.device)).squeeze()
            if self.pool == 'sum':
                inputs = inputs.sum(dim=1)

        return inputs


class CoherentEncoder(nn.Module):
    def __init__(self,
                 img_dim,
                 txt_dim,
                 hidden_dim,
                 max_seq_len,
                 nlayer,
                 pool):
        super().__init__()

        # self.img_linear = nn.Sequential(
        #     nn.Linear(img_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim))

        # self.txt_linear = nn.Sequential(
        #     nn.Linear(txt_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim))
        
        self.item_fusion = GraphReasoning(hidden_dim, nlayer, pool=pool)

    def forward(self,
                item_txt,
                item_txt_unpadding_mask,
                item_img,
                item_img_unpadding_mask,
                claims=None):
        # item_txt = self.txt_linear(item_txt)
        # item_img = self.img_linear(item_img)
        item_fusion_input = torch.cat((item_txt, item_img), dim=1)
        item_fusion_padding_mask = torch.cat((item_txt_unpadding_mask, item_img_unpadding_mask), dim=1).eq(0)
        item_fusion = self.item_fusion(item_fusion_input, claims, item_fusion_padding_mask)
        return item_fusion


class GatedCoherentEncoder(nn.Module):
    def __init__(self,
                 img_dim,
                 txt_dim,
                 hidden_dim,
                 max_seq_len,
                 nlayer,
                 pool):
        super().__init__()

        self.img_linear = GatedTanh(img_dim, hidden_dim)
        self.txt_linear = GatedTanh(txt_dim, hidden_dim)
        self.item_fusion = GraphReasoning(hidden_dim, nlayer, pool=pool)

    def forward(self,
                item_txt,
                item_txt_unpadding_mask,
                item_img,
                item_img_unpadding_mask,
                claims=None):
        item_txt = self.txt_linear(item_txt)
        item_img = self.img_linear(item_img)
        item_fusion_input = torch.cat((item_txt, item_img), dim=1)
        item_fusion_padding_mask = torch.cat((item_txt_unpadding_mask, item_img_unpadding_mask), dim=1).eq(0)
        item_fusion = self.item_fusion(item_fusion_input, claims, item_fusion_padding_mask)
        return item_fusion
