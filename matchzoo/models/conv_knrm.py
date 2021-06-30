"""An implementation of ConvKNRM Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.modules import GaussianKernel
from matchzoo.modules import EmbeddingLayer
from matchzoo.pipeline import RHPPipeline


class ConvKNRM(nn.Module):
    def __init__(self, config, pipeline: RHPPipeline, stage: str):
        super().__init__()

        self.config = config
        q_vocab = pipeline.prd_text_field.vocab
        d_vocab = pipeline.rvw_text_field.vocab
        self.use_crossmatch = config.cross_match.use_crossmatch

        self.q_embedding = EmbeddingLayer(
            vocab_map=q_vocab.v2i,
            embedding_dim=config.prd_txt_encoder.embedding.embed_dim,
            vocab_name="q_vocab",
            dropout=config.prd_txt_encoder.embedding.dropout,
            embed_type=config.prd_txt_encoder.embedding.embed_type,
            padding_index=q_vocab.pad_index,
            pretrained_dir=config.prd_txt_encoder.embedding.pretrained_file,
            stage=stage,
            initial_type=config.prd_txt_encoder.embedding.init_type
        )
        self.d_embedding = EmbeddingLayer(
            vocab_map=d_vocab.v2i,
            embedding_dim=config.rvw_txt_encoder.embedding.embed_dim,
            vocab_name="d_vocab",
            dropout=config.rvw_txt_encoder.embedding.dropout,
            embed_type=config.rvw_txt_encoder.embedding.embed_type,
            padding_index=d_vocab.pad_index,
            pretrained_dir=config.rvw_txt_encoder.embedding.pretrained_file,
            stage=stage,
            initial_type=config.rvw_txt_encoder.embedding.init_type
        )

        self.q_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
        self.cnn_num = len(config.prd_txt_encoder.encoder.kernel_size)
        for i in config.prd_txt_encoder.encoder.kernel_size:
            conv = nn.Sequential(
                nn.ConstantPad1d((0, i), 0),
                nn.Conv1d(
                    in_channels=config.prd_txt_encoder.embedding.embed_dim,
                    out_channels=config.prd_txt_encoder.encoder.hidden_dimension,
                    kernel_size=i + 1
                ),
                nn.Tanh()
            )
            self.q_convs.append(conv)
            self.d_convs.append(conv)

        self.kernels = nn.ModuleList()
        self.kernel_num = config.gausian_kernel.kernel_num
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (self.kernel_num - 1) - 1.0
            sigma = config.gausian_kernel.sigma
            if mu > 1.0:
                sigma = config.gausian_kernel.exact_sigma
                mu = 1.0
            self.kernels.append(GaussianKernel(mu=mu, sigma=sigma))

        dim = self.cnn_num ** 2 * self.kernel_num
        self.out = nn.Linear(dim, 1)

    def forward(self, inputs):
        """Forward."""

        query, doc = inputs['text_left'], inputs['text_right']

        q_embed = self.q_embedding(query.long()).transpose(1, 2)
        d_embed = self.d_embedding(doc.long()).transpose(1, 2)

        q_convs = []
        d_convs = []
        for q_conv, d_conv in zip(self.q_convs, self.d_convs):
            q_convs.append(q_conv(q_embed).transpose(1, 2))
            d_convs.append(d_conv(d_embed).transpose(1, 2))

        KM = []
        for qi in range(self.cnn_num):
            for di in range(self.cnn_num):
                # do not match n-gram with different length if use crossmatch
                if not self.use_crossmatch and qi != di:
                    continue
                mm = torch.einsum(
                    'bld,brd->blr',
                    F.normalize(q_convs[qi], p=2, dim=-1),
                    F.normalize(d_convs[di], p=2, dim=-1)
                )
                for kernel in self.kernels:
                    K = torch.log1p(kernel(mm).sum(dim=-1)).sum(dim=-1)
                    KM.append(K)

        phi = torch.stack(KM, dim=1)

        out = self.out(phi)
        return out
