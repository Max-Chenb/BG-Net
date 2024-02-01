'''
# The following code snippet is derived from [LoFTR: Detector-Free Local Feature Matching with Transformers] by [Jiaming Sun, Zehong Shen, Yu'ang Wang, Hujun Bao, Xiaowei Zhou]
# Original URL [https://github.com/zju3dv/LoFTR]
'''

import copy
import torch
import torch.nn as nn
from linear_attention import LinearAttention, FullAttention


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        bs = x.size(0)
        query, key, value = x, source, source

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))
        message = self.norm1(message)

        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):

    def __init__(self, d_model = 256,nhead = 8,layer_names = ['self', 'cross'] * 1,attention = 'linear'):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = EncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1):

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0)
                feat1 = layer(feat1, feat1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
            else:
                raise KeyError

        return feat0, feat1
