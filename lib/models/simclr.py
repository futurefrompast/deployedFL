import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os

from models.modules import MetaModule

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class SimCLR(MetaModule):
    def __init__(self, encoder, hidden_dim=2048, proj_dim=128):
        super().__init__()

        self.encoder = encoder
        self.input_dim = encoder.num_features
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        num_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        #print(f'======> Encoder: output dim {self.proj_dim} | {num_params/1e6:.3f}M parameters')

        projection_layers = [
            ('fc1', nn.Linear(self.input_dim, self.hidden_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden_dim, self.proj_dim, bias=False)),
            ('bn2', BatchNorm1dNoBias(self.proj_dim)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, x, params=None):
        logits, feats = self.encoder(x, params=self.get_subdict(params, 'encoder'))
        z = self.projection(feats)
        return logits, z
