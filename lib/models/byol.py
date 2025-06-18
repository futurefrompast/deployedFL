import copy
import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os

from models.modules import MetaModule, MetaSequential, MetaLinear, MetaBatchNorm1d


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class BYOL(MetaModule):
    def __init__(self, encoder, hidden_dim=4096, proj_dim=256):
        super().__init__()

        self.encoder = encoder
        self.input_dim = encoder.num_features
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        num_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        #print(f'======> Encoder: output dim {self.proj_dim} | {num_params/1e6:.3f}M parameters')

        projector_layers = [
            ('fc1', MetaLinear(self.input_dim, self.hidden_dim, bias=False)),
            ('bn1', MetaBatchNorm1d(self.hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', MetaLinear(self.hidden_dim, self.proj_dim, bias=True)),
        ]
        self.projector = MetaSequential(OrderedDict(projector_layers))

        predictor_layers = [
            ('fc1', nn.Linear(self.proj_dim, self.hidden_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden_dim, self.proj_dim, bias=True)),
        ]
        self.predictor = nn.Sequential(OrderedDict(predictor_layers))

        self.enc_tar = copy.deepcopy(self.encoder)
        self.proj_tar = copy.deepcopy(self.projector)
        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.enc_tar.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector.parameters(), self.proj_tar.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.encoder.parameters(), self.enc_tar.parameters()):
            #param_k.data.mul_(mm).add_(1. - mm, param_q.data)
            param_k.data = mm * param_k.data + (1 - mm) * param_q.data
        for param_q, param_k in zip(self.projector.parameters(), self.proj_tar.parameters()):
            #param_k.data.mul_(mm).add_(1. - mm, param_q.data)
            param_k.data = mm * param_k.data + (1 - mm) * param_q.data

    def forward(self, x, params=None, with_momentum=False):
        logits, feats = self.encoder(x, params=self.get_subdict(params, 'encoder'))

        if with_momentum:
            projs = self.projector(feats, params=self.get_subdict(params, 'projector'))
            preds = self.predictor(projs)

            with torch.no_grad():
                _, feats_tar = self.enc_tar(x, params=self.get_subdict(params, 'enc_tar'))
                projs_tar = self.proj_tar(feats_tar, params=self.get_subdict(params, 'proj_tar'))
            return logits, preds, projs_tar
        else:
            return logits
    
